import sys
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from speaker_verification_hf.dataset import VoxCeleb2Dataset, VoxCeleb2ValDataset, VoxCeleb2Collater
from speaker_verification_hf.layers import ASR2SPVProjection
from speaker_verification_hf.ecapa import ECAPA_TDNN
from speaker_verification.preparation import get_num_classes
from RJCAforSpeakerVerification.trainer import FusionModel
from RJCAforSpeakerVerification.loss import AAMsoftmax
from RJCAforSpeakerVerification.tools import *
from transformers import Speech2TextTokenizer
from av_hubert_s2s.model.avhubert2text import AV2TextForConditionalGeneration
from privacy.av_hubert import AV2TextWrapper
from privacy.trainer import AV2TextAdaLoRaTrainer
from privacy.privacy_module import AdaptorScheduler
from custom_utils.util import check_nan_in_model_params, get_run_name, compute_eer

class Trainer(nn.Module):
    def __init__(
            self,
            ckpt_path, 
            lr, 
            lr_step, 
            lr_gamma,
            lr_decay,
            lr_decay_every,
            lr_decay_rate,
            n_class,
            margin_a,
            scale_a,
            margin_v,
            scale_v,
            test_step,
            lr_decay_start,
            vox_path,
            metadata_file,
            train_speakers_file,
            batch_size,
            val_trials_file,
            test_trials_file,
            stack_order_audio,
            config,
            run_folder,
            proj_in=1024,
            proj_out_audio=512,
            proj_out_video=512,
            long_videos_file=None,
            max_sequence_len=180,
            log_dir=None,
            mode="run",
            dataloader_workers=0,
            pin_memory=False,
            h_conv=768,
            bad_samples_dev=None
    ):
        super(Trainer, self).__init__()
        self.ckpt_path = ckpt_path
        self.run_folder = run_folder
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.lr_decay = lr_decay
        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        self.n_class = n_class
        self.margin_a = margin_a
        self.scale_a = scale_a
        self.margin_v = margin_v
        self.scale_v = scale_v
        self.test_step = test_step
        self.lr_decay_start = lr_decay_start
        self.vox_path = vox_path
        self.metadata_file = metadata_file
        self.train_speakers_file = train_speakers_file
        self.batch_size = batch_size
        self.val_trials_file = val_trials_file
        self.test_trials_file = test_trials_file
        self.stack_order_audio = stack_order_audio
        self.proj_in = proj_in
        self.proj_out_audio = proj_out_audio
        self.proj_out_video = proj_out_video
        self.long_videos_file = long_videos_file
        self.max_sequence_len = max_sequence_len
        self.log_dir = log_dir
        self.mode = mode
        self.dataloader_workers = dataloader_workers
        self.pin_memory = pin_memory
        self.h_conv = h_conv
        self.av_model = self.get_av_frontends_hf(config)#.cuda()  # Use AV-HuBERT model
        if h_conv != 0:
            self.ecapa_encoder = ECAPA_TDNN(h_conv=h_conv).cuda()
        else:
            self.ecapa_encoder = torch.nn.Identity().cuda()
        self.speaker_loss = AAMsoftmax(n_class=get_num_classes(os.path.join(vox_path, train_speakers_file)), m=margin_a, s=scale_a, c=192).cuda()
        self.speaker_face_loss = AAMsoftmax(n_class=get_num_classes(os.path.join(vox_path, train_speakers_file)), m=margin_v, s=scale_v, c=512).cuda()
        self.asr2spv_proj_audio = ASR2SPVProjection(proj_in, proj_out_audio).cuda()
        self.asr2spv_proj_video = ASR2SPVProjection(proj_in, proj_out_video).cuda()
        self.fusion_model = FusionModel(max_sequence_len=max_sequence_len).cuda()
        
        self.av_model = self.av_model.cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.pin_memory = pin_memory
        self.mode = mode
        self.vox_path = vox_path

        vox_path_dev = os.path.join(vox_path, 'vox2_mp4', 'dev', 'mp4')
        vox_path_test = os.path.join(vox_path, 'vox2_mp4', 'test', 'mp4')
        metadata_file = os.path.join(vox_path, metadata_file)
        train_speakers_file = os.path.join(vox_path, train_speakers_file)
        val_trials_file = os.path.join(vox_path, val_trials_file)
        test_trials_file = os.path.join(vox_path, test_trials_file)
        long_videos_file = os.path.join(vox_path, long_videos_file)

        train_set = VoxCeleb2Dataset(train_speakers_path=train_speakers_file, transform=None, stack_order_audio=stack_order_audio, long_videos_file=long_videos_file, max_len=max_sequence_len, bad_samples_dev=bad_samples_dev)
        val_set = VoxCeleb2ValDataset(trials_file=val_trials_file, stack_order_audio=stack_order_audio, long_videos_file=long_videos_file, max_len=max_sequence_len)
        test_set = VoxCeleb2ValDataset(trials_file=test_trials_file, stack_order_audio=stack_order_audio, max_len=max_sequence_len)
 

        collater = VoxCeleb2Collater(max_len=max_sequence_len)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collater.voxceleb2_collate_fn_train, num_workers=dataloader_workers, pin_memory=pin_memory, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=1, collate_fn=collater.voxceleb2_collate_fn_eval, num_workers=dataloader_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(test_set, batch_size=1, collate_fn=collater.voxceleb2_collate_fn_eval, num_workers=dataloader_workers, pin_memory=pin_memory)

    def train_network(self, epoch):

        self.av_model.eval()
        self.speaker_loss.eval()
        self.speaker_face_loss.train()
        self.asr2spv_proj_audio.train()
        self.asr2spv_proj_video.train()
        self.fusion_model.train()

        loss = 0
        time_start = time.time()

        train_loader_progress = tqdm(enumerate(self.train_loader), desc="Training", leave=True, total=len(self.train_loader))
        for num, batch in train_loader_progress:
            if self.mode in ["debug", "debug_train"] and num > 2:
                break

            self.zero_grad()
            face = batch['data'][1].cuda()
            speech = batch['data'][0].cuda()
            labels = batch['labels'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            with torch.no_grad():
                input_features = {'audio': speech, 'video': face}
                av_embeddings = self.av_model.model.forward_frontend(input_features, attention_mask)  # New AV-HuBERT frontend call
            a_embeddings, v_embeddings = torch.split(av_embeddings, av_embeddings.shape[-1] // 2, dim=-1)
            
            a_embeddings = a_embeddings.permute(0, 2, 1)
            a_embeddings = self.ecapa_encoder(a_embeddings)
            a_embeddings = a_embeddings.permute(0, 2, 1)
            a_embeddings = self.asr2spv_proj_audio(a_embeddings)
            v_embeddings = self.asr2spv_proj_video(v_embeddings)

            AV_embeddings = self.fusion_model(a_embeddings, v_embeddings)

            AVloss, _ = self.speaker_face_loss.forward(AV_embeddings, labels)

            if torch.isnan(AVloss).any():
                AVloss = torch.nan_to_num(AVloss)

            AVloss.backward()
            self.optim.step()

            loss += (AVloss.detach().item() - loss) / (num + 1)
            train_loader_progress.set_postfix(loss=loss)

        avg_loss = loss
        return avg_loss

    def eval_network(self, mode="val"):
        
        """Validates the model and returns EER and MinDCF."""
        self.av_model.eval()
        self.speaker_loss.eval()
        self.speaker_face_loss.eval()
        self.asr2spv_proj_audio.eval()
        self.asr2spv_proj_video.eval()
        self.fusion_model.eval()

        eer_total = 0
        min_dcf_total = 0
        num_batches = 0

        cosine_similarity = nn.CosineSimilarity(dim=0)
        with torch.no_grad():
            
            if mode == "val":
                eval_loader_progress = tqdm(enumerate(self.val_loader), desc="Validation", leave=True, total=len(self.val_loader))
            elif mode == "test":
                eval_loader_progress = tqdm(enumerate(self.test_loader), desc="Test", leave=True, total=len(self.test_loader))
            labels = []
            embeddings = []
            for num, batch in eval_loader_progress:
                
                label = int(batch["labels"])
                data1 = batch["data1"]
                data2 = batch["data2"]
                attention_mask1 = batch["attention_mask1"]
                attention_mask2 = batch["attention_mask2"]
                labels.append(int(label))
                
                face1 = data1[1].cuda()
                speech1 = data1[0].cuda()
                face2 = data2[1].cuda()
                speech2 = data2[0].cuda()

                # Get embeddings
                input_features_1 = {'audio': speech1, 'video': face1}
                input_features_2 = {'audio': speech2, 'video': face2}
                #attention_mask = None
                
                embedding_1 = self.av_model.model.forward_frontend(input_features_1, attention_mask1)
                embedding_2 = self.av_model.model.forward_frontend(input_features_2, attention_mask2)


                embedding_1_a, embedding_1_v = torch.split(embedding_1, embedding_1.shape[-1] // 2, dim=-1)
                embedding_2_a, embedding_2_v = torch.split(embedding_2, embedding_2.shape[-1] // 2, dim=-1)

                embedding_1_a = embedding_1_a.permute(0, 2, 1)
                embedding_2_a = embedding_2_a.permute(0, 2, 1)
                embedding_1_a = self.ecapa_encoder(embedding_1_a)
                embedding_2_a = self.ecapa_encoder(embedding_2_a)

                embedding_1_a = embedding_1_a.permute(0, 2, 1)
                embedding_2_a = embedding_2_a.permute(0, 2, 1)

                embedding_1_a = self.asr2spv_proj_audio(embedding_1_a)
                embedding_1_v = self.asr2spv_proj_video(embedding_1_v)
                embedding_2_a = self.asr2spv_proj_audio(embedding_2_a)
                embedding_2_v = self.asr2spv_proj_video(embedding_2_v)

                fused_embedding_1 = self.fusion_model(embedding_1_a, embedding_1_v)
                fused_embedding_2 = self.fusion_model(embedding_2_a, embedding_2_v)
                embeddings.append([fused_embedding_1, fused_embedding_2])

            # Compute similarity and EER
            embs_progress = tqdm(enumerate(embeddings), leave=True, desc="EER Computation", total=len(embeddings))
            avg_eer = 0
            avg_min_dcf = 0
            scores = []
            for emb_idx, emb in embs_progress:
        
                e1 = emb[0][0]
                e2 = emb[1][0]

                score = cosine_similarity(e1, e2).item()
                scores.append(score)
            eer = compute_eer(labels, scores)

        
        return eer, 0
    

    def save_checkpoint(self, epoch):
        """Saves a checkpoint of the current model state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        os.makedirs(self.run_folder, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.run_folder, f'spv_checkpoint_epoch_{epoch}.pth'))
        print(f"Checkpoint saved at epoch {epoch}")
    
    
    def load_checkpoint(self, epoch):
        """Loads a checkpoint of the model state."""
        checkpoint_path = os.path.join(self.run_folder, f'spv_checkpoint_epoch_{epoch}.pth')
        print("Checkpoint at", checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} not found. Skipping loading.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:0")
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {checkpoint['epoch']}")

    def get_av_frontends_hf(self, config):
        """ Loads the AV-HuBERT model and ensures it is ready for processing audio-visual inputs. """
        base_model = AV2TextForConditionalGeneration.from_pretrained('nguyenvulebinh/AV-HuBERT')
        tokenizer = Speech2TextTokenizer.from_pretrained('nguyenvulebinh/AV-HuBERT')
        av_model = AV2TextWrapper(
            base_model=base_model,
            features_audio=config['av_hubert']['av_hubert_trainer']['features_audio'],
            features_video=config['av_hubert']['av_hubert_trainer']['features_video'],
            n_heads_audio=config['av_hubert']['av_hubert_trainer']['n_heads_audio'],
            n_heads_video=config['av_hubert']['av_hubert_trainer']['n_heads_video'],
            residual_type=config['av_hubert']['av_hubert_trainer']['residual_type'],
            ib_activation=config['av_hubert']['av_hubert_trainer']['ib_activation'],
            adaptor_scheduler=AdaptorScheduler(
                strategy=config['av_hubert']['adaptor_scheduler']['strategy'],
                warmup_steps=config['av_hubert']['adaptor_scheduler']['warmup_steps'],
                max_steps=config['av_hubert']['adaptor_scheduler']['max_steps']
            ),
            power_coefficient=config['av_hubert']['av_hubert_trainer']['power_coefficient'],
            sensitivity=config['av_hubert']['av_hubert_trainer']['sensitivity'],
        ).cuda()

        model = AV2TextAdaLoRaTrainer(
            model=av_model,
            init_r=config['av_hubert']['av_hubert_trainer']['lora_init_r'],
            target_r=config['av_hubert']['av_hubert_trainer']['lora_target_r'],
            train_privacy_module_only=config['av_hubert']['av_hubert_trainer']['train_privacy_module_only']
        ).cuda()

        best_checkpoint_path = os.path.join(config['av_hubert']['checkpoint_dir'], get_run_name(config), "av_best_model.pt")
        state_dict_orig = model.state_dict()
        state_dict_loaded = torch.load(best_checkpoint_path, map_location=torch.device("cuda"))
        print("orig")
        for key in list(state_dict_orig.keys())[:10]:  # Print only the first 10 keys
            print(f"{key}: {state_dict_orig[key].shape}")
        print("loaded")
        for key in list(state_dict_loaded.keys())[:10]:  # Print only the first 10 keys
            print(f"{key}: {state_dict_loaded[key].shape}")
        
        new_state_dict = {}
        for key in state_dict_loaded.keys():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            new_state_dict[new_key] = state_dict_loaded[key]
        
        model.load_state_dict(new_state_dict)
        model.eval()

        return model
