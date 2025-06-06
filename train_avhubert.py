import torch
from torch.utils.data import DataLoader
from privacy.dataset import AVHubertDataset, AVHubertCIMDataset, AVHubertCIMColater, avhubert_collate_fn
from privacy.losses import ASRLoss, ConditionalEntropyLoss, InfoNCELoss, InfoNCELossPooling
from privacy.av_hubert import AV2TextWrapper
from privacy.trainer import AV2TextAdaLoRaTrainer
from privacy.metrics import calculate_wer, best_epoch
from privacy.privacy_module import AdaptorScheduler
from custom_utils.util import CustomLRScheduleComputer, get_run_name
from transformers import Speech2TextTokenizer
from av_hubert_s2s.model.avhubert2text import AV2TextForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import gc
import argparse
import logging
import warnings
from tqdm import tqdm
import random
import numpy as np

# You may want to suppress warnings due to package inconsistencies
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def init_seeds(seed=0):
    # Setting the seed
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train AV-HuBERT with Privacy Modules on a single GPU using CUDA")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file", default="config/feb19_very_tough2.yaml")
    return parser.parse_args()

def main():

    init_seeds(0)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading configuration file.")
    with open(args.config, "r") as f:
        config_all = yaml.safe_load(f)
    config = config_all["av_hubert"]

    run_name = get_run_name(config_all)
    log_dir = config["log_root_path"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs saved to {log_dir}.")

    logger.info("Loading pre-trained model and tokenizer.")
    model, tokenizer, adaptor_scheduler = load_main_model(device, config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["av_hubert_trainer"]["lr_phase2"])

    logger.info("Initializing datasets and dataloaders.")
    av_hubert_cim_collater = AVHubertCIMColater(max_different_samples=config["av_hubert_trainer"]["max_different_samples"])
    train_dataset = AVHubertCIMDataset(
        files=config["av_hubert_trainer"]["train_files"],
        data_dir=config["av_hubert_trainer"]["train_data"],
        labels_dir=config["av_hubert_trainer"]["train_labels"],
        tokenizer=tokenizer,
        augmentations=config["av_hubert_trainer"].get("augmentations", None),
        noise=config["av_hubert_trainer"]["noise"],
        max_different_samples=config["av_hubert_trainer"]["max_different_samples"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["av_hubert_trainer"]["batch_size"],
        shuffle=True,
        collate_fn=av_hubert_cim_collater.avhubert_cim_collate_fn,
        num_workers=config["av_hubert_trainer"]["num_workers"]
    )

    val_dataset = AVHubertDataset(
        files=config['av_hubert_trainer']['val_files'],
        data_dir=config['av_hubert_trainer']['val_data'],
        labels_dir=config['av_hubert_trainer']['val_labels'],
        tokenizer=tokenizer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=avhubert_collate_fn,
        num_workers=config['av_hubert_trainer']['num_workers']
    )

    test_dataset = AVHubertDataset(
        files=config['av_hubert_trainer']['test_files'],
        data_dir=config['av_hubert_trainer']['test_data'],
        labels_dir=config['av_hubert_trainer']['test_labels'],
        tokenizer=tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=avhubert_collate_fn,
        num_workers=config['av_hubert_trainer']['num_workers']
    )

    asr_loss_fn = ASRLoss() if config["av_hubert_trainer"]["loss_fn"] == "ce" else ASRLoss()
    contrastive_loss_fn = InfoNCELossPooling() if config['av_hubert_trainer']['residual_type'] == 'film_pool' else InfoNCELoss()

    validation_performances = {}
    for epoch in range(config["epochs"]):
        logger.info(f"Starting epoch {epoch}.")
        model.train()
        epoch_loss = 0.0

        train_loader_progress = tqdm(enumerate(train_loader), desc="Training", total=len(train_loader))
        for batch_idx, batch in train_loader_progress:
            
            optimizer.zero_grad()
            tokens = batch["tokens"].to(device)
            audio_feats, video_feats = batch["data"]
            audio_feats = audio_feats.to(device)
            video_feats = video_feats.to(device)
            attention_mask = batch["attention_mask"].to(device)
            attention_mask_token = batch["attention_mask_token"].to(device)

            num_samples = audio_feats.shape[1]
            main_audio = audio_feats[:, 0]
            main_video = video_feats[:, 0]
            pos_audio = audio_feats[:, 1]
            pos_video = video_feats[:, 1]
            neg_audio = audio_feats[:, 2:].contiguous().view(-1, audio_feats.size(-2), audio_feats.size(-1))
            neg_video = video_feats[:, 2:].contiguous().view(-1, video_feats.size(-3), video_feats.size(-2), video_feats.size(-1))
            neg_len = audio_feats[:, 2:].shape[1]

            outputs = model(input_features=(main_audio, main_video), attention_mask=attention_mask, input_ids=tokens)
            outputs_pos = model.model.forward_frontend_contrastive((pos_audio, pos_video), attention_mask=attention_mask)
            outputs_neg = model.model.forward_frontend_contrastive((neg_audio, neg_video), attention_mask=attention_mask)
            outputs_neg = outputs_neg.view(-1, neg_len, outputs_neg.shape[-2], outputs_neg.shape[-1])

            asr_loss = asr_loss_fn(outputs["logits"], tokens, target_padding_mask=attention_mask_token)
            if config['av_hubert_trainer']['residual_type'] != 'identity' and config['av_hubert_trainer']['residual_type'] != 'awgn':
                contrastive_loss_audio = contrastive_loss_fn(outputs["remaining_features"]["latent_audio"], outputs_pos[:, :, :outputs_pos.shape[-1]//2], outputs_neg[:, :, :, :outputs_neg.shape[-1]//2])
                contrastive_loss_video = contrastive_loss_fn(outputs["remaining_features"]["latent_video"], outputs_pos[:, :, outputs_pos.shape[-1]//2:], outputs_neg[:, :, :, outputs_neg.shape[-1]//2:])
            
                total_loss = (asr_loss
                            + config['av_hubert_trainer']['beta_1'] * contrastive_loss_audio
                            + config['av_hubert_trainer']['beta_1'] * contrastive_loss_video)

            else:
                total_loss = asr_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            train_loader_progress.set_postfix(total_loss=total_loss.item())

        logger.info(f"Epoch {epoch}: Train Loss: {epoch_loss / len(train_loader):.4f}")
        writer.add_scalar("Loss/Train", epoch_loss / len(train_loader), epoch)

        # Validation
        model.eval()
        total_wer = 0.0
        val_loader_progress = tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader))
        with torch.no_grad():
            for batch_idx, batch in val_loader_progress:
                tokens = batch['tokens'].to(device)
                audio_feats = batch['data'][0].to(device)
                video_feats = batch['data'][1].to(device)
                attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).to(device)
                outputs = model.model.model.base_model.generate(audio_feats, attention_mask=attention_mask, video=video_feats, max_new_tokens=config['av_hubert_trainer']['max_generation_length'])
                predicted_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ground_truth_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                wer = calculate_wer(predicted_text, ground_truth_text)
                total_wer += (wer - total_wer) / (batch_idx + 1)

        logger.info(f"Epoch {epoch}: Validation WER: {total_wer:.4f}")
        writer.add_scalar("WER/Validation", total_wer, epoch)

        validation_performances[epoch] = {'wer': total_wer}

        run_folder = os.path.join(config['checkpoint_dir'], run_name)
        os.makedirs(run_folder, exist_ok=True)
        checkpoint_path = os.path.join(run_folder, f"av_checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}.")

    # Evaluate best model
    best_epoch_idx = best_epoch(validation_performances, {'wer': 1.0})
    best_model_path = os.path.join(run_folder, f"av_checkpoint_epoch_{best_epoch_idx}.pt")
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), os.path.join(run_folder, "av_best_model.pt"))

    # Test step
    model.eval()
    total_wer = 0.0
    test_loader_progress = tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader))
    with torch.no_grad():
        for batch_idx, batch in test_loader_progress:
            tokens = batch['tokens'].to(device)
            audio_feats = batch['data'][0].to(device)
            video_feats = batch['data'][1].to(device)
            attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).to(device)
            outputs = model.model.model.base_model.generate(audio_feats, attention_mask=attention_mask, video=video_feats, max_new_tokens=config['av_hubert_trainer']['max_generation_length'])
            predicted_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ground_truth_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            wer = calculate_wer(predicted_text, ground_truth_text)
            total_wer += (wer - total_wer) / (batch_idx + 1)

    logger.info(f"Test WER: {total_wer:.4f}")
    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config_all, f)
    with open(os.path.join(run_folder, 'wer.txt'), 'w') as f:
        f.write(f"Test WER: {total_wer:.4f}\n")
    with open(os.path.join(run_folder, 'validation_performances.yaml'), 'w') as f:
        yaml.dump(validation_performances, f)

    writer.close()
    logger.info("Training and testing complete.")

def load_main_model(device, config, path="nguyenvulebinh/AV-HuBERT"):
    base_model = AV2TextForConditionalGeneration.from_pretrained(path)
    tokenizer = Speech2TextTokenizer.from_pretrained(path)
    adaptor_scheduler = AdaptorScheduler(
        strategy=config["adaptor_scheduler"]["strategy"],
        warmup_steps=config["adaptor_scheduler"]["warmup_steps"],
        max_steps=config["adaptor_scheduler"]["max_steps"]
    )
    av_model = AV2TextWrapper(
        base_model=base_model,
        features_audio=config["av_hubert_trainer"]["features_audio"],
        features_video=config["av_hubert_trainer"]["features_video"],
        n_heads_audio=config["av_hubert_trainer"]["n_heads_audio"],
        n_heads_video=config["av_hubert_trainer"]["n_heads_video"],
        residual_type=config["av_hubert_trainer"]["residual_type"],
        ib_activation=config["av_hubert_trainer"]["ib_activation"],
        adaptor_scheduler=adaptor_scheduler,
        power_coefficient=config["av_hubert_trainer"]["power_coefficient"],
        sensitivity=config["av_hubert_trainer"]["sensitivity"],
    ).to(device)
    model = AV2TextAdaLoRaTrainer(
        model=av_model,
        init_r=config["av_hubert_trainer"]["lora_init_r"],
        target_r=config["av_hubert_trainer"]["lora_target_r"],
        train_privacy_module_only=config["av_hubert_trainer"]["train_privacy_module_only"]
    ).to(device)
    return model, tokenizer, adaptor_scheduler

if __name__ == "__main__":
    main()