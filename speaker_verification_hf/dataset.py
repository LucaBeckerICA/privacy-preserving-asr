import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
#from av_hubert.avhubert.utils import load_video
from custom_utils.util import read_audio_from_video, stacker, load_video, load_av, load_feature
import librosa
from python_speech_features import logfbank
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random


class VoxCeleb2Dataset(Dataset):
    def __init__(self, train_speakers_path, transform=None, stack_order_audio=4, long_videos_file=None, bad_samples_dev=None, max_len=180):
        """
        Args:
            train_speakers_path (string): Path to the file containing video paths for training speakers.
            transform (callable, optional): Optional transform to be applied on a sample.
            stack_order_audio (int): Number of audio frames to stack.
            long_videos_file (string, optional): Path to the file containing paths of long videos to exclude.
        """
        self.transform = transform
        self.stack_order_audio = stack_order_audio
        self.max_len = max_len
        
        # Load bad videos if provided
        if bad_samples_dev is not None:
            with open(bad_samples_dev, 'r') as f:
                self.bad_samples_dev = [line.strip() for line in f]
        else:
            self.bad_samples_dev = None

        # Load video paths from train_speakers_path
        if self.bad_samples_dev is not None:
            with open(train_speakers_path, 'r') as f:
                print(f"Excluding {len(self.bad_samples_dev)} corrupted samples from training set.")
                video_paths = [line.strip() for line in f if line.strip() not in self.bad_samples_dev]
        
        else:
            with open(train_speakers_path, 'r') as f:
                video_paths = [line.strip() for line in f]
        
        # Load long videos if provided
        if long_videos_file is not None:
            with open(long_videos_file, 'r') as f:
                self.long_videos = set(line.strip() for line in f)
        else:
            self.long_videos = set()

        self.samples, self.label_dict = self._prepare_samples(video_paths)

    def _prepare_samples(self, video_paths):
        samples = []
        label_dict = {}
        label_counter = 0

        for video_path in tqdm(video_paths, desc="Preparing samples"):
            speaker_id = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
            
            if speaker_id not in label_dict:
                label_dict[speaker_id] = label_counter
                label_counter += 1
            
            if video_path not in self.long_videos:
                samples.append((speaker_id, video_path))

        # Debug:
        new_samples = []
        for sample in samples:
            if sample[1] in self.bad_samples_dev:
                pass
            else:
                new_samples += (sample,)

        return new_samples, label_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        speaker_id, video_path = self.samples[idx]
        data = load_feature(video_path)
        data = (data['video_source'], data['audio_source'])
        tmp_data_0 = data[0][0][0]
        tmp_data_1 = data[1][0].permute(1, 0)
        data = (tmp_data_0, tmp_data_1)
        label = self.label_dict[speaker_id]

        return {"data": data, "label": label}
    
class VoxCeleb2ValDataset(Dataset):
    def __init__(self, trials_file, stack_order_audio=4, long_videos_file=None, bad_samples_val=None, max_len=180):
        self.stack_order_audio = stack_order_audio
        self.max_len = max_len
        if bad_samples_val is not None:
            with open(bad_samples_val, 'r') as f:
                self.bad_samples_val = [line.strip() for line in f]
        else:
            self.bad_samples_val = None
        if long_videos_file is not None:
            with open(long_videos_file, 'r') as f:
                self.long_videos = [line.strip() for line in f]
        else:
            self.long_videos = []
        
        self.pairs = self._load_trials(trials_file)
        
    def _load_trials(self, file_path):
        pairs = []
        with open(file_path, 'r') as f:
            for line in f:

                if self.bad_samples_val is not None:
                    label, clip1, clip2 = line.strip().split(' ')
                    if clip1 in self.bad_samples_val or clip2 in self.bad_samples_val:
                        continue
                
                else:
                    label, clip1, clip2 = line.strip().split(' ')
                label = int(label)
                pairs.append((clip1, clip2, label))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clip1, clip2, label = self.pairs[idx]
        data1 = load_feature(clip1)
        data1 = (data1['video_source'], data1['audio_source'])
        tmp_data_10 = data1[0][0][0]
        tmp_data_11 = data1[1][0].permute(1, 0)
        data1 = (tmp_data_10, tmp_data_11)
        data2 =  load_feature(clip2)
        data2 = (data2['video_source'], data2['audio_source'])
        tmp_data_20 = data2[0][0][0]
        tmp_data_21 = data2[1][0].permute(1, 0)
        data2 = (tmp_data_20, tmp_data_21)
        return {"data1": data1, "data2": data2, "label": label}
    

class VoxCeleb2Collater():
    
    def __init__(self, max_len):
        self.max_len = max_len
    
    def voxceleb2_collate_fn_train(self, batch):
        """
        Collate function for VoxCeleb2 DataLoader.
        Ensures all sequences have length self.max_len:
        - Longer sequences are truncated.
        - Shorter sequences are padded with zeros.
        """

        labels = torch.tensor([sample['label'] for sample in batch])
        audio_features = [sample['data'][1].clone() for sample in batch]
        video_features = [sample['data'][0].clone().float() for sample in batch]


        # Determine the max length constraint
        max_len = self.max_len

        def adjust_length(features, target_length):
            """
            Adjusts a sequence to match target_length.
            - Truncates if too long.
            - Pads with zeros if too short.
            """
            seq_length = features.shape[0]
            
            if seq_length > target_length:
                return features[:target_length]  # Truncate
            
            elif seq_length < target_length:
                pad_shape = (target_length - seq_length,) + features.shape[1:]  # Padding shape
                return torch.cat([features, torch.zeros(pad_shape, dtype=features.dtype)], dim=0)  # Pad
            
            return features  # No change if already the right length

        # Apply length adjustment to audio and video features
        audio_features = [adjust_length(feat, max_len) for feat in audio_features]
        video_features = [adjust_length(feat, max_len) for feat in video_features]

        # Convert to tensor
        audio_padded = torch.stack(audio_features)
        video_padded = torch.stack(video_features)

        # Create attention mask (1 for real data, 0 for padded)
        attention_mask = (audio_padded != 0.0).any(dim=-1).int()

        # Permute audio to match expected shape: [B, F, T]
        audio_padded = audio_padded.permute(0, 2, 1)

        # Ensure consistency: [B, 1, T, H, W] for video
        video_padded = video_padded.unsqueeze(1)

        return {
            'data': (audio_padded, video_padded),
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def voxceleb2_collate_fn_eval(self, batch):
        """
        Collate function for evaluation dataset in VoxCeleb2.
        Ensures all sequences have length self.max_len:
        - Longer sequences are truncated.
        - Shorter sequences are padded with zeros.
        """

        labels = torch.tensor([sample["label"] for sample in batch])


        audio_1 = [sample["data1"][1].clone() for sample in batch]  # Audio from data1
        video_1 = [sample["data1"][0].clone().float() for sample in batch]  # Video from data1

        audio_2 = [sample["data2"][1].clone() for sample in batch]  # Audio from data2
        video_2 = [sample["data2"][0].clone().float() for sample in batch]  # Video from data2


        # Determine max sequence length constraint
        max_len = self.max_len

        def adjust_length(features, target_length):
            """
            Adjusts a sequence to match target_length.
            - Truncates if too long.
            - Pads with zeros if too short.
            """
            seq_length = features.shape[0]

            if seq_length > target_length:
                return features[:target_length]  # Truncate

            elif seq_length < target_length:
                pad_shape = (target_length - seq_length,) + features.shape[1:]  # Padding shape
                return torch.cat([features, torch.zeros(pad_shape, dtype=features.dtype)], dim=0)  # Pad
            
            return features  # No change if already the right length

        # Apply length adjustment
        audio_1 = [adjust_length(feat, max_len) for feat in audio_1]
        video_1 = [adjust_length(feat, max_len) for feat in video_1]

        audio_2 = [adjust_length(feat, max_len) for feat in audio_2]
        video_2 = [adjust_length(feat, max_len) for feat in video_2]

        # Convert to tensor
        audio_1 = torch.stack(audio_1)
        video_1 = torch.stack(video_1)

        audio_2 = torch.stack(audio_2)
        video_2 = torch.stack(video_2)

        # Create attention masks (1 for real data, 0 for padding)
        attention_mask_1 = (audio_1 != 0.0).any(dim=-1).int()
        attention_mask_2 = (audio_2 != 0.0).any(dim=-1).int()

        # Permute audio shape: [B, F, T]
        audio_1 = audio_1.permute(0, 2, 1)
        audio_2 = audio_2.permute(0, 2, 1)

        # Ensure video consistency: [B, 1, T, H, W]
        video_1 = video_1.unsqueeze(1)
        video_2 = video_2.unsqueeze(1)

        return {
            "data1": (audio_1, video_1),
            "data2": (audio_2, video_2),
            "attention_mask1": attention_mask_1,
            "attention_mask2": attention_mask_2,
            "labels": labels,
        }

def voxceleb2_collate_fn_old(batch):
    """
    Collate function for VoxCeleb2 DataLoader.
    
    """
    labels = torch.tensor([sample['label'] for sample in batch])
    audio_features = [torch.tensor(sample['data'][1])for sample in batch]
    video_features = [torch.tensor(sample['data'][0]).to(torch.float32) for sample in batch]
    

    # Pad audio features
    audio_padded = pad_sequence(audio_features, batch_first=True, padding_value=0.0)
    
    # Pad video features
    video_padded = pad_sequence(video_features, batch_first=True, padding_value=0.0)
    
    # Create attention masks for audio and video
    attention_mask = (audio_padded != 0.0).any(dim=-1).int()
    
    # audio_padded: [B, T, F] -> [B, F, T]
    audio_padded = audio_padded.permute(0, 2, 1)

    # Audio and video sequences should have the same length
    min_length = min(audio_padded.size(2), video_padded.size(1))
    
    # Truncate the longer sequence
    audio_padded = audio_padded[:, :, :min_length]
    video_padded = video_padded[:, :min_length]

    # video_padded: [B, T, H, W] -> [B, 1, T, H, W]
    video_padded = video_padded.unsqueeze(1)
    
    return {
        'data': (audio_padded, video_padded),  # Padded audio and video feature sequences
        'attention_mask': attention_mask, # Attention mask
        'labels': labels
    }