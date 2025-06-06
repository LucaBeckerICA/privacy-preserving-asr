import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os
from custom_utils.util import load_av, load_feature
import random


class AVHubertDataset(Dataset):
    """
    Audio-Visual Dataset for AV-HuBERT fine-tuning.
    
    Args:
        files (str): Path to the text file listing filenames (without suffixes) used in the dataset.
        data_dir (str): Directory containing video files (.mp4).
        labels_dir (str): Directory containing transcription files (.txt).
        tokenizer (object): Tokenizer used to convert transcriptions into token sequences.
        augmentations (callable, optional): A function or transformation to apply to the audio-visual data.
    """
    def __init__(self, files, data_dir, labels_dir, tokenizer, augmentations=None):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.tokenizer = tokenizer
        self.augmentations = augmentations
        
        # Load file list
        with open(files, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Preload transcriptions
        self.transcriptions = {}
        for file_id in self.file_list:
            label_path = os.path.join(self.labels_dir, f"{file_id}.txt")
            with open(label_path, 'r') as f:
                transcription = f.read().strip().lower() # has to be lower case for our tokenizer (bug) !
                self.transcriptions[file_id] = self.tokenizer(transcription, return_tensors='pt')["input_ids"]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Get a single audio-visual sample and its tokenized transcription.
        
        Returns:
            dict: A dictionary containing:
                - 'tokens': Tokenized transcription.
                - 'data': Audio-visual input tensors.
        """
        file_id = self.file_list[idx]
        
        # Load audio-visual data
        video_path = os.path.join(self.data_dir, f"{file_id}.mp4")
        data = load_feature(video_path)
        data = (data['video_source'], data['audio_source'])
        tmp_data_0 = data[0][0][0]
        tmp_data_1 = data[1][0].permute(1, 0)
        data = (tmp_data_0, tmp_data_1)
        # Apply augmentations if specified
        if self.augmentations:
            data = self.augmentations(data)
        
        # Retrieve preloaded tokenized transcription
        tokens = self.transcriptions[file_id]
        
        return {
            'tokens': tokens,
            'data': data
        }

class AVHubertCIMDataset(Dataset):

    def __init__(self, files, data_dir, labels_dir, tokenizer, augmentations=None, max_different_samples=2, noise=0.5):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.tokenizer = tokenizer
        self.dropout = UnscaledDropout(p=0.3)
        self.augmentations = augmentations
        self.max_different_samples = max_different_samples
        self.noise = noise

        # Load file list
        with open(files, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Preload transcriptions
        self.transcriptions = {}
        for file_id in self.file_list:
            label_path = os.path.join(self.labels_dir, f"{file_id}.txt")
            with open(label_path, 'r') as f:
                transcription = f.read().strip().lower()
                self.transcriptions[file_id] = self.tokenizer(transcription, return_tensors='pt')["input_ids"]
        
        # Generate dictionary with key = video_id + utterance and value = video_id + utterances where utterance is not in utterances
        self.positive_maps = {}
        self.negative_maps = {}
        for file_id in self.file_list:
            video_id = file_id.split("_")[0]
            utterance = file_id.split("_")[1]
            self.positive_maps[file_id] = [f for f in self.file_list if f.startswith(video_id) and not f.endswith(utterance)]
            self.negative_maps[file_id] = random.sample([f for f in self.file_list if not f.startswith(video_id)], self.max_different_samples)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Get a single audio-visual sample, its tokenized transcription, 
        a set of positive samples, and a set of negative samples.
        
        Returns:
            dict: A dictionary containing:
                - 'tokens': Tokenized transcription of the main sample.
                - 'data': Audio-visual input tensor of shape [samples, sequence_length, feature_dimension].
        """
        file_id = self.file_list[idx]

        # Load main sample
        video_path = os.path.join(self.data_dir, f"{file_id}.mp4")
        data = load_feature(video_path)  # Assuming this function loads the AV features

        # Extract video and audio features for the main sample
        data = (data['video_source'], data['audio_source'])
        tmp_data_0 = data[0][0][0]  # Extract video features
        tmp_data_1 = data[1][0].permute(1, 0)  # Extract audio features
        main_sample = (tmp_data_0, tmp_data_1)

        # Extract video and audio features for cloned data_pos
        data_pos = load_feature(video_path)
        data_pos = (data_pos['video_source'], data_pos['audio_source'])
        tmp_data_pos_0 = data_pos[0][0][0]
        tmp_data_pos_1 = data_pos[1][0].permute(1, 0)
        positive_sample = (tmp_data_pos_0, tmp_data_pos_1)

        # Apply augmentations if specified
        if self.augmentations:
            main_sample = self.augmentations(main_sample)
            positive_sample = self.augmentations(positive_sample)
        
        # Apply dropout and noise
        if self.noise > 0.0:
            # Compute power of the original tensors
            power_0 = torch.mean(positive_sample[0] ** 2)  # Power of Tensor_0
            power_1 = torch.mean(positive_sample[1] ** 2)  # Power of Tensor_1
            
            # Define noise power (0.5 * original power)
            noise_std_0 = torch.sqrt(self.noise * power_0)
            noise_std_1 = torch.sqrt(self.noise * power_1)
            
            # Generate Gaussian noise with the computed standard deviation
            noise_0 = torch.randn_like(positive_sample[0]) * noise_std_0
            noise_1 = torch.randn_like(positive_sample[1]) * noise_std_1
            
            positive_sample = (positive_sample[0] + noise_0, positive_sample[1] + noise_1)
        
        # Apply dropout to the positive sample
        mask_tensor = torch.ones(positive_sample[1].shape[0], 1)
        mask_tensor = self.dropout(mask_tensor)
        positive_sample = (positive_sample[0] * mask_tensor[:, None], positive_sample[1] * mask_tensor)

        # Retrieve preloaded tokenized transcription
        tokens = self.transcriptions[file_id]

        # Load negative samples
        negative_files = self.positive_maps.get(file_id, [])[:self.max_different_samples]
        negative_samples = []
        for neg_file in negative_files:
            video_path = os.path.join(self.data_dir, f"{neg_file}.mp4")
            neg_data = load_feature(video_path)
            neg_data = (neg_data['video_source'], neg_data['audio_source'])
            neg_tmp_data_0 = neg_data[0][0][0]
            neg_tmp_data_1 = neg_data[1][0].permute(1, 0)
            negative_samples.append((neg_tmp_data_0, neg_tmp_data_1))

        # Stack all samples to match shape [samples, sequence_length, feature_dimension]
        all_samples = [main_sample] + [positive_sample] + negative_samples

        # Ensure samples are correctly shaped for batching
        video_tensors = [s[0] for s in all_samples]  # [Samples, Seq_Len, Feat_Dim]
        audio_tensors = [s[1] for s in all_samples]  # [Samples, Seq_Len, Feat_Dim]

        return {
            'tokens': tokens,
            'data': (video_tensors, audio_tensors)  # Final shape: [Samples, Seq_Len, Feat_Dim]
        }

def avhubert_collate_fn(batch):
    """
    Collate function for AV-HuBERT DataLoader.
    
    Args:
        batch (list): A list of dataset samples, where each sample is a dict:
                      {'tokens': tensor, 'data': (audio_features, video_features)}
    
    Returns:
        dict: A dictionary containing padded tensors:
            - 'tokens': Padded token sequences.
            - 'audio_features': Padded audio feature sequences.
            - 'video_features': Padded video feature sequences.
            - 'attention_mask': Attention masks for padded sequences.
    """
    tokens = [sample['tokens'] for sample in batch]
    audio_features = [torch.tensor(sample['data'][1]) for sample in batch]  # Assuming tuple (video, audio)
    video_features = [torch.tensor(sample['data'][0]).to(torch.float32) for sample in batch]
    
    # Pad tokens to the longest token sequence
    tokens = [token.squeeze() for token in tokens]
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    # Get padding masks
    attention_mask_token = (tokens_padded != 0).int()
    
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
        'tokens': tokens_padded,  # Padded token sequences
        'data': (audio_padded, video_padded),  # Padded audio and video feature sequences
        'attention_mask': attention_mask, # Attention mask
        'attention_mask_token': attention_mask_token, # Attention mask for token
    }

class AVHubertCIMColater():
    def __init__(self, max_different_samples=2):
        self.max_different_samples = max_different_samples
    
    def avhubert_cim_collate_fn(self, batch):
        """
        Collate function for AV-HuBERT DataLoader with contrastive learning setup.
        
        Args:
            batch (list): A list of dataset samples, where each sample is a dict:
                        {'tokens': tensor, 'data': (audio_features, video_features)}
                        where:
                        - 'data' is a tuple of stacked tensors: (video_features, audio_features)
                        - Shape of video_features/audio_features: [Samples, Seq_Len, Feat_Dim]

        Returns:
            dict: A dictionary containing padded tensors:
                - 'tokens': Padded token sequences.
                - 'audio_features': Padded audio feature sequences [B, Samples, Feat_Dim, Seq_Len].
                - 'video_features': Padded video feature sequences [B, Samples, 1, Seq_Len, H, W].
                - 'attention_mask': Attention masks for padded sequences.
        """
        tokens = [sample['tokens'] for sample in batch]
        
        # Extract and convert to tensor
        audio_features = [sample['data'][1] for sample in batch]  # Shape: [B, Samples, T, D]
        video_features = [sample['data'][0] for sample in batch]  # Shape: [B, Samples, T, D]
        
        # Pad tokens (only for main sample)
        tokens = [token.squeeze() for token in tokens]
        tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

        # Get padding masks for tokens
        attention_mask_token = (tokens_padded != 0).int()

        audio_features_padded_ = [pad_sequence(af, batch_first=True, padding_value=0.0) for af in audio_features]
        video_features_padded_ = [pad_sequence(vf, batch_first=True, padding_value=0.0) for vf in video_features]
        
        # For batched data
        audio_shapes = [af.shape for af in audio_features_padded_]
        video_shapes = [vf.shape for vf in video_features_padded_]
        audio_max_len = max([af.shape[1] for af in audio_features_padded_])
        video_max_len = max([vf.shape[1] for vf in video_features_padded_])

        audio_features_padded_ = [F.pad(af, (0, 0, 0, audio_max_len - af.shape[1])) for af in audio_features_padded_]
        video_features_padded_ = [F.pad(vf, (0, 0, 0, 0, 0, video_max_len - vf.shape[1])) for vf in video_features_padded_]

        for idx, (audio_feature_, video_feature_) in enumerate(zip(audio_features_padded_, video_features_padded_)):
            if audio_feature_.shape[0] < self.max_different_samples + 2:
                
                # Repeat the last sample to match the max_different_samples
                missing_items = self.max_different_samples + 2 - audio_feature_.shape[0]
                if audio_feature_.shape[0] <= 2:
                    for _ in range(missing_items):
                        audio_feature_ = torch.cat([audio_feature_, torch.zeros([1, audio_feature_.shape[1], audio_feature_.shape[2]]).to(audio_feature_.device)], dim=0)
                        video_feature_ = torch.cat([video_feature_, torch.zeros([1, video_feature_.shape[1], video_feature_.shape[2], video_feature_.shape[3]]).to(video_feature_.device)], dim=0)
                        audio_features_padded_[idx] = audio_feature_
                        video_features_padded_[idx] = video_feature_
                else:
                    for _ in range(missing_items):
                        audio_feature_ = torch.cat([audio_feature_, audio_feature_[-1:].detach().clone()], dim=0)
                        video_feature_ = torch.cat([video_feature_, video_feature_[-1:].detach().clone()], dim=0)
                        audio_features_padded_[idx] = audio_feature_
                        video_features_padded_[idx] = video_feature_

        audio_features_padded = torch.stack(audio_features_padded_)
        video_features_padded = torch.stack(video_features_padded_)
        audio_padded = audio_features_padded.permute(0, 1, 3, 2)  # [B, Samples, Feat_Dim, Seq_Len]
        video_padded = video_features_padded.unsqueeze(2)  # [B, Samples, 1, Seq_Len, Feat_Dim]

        # Create attention mask for valid (non-zero) positions
        attention_mask = (audio_padded != 0.0).any(dim=-1).int()
        
        return {
            'tokens': tokens_padded,  # Padded token sequences
            'data': (audio_padded, video_padded),  # Padded audio & video features
            'attention_mask': attention_mask,  # Attention mask
            'attention_mask_token': attention_mask_token,  # Token mask
        }

class UnscaledDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(UnscaledDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:  # Only apply dropout during training
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask  # No scaling applied
        return x  # No dropout during evaluation