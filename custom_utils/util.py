from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import librosa
from python_speech_features import logfbank
import torch
from tqdm import tqdm
import scipy
from scipy.io import wavfile
from av_hubert_s2s.dataset.load_data import load_video_features
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
import sklearn.metrics


class BaseModelOutputWithLatents(BaseModelOutput):
    def __init__(self, last_hidden_state, hidden_states, attentions, latents):
        super().__init__(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions)
        self.latents = latents
    
def close_clip(clip):
        try:
            clip.reader.close()
            del clip.reader
            if clip.audio != None:
                    clip.audio.reader.close_proc()
                    del clip.audio
            del clip
        except Exception as e:
            pass

def read_audio_from_video(filename):
    clip = VideoFileClip(filename)
    
    # Extract the audio
    audio = clip.audio
    fs = clip.audio.fps
    n_channels = clip.audio.nchannels
    audio_array = audio.to_soundarray()
    if n_channels > 1:
        audio_array = np.transpose(audio_array)
    
    close_clip(clip)
    return audio_array, fs, n_channels

def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")
            return None
            
def load_av(video_path, stack_order_audio=4):
        video_feats = load_video(video_path)
        wav_data, sample_rate, n_channels = read_audio_from_video(video_path)
        if n_channels > 1:
            wav_data = wav_data[0]
        if sample_rate != 16000:
            wav_data = librosa.resample(y=wav_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
        audio_feats = stacker(audio_feats, stack_order_audio)

        return video_feats, audio_feats

def sample_diag_gaussian(mu, logvar): #reparametrization trick
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_div_ngaussian(mu, log_var):

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kld_loss

def entropy(dist, epsilon):
    return -1*torch.sum(dist * torch.log(dist + epsilon), dim=-1)

def check_nan_in_model_params(model):
    for name, param in tqdm(model.named_parameters()):
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")


# IO functions for AV-HuBERT:
def extract_audio_as_numpy(video_path, target_sample_rate=16000):
    """
    Extract audio from a video file and return it as a NumPy array.

    Args:
        video_path (str): Path to the video file.
        target_sample_rate (int): Target sample rate for the audio (default: 16000 Hz).

    Returns:
        tuple: (audio_array, sample_rate) where:
            - audio_array (np.ndarray): Audio data as a NumPy array with dtype=np.float32.
            - sample_rate (int): The sample rate of the audio (default: 16000 Hz).
    """
    # Load video and extract audio
    video = VideoFileClip(video_path)
    audio = video.audio

    # Export audio as a raw numpy array
    audio_array = audio.to_soundarray()
    target_samples = int(len(audio_array) * target_sample_rate / audio.fps)
    audio_array = scipy.signal.resample(audio_array, target_samples)
    audio_array = np.mean(audio_array, axis=1) 
    audio_array = (audio_array * 32767).astype(np.int16)


    video.close()
    return audio_array, target_sample_rate


def load_feature(video_path):
    """
    Load image and audio feature
    Returns:
    video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
    """
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats

    video_feats = load_video_features(video_path) # [T, H, W, 1]
    wav_data, sample_rate = extract_audio_as_numpy(video_path)
    
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
    audio_feats = stacker(audio_feats, 4) # [T/stack_order_audio, F*stack_order_audio]

    if audio_feats is not None and video_feats is not None:
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]


    audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
    # if self.normalize and 'audio' in self.modalities:
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

    # audio_feats shape [batch, F, T]
    audio_feats = audio_feats.permute(1, 0).unsqueeze(0)

    # video_feats shape [batch, C, T, H, W]
    video_feats = video_feats.permute(3, 0, 1, 2).unsqueeze(0)

    return {"video_source": video_feats, 'audio_source': audio_feats}


class CustomLRScheduleComputerPhase1():
    def __init__(self, lr_initial, lr_final, init_epoch, zero_epoch, final_epoch):
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.init_epoch = init_epoch

        if zero_epoch >= init_epoch:
            self.zero_epoch = 0
        else:
            self.zero_epoch = zero_epoch
        self.final_epoch = final_epoch
    
    def custom_lr_schedule(self, epoch):
        if epoch < self.zero_epoch:
            return 0.0
        elif self.zero_epoch <= epoch <= self.init_epoch:
            return 1.0  # Use the initial learning rate
        elif self.init_epoch < epoch <= self.final_epoch:
            # Linearly interpolate between lr_initial and lr_final
            return (self.lr_final / self.lr_initial) + (1 - self.lr_final / self.lr_initial) * (1 - (epoch - self.init_epoch) / (self.final_epoch - self.init_epoch))
        else:
            # Keep the final learning rate constant
            return self.lr_final / self.lr_initial

class CustomLRScheduleComputer():
    def __init__(self, lr_initial, lr_final, init_epoch, final_epoch):
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.init_epoch = init_epoch
        self.final_epoch = final_epoch
    
    def custom_lr_schedule(self, epoch):
        if epoch <= self.init_epoch:
            return 1.0  # Use the initial learning rate
        elif self.init_epoch < epoch <= self.final_epoch:
            # Linearly interpolate between lr_initial and lr_final
            return (self.lr_final / self.lr_initial) + (1 - self.lr_final / self.lr_initial) * (1 - (epoch - self.init_epoch) / (self.final_epoch - self.init_epoch))
        else:
            # Keep the final learning rate constant
            return self.lr_final / self.lr_initial

def mirrored_sigmoid(x):
    x_clamped = torch.clamp(x, min=-10, max=10)
    #return 1 / (1 + torch.exp(x_clamped))
    return 1 / (1 + torch.exp(x_clamped - torch.tensor(2.1972)))

def inverse_mirrored_sigmoid(y):

    return torch.log((1 - y) / y) + torch.tensor(2.1972)

def print_params(module):

    for name, param in module.named_parameters():
        print(name, param)

def get_run_name(config):

    return config['description'] + f"a1_{config['av_hubert']['av_hubert_trainer']['alpha_1']}_a2_{config['av_hubert']['av_hubert_trainer']['alpha_2']}_b1_{config['av_hubert']['av_hubert_trainer']['beta_1']}_b2_{config['av_hubert']['av_hubert_trainer']['beta_2']}_g1_{config['av_hubert']['av_hubert_trainer']['gamma_1']}_g2_{config['av_hubert']['av_hubert_trainer']['gamma_2']}"

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr
    
    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer