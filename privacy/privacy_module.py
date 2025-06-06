import torch
from av_hubert_s2s.model.avhubert import AVHubertEncoderLayer
from transformers import Wav2Vec2Config
import torch.nn.functional as F
import math

class PrivacyModule(torch.nn.Module):
    def __init__(self, features_audio, features_video, n_heads_audio, n_heads_video, ib_activation="tanh"):
        """
        Initialize the Privacy Module with audio and video information bottlenecks.

        Args:
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio bottleneck.
            n_heads_video (int): Number of attention heads for the video bottleneck.
        """
        super().__init__()
        # Define information bottlenecks
        self.audio_transformation_layer = InformationBottleneck(features_audio, n_heads_audio, residual_type='None', activation=ib_activation)  # Audio bottleneck
        self.video_transformation_layer = InformationBottleneck(features_video, n_heads_video, residual_type='None', activation=ib_activation)  # Video bottleneck

        # Define cross-modality reconstructors
        self.audio_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_video,
            num_heads=n_heads_audio,
            output_features=features_audio
        )
        self.video_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_audio,
            num_heads=n_heads_video,
            output_features=features_video
        )

    def forward(self, audio_features, video_features, attention_mask=None):
        """
        Perform privacy-preserving transformations and cross-modal reconstruction on the input features.

        Args:
            audio_features (torch.Tensor): Input audio features of shape (batch_size, sequence_length, features_audio).
            video_features (torch.Tensor): Input video features of shape (batch_size, sequence_length, features_video).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Transformed audio features of shape (batch_size, sequence_length, features_audio).
                - Transformed video features of shape (batch_size, sequence_length, features_video).
                - Reconstructed audio features of shape (batch_size, sequence_length, features_audio).
                - Reconstructed video features of shape (batch_size, sequence_length, features_video).
        """
        # Apply information bottleneck transformations
        transformed_audio, mu_audio, log_sigma_audio = self.audio_transformation_layer(audio_features, attention_mask=attention_mask)
        transformed_video, mu_video, log_sigma_video = self.video_transformation_layer(video_features, attention_mask=attention_mask)

        reconstructed_audio = transformed_audio
        reconstructed_video = transformed_video

        return transformed_audio, transformed_video, reconstructed_audio, reconstructed_video, audio_features, video_features, mu_audio, log_sigma_audio, mu_video, log_sigma_video

class PrivacyModuleResidual(torch.nn.Module):
    def __init__(self, features_audio, features_video, n_heads_audio, n_heads_video, residual_type="mult", ib_activation="tanh", adaptor_scheduler=None):
        """
        Initialize the Privacy Module with audio and video information bottlenecks.

        Args:
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio bottleneck.
            n_heads_video (int): Number of attention heads for the video bottleneck.
        """
        super().__init__()

        config_audio = Wav2Vec2Config(hidden_size=features_audio, num_attention_heads=n_heads_audio,attention_dropout=0.1, hidden_droupout=0.1)
        self.self_attention_audio = AVHubertEncoderLayer(config_audio)

        config_video = Wav2Vec2Config(hidden_size=features_video, num_attention_heads=n_heads_video,attention_dropout=0.1, hidden_droupout=0.1)
        self.self_attention_video = AVHubertEncoderLayer(config_video)

        self.self_attention_audio2 = AVHubertEncoderLayer(config_audio)
        self.self_attention_video2 = AVHubertEncoderLayer(config_video)

        self.residual_type = residual_type
        # Define information bottlenecks
        self.audio_transformation_layer = InformationBottleneck(features_audio, n_heads_audio, activation=ib_activation, residual_type=residual_type)  # Audio bottleneck
        self.video_transformation_layer = InformationBottleneck(features_video, n_heads_video, activation=ib_activation, residual_type=residual_type)  # Video bottleneck

        # Define cross-modality reconstructors
        self.audio_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_video,
            num_heads=n_heads_audio,
            output_features=features_audio
        )
        self.video_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_audio,
            num_heads=n_heads_video,
            output_features=features_video
        )
        self.adaptor_scheduler = adaptor_scheduler

    def forward(self, audio_features, video_features, attention_mask=None):
        """
        Perform privacy-preserving transformations and cross-modal reconstruction on the input features.

        Args:
            audio_features (torch.Tensor): Input audio features of shape (batch_size, sequence_length, features_audio).
            video_features (torch.Tensor): Input video features of shape (batch_size, sequence_length, features_video).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Transformed audio features of shape (batch_size, sequence_length, features_audio).
                - Transformed video features of shape (batch_size, sequence_length, features_video).
                - Reconstructed audio features of shape (batch_size, sequence_length, features_audio).
                - Reconstructed video features of shape (batch_size, sequence_length, features_video).
        """
        # Apply information bottleneck transformations
        audio_features_weight, mu_audio, log_sigma_audio = self.audio_transformation_layer(audio_features, attention_mask=attention_mask)
        video_features_weight, mu_video, log_sigma_video = self.video_transformation_layer(video_features, attention_mask=attention_mask)

        # Apply residual connection
        if self.residual_type == "add":
            if self.adaptor_scheduler is None:
                transformed_audio = audio_features + audio_features_weight
                transformed_video = video_features + video_features_weight
            else:
                alpha = self.adaptor_scheduler.get_alpha()
                transformed_audio = audio_features + audio_features_weight
                transformed_video = video_features + video_features_weight
        elif self.residual_type == "mult":
            if self.adaptor_scheduler is None:
                transformed_audio = audio_features * audio_features_weight
                transformed_video = video_features * video_features_weight
            else:
                alpha = self.adaptor_scheduler.get_alpha()
                transformed_audio = audio_features * audio_features_weight
                transformed_video = video_features * video_features_weight

        reconstructed_audio = self.audio_reconstruction_layer(transformed_video, attention_mask=attention_mask)
        reconstructed_video = self.video_reconstruction_layer(transformed_audio, attention_mask=attention_mask)

        return transformed_audio, transformed_video, reconstructed_audio, reconstructed_video, audio_features, video_features, mu_audio, log_sigma_audio, mu_video, log_sigma_video

class PrivacyModuleResidualFiLM(torch.nn.Module):
    def __init__(self, features_audio, features_video, n_heads_audio, n_heads_video, residual_type="mult", ib_activation="tanh", adaptor_scheduler=None):
        """
        Initialize the Privacy Module with audio and video information bottlenecks.

        Args:
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio bottleneck.
            n_heads_video (int): Number of attention heads for the video bottleneck.
        """
        super().__init__()

        config_audio = Wav2Vec2Config(hidden_size=features_audio, num_attention_heads=n_heads_audio, attention_dropout=0.1, hidden_dropout=0.1)
        self.gamma_audio = AVHubertEncoderLayer(config_audio)
        self.beta_audio = AVHubertEncoderLayer(config_audio)

        config_video = Wav2Vec2Config(hidden_size=features_video, num_attention_heads=n_heads_video, attention_dropout=0.1, hidden_dropout=0.1)
        self.gamma_video = AVHubertEncoderLayer(config_video)
        self.beta_video = AVHubertEncoderLayer(config_video)

        # Define cross-modality reconstructors
        self.audio_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_video,
            num_heads=n_heads_audio,
            output_features=features_audio
        )
        self.video_reconstruction_layer = CrossModalityReconstructor(
            input_features=features_audio,
            num_heads=n_heads_video,
            output_features=features_video
        )
        self.adaptor_scheduler = adaptor_scheduler

    def forward(self, audio_features, video_features, attention_mask=None):
        """
        Perform privacy-preserving transformations and cross-modal reconstruction on the input features.

        Args:
            audio_features (torch.Tensor): Input audio features of shape (batch_size, sequence_length, features_audio).
            video_features (torch.Tensor): Input video features of shape (batch_size, sequence_length, features_video).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Transformed audio features of shape (batch_size, sequence_length, features_audio).
                - Transformed video features of shape (batch_size, sequence_length, features_video).
                - Reconstructed audio features of shape (batch_size, sequence_length, features_audio).
                - Reconstructed video features of shape (batch_size, sequence_length, features_video).
        """
        # Apply FiLM transformations
        gamma_audio = self.gamma_audio(audio_features, attention_mask=attention_mask)[0]
        beta_audio = self.beta_audio(audio_features, attention_mask=attention_mask)[0]
        gamma_video = self.gamma_video(video_features, attention_mask=attention_mask)[0]
        beta_video = self.beta_video(video_features, attention_mask=attention_mask)[0]

        transformed_audio = audio_features * gamma_audio + beta_audio
        transformed_video = video_features * gamma_video + beta_video

        if torch.isnan(transformed_audio).any():
            print("NaN in transformed audio")
        if torch.isnan(transformed_video).any():
            print("NaN in transformed video")
        reconstructed_audio = self.audio_reconstruction_layer(transformed_video, attention_mask=attention_mask)
        reconstructed_video = self.video_reconstruction_layer(transformed_audio, attention_mask=attention_mask)

        return transformed_audio, transformed_video, reconstructed_audio, reconstructed_video, audio_features, video_features, None, None, None, None
    


class IdentityAdaptor(torch.nn.Module):
    def __init__(self):
        """
        Initialize the Privacy Module with audio and video information bottlenecks.

        Args:
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio bottleneck.
            n_heads_video (int): Number of attention heads for the video bottleneck.
        """
        super().__init__()

    def forward(self, audio_features, video_features, attention_mask=None):

        return audio_features, video_features, None, None, None, None, None, None, None, None


class AWGNAdaptorED(torch.nn.Module):
    def __init__(self, delta=1e-5, epsilon=0.1, sensitivity=1.0):
        """
        AWGN-based Privacy Module using Gaussian Mechanism for (epsilon, delta)-DP.

        Args:
            delta (float): The delta parameter in (epsilon, delta)-DP.
            epsilon (float): The epsilon parameter in (epsilon, delta)-DP.
            sensitivity (float): L2-sensitivity of the function. Defaults to 1.0.
        """
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.sensitivity = sensitivity

        # Compute Gaussian noise scale
        self.noise_std = self.compute_noise_std()

    def compute_noise_std(self):
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def forward(self, audio_features, video_features, attention_mask=None):
        """
        Args:
            audio_features: Tensor of shape [B, T, D_audio]
            video_features: Tensor of shape [B, T, D_video]
            attention_mask: (optional) not used here

        Returns:
            Tuple: (transformed_audio, transformed_video, None, ..., None)
        """
        noise_audio = torch.randn_like(audio_features) * self.noise_std
        noise_video = torch.randn_like(video_features) * self.noise_std

        transformed_audio = audio_features + noise_audio
        transformed_video = video_features + noise_video

        return transformed_audio, transformed_video, None, None, None, None, None, None, None, None


class AWGNAdaptor(torch.nn.Module):
    def __init__(self, power_coefficient=0.1):
        """
        Initialize the Privacy Module with audio and video information bottlenecks.

        Args:
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio bottleneck.
            n_heads_video (int): Number of attention heads for the video bottleneck.
        """
        super().__init__()
        self.power_coefficient = power_coefficient

    def forward(self, audio_features, video_features, attention_mask=None):

        power_audio = torch.mean(audio_features ** 2, dim=(1, 2), keepdim=True)  # shape [B, 1, 1]
        power_video = torch.mean(video_features ** 2, dim=(1, 2), keepdim=True)  # shape [B, 1, 1]
        noise_audio = torch.randn_like(audio_features)
        noise_video = torch.randn_like(video_features)
        noise_audio = noise_audio / torch.sqrt(torch.mean(noise_audio ** 2, dim=(1, 2), keepdim=True))
        noise_video = noise_video / torch.sqrt(torch.mean(noise_video ** 2, dim=(1, 2), keepdim=True))
        transformed_audio = audio_features + noise_audio * torch.sqrt(power_audio) * self.power_coefficient
        transformed_video = video_features + noise_video * torch.sqrt(power_video) * self.power_coefficient

        return transformed_audio, transformed_video, None, None, None, None, None, None, None, None

class InformationBottleneck(torch.nn.Module):
    def __init__(self, features, num_heads=8, activation="sigmoid", residual_type="mult"):
        """
        Initialize the Information Bottleneck module with self-attention.

        Args:
            features (int): Number of input and output features.
            num_heads (int): Number of attention heads for the self-attention mechanism.
        """
        super().__init__()
        config = Wav2Vec2Config(hidden_size=features, num_attention_heads=num_heads,attention_dropout=0.1, hidden_droupout=0.1)
        self.self_attention = AVHubertEncoderLayer(config)
        self.self_attention2 = AVHubertEncoderLayer(config)
        self.mu_layer = torch.nn.Linear(features, features, bias=False)
        self.sigma_layer = torch.nn.Linear(features, features, bias=False)
        if residual_type == "mult":
            pass
        elif residual_type == "add":
            pass
        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Identity()

    def forward(self, x, attention_mask=None):
        """
        Perform the forward pass of the Information Bottleneck with self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reparameterized latent representation of shape (batch_size, sequence_length, features).
                - Mean (mu) of the latent distribution of shape (batch_size, sequence_length, features).
                - Log variance (log_sigma) of the latent distribution of shape (batch_size, sequence_length, features).
        """
        # Apply self-attention
        attn_output = self.self_attention(x, attention_mask=attention_mask)[0]
        attn_output = self.self_attention2(x, attention_mask=attention_mask)[0]
        # Compute mu and log_sigma
        mu = self.mu_layer(attn_output)
        log_sigma = F.relu(self.sigma_layer(attn_output))

        # Reparameterize
        z = reparameterize(mu, log_sigma)

        return z, mu, log_sigma
    
class SkipVariance(torch.nn.Module):
    def __init__(self, features, num_heads=8, activation="sigmoid", residual_type="mult"):
        """
        Initialize the Information Bottleneck module with self-attention.

        Args:
            features (int): Number of input and output features.
            num_heads (int): Number of attention heads for the self-attention mechanism.
        """
        super().__init__()

        config = Wav2Vec2Config(hidden_size=features, num_attention_heads=num_heads,attention_dropout=0.1, hidden_droupout=0.1)
        self.self_attention = AVHubertEncoderLayer(config)
        self.sigma_layer = torch.nn.Linear(features, features, bias=False)

        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Perform the forward pass of the Information Bottleneck with self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reparameterized latent representation of shape (batch_size, sequence_length, features).
                - Mean (mu) of the latent distribution of shape (batch_size, sequence_length, features).
                - Log variance (log_sigma) of the latent distribution of shape (batch_size, sequence_length, features).
        """
        # Apply self-attention
        attn_output, _ = self.self_attention(x, x, x)

        # Compute mu and log_sigma
        log_sigma = self.sigma_layer(attn_output)

        # Reparameterize
        z = reparameterize(torch.zeros_like(log_sigma), log_sigma)
        z = self.activation(z)
        #print(abs(z))
        return z, log_sigma

class CrossModalityReconstructor(torch.nn.Module):
    def __init__(self, input_features, num_heads, output_features):
        """
        Initialize the Cross Modality Reconstructor module.

        Args:
            input_features (int): Number of input features for the encoder.
            num_heads (int): Number of attention heads for the multi-head attention.
            output_features (int): Number of output features for the decoder.
        """
        super().__init__()

        # Multi-head attention encoder
        config = Wav2Vec2Config(hidden_size=input_features, num_attention_heads=num_heads,attention_dropout=0.1, hidden_droupout=0.1)
        self.encoder = AVHubertEncoderLayer(config)

        # Cross-modal mapping
        self.modality_map = torch.nn.Linear(input_features, output_features)

        self.modality_activation = torch.nn.ReLU()
      
        # Multi-head attention decoder
        self.decoder = AVHubertEncoderLayer(config)

    def forward(self, source_sequence, attention_mask=None):
        """
        Perform cross-modality reconstruction from the source sequence.

        Args:
            source_sequence (torch.Tensor): Source modality sequence of shape (batch_size, sequence_length, input_features).

        Returns:
            torch.Tensor: Reconstructed target sequence of shape (batch_size, sequence_length, output_features).
        """
        # Encode the source sequence
        encoded_sequence = self.encoder(source_sequence, attention_mask=attention_mask)[0]
        # Map the modalities
        encoded_sequence = self.modality_map(encoded_sequence)
        encoded_sequence = self.modality_activation(encoded_sequence)
        # Decode to reconstruct the target sequence
        reconstructed_sequence = self.decoder(encoded_sequence, attention_mask=attention_mask)[0]

        return reconstructed_sequence

def reparameterize(mu, log_sigma):
    """
    Reparameterize using the reparameterization trick.

    Args:
        mu (torch.Tensor): Mean of the latent distribution of shape (batch_size, sequence_length, features).
        log_sigma (torch.Tensor): Log variance of the latent distribution of shape (batch_size, sequence_length, features).

    Returns:
        torch.Tensor: Sampled latent variable of shape (batch_size, sequence_length, features).
    """
    std = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(std)
    return mu + eps * std

class AdaptorScheduler():
    def __init__(self, strategy="linear", warmup_steps=500, max_steps=1000):
        
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.alpha = 1.0
        self.step = 0
        
    
    def update(self):
        
        self.step += 1
        if self.step < self.warmup_steps:
            pass

        elif self.strategy == "linear":
            self.alpha = max(0, 1 - (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps))
        

    def get_alpha(self):
        return self.alpha

class AttentiveStatisticsPooling(torch.nn.Module):
    def __init__(self, input_dim=1024, attention_hidden_dim=1024):
        super(AttentiveStatisticsPooling, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, attention_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(attention_hidden_dim, 1)
        )

    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape (B, T, D) — batch of sequences
            lengths: Optional tensor of shape (B,) with sequence lengths
                     for masking (useful when sequences are padded)
        Returns:
            pooled: Tensor of shape (B, 2 * D) with [weighted mean; weighted std]
        """
        B, T, D = x.shape
        
        # Compute attention scores (B, T, 1) → squeeze to (B, T)
        attn_scores = self.attention(x).squeeze(-1)

        if lengths is not None:
            # Mask out padded elements
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Attention weights (B, T)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # Weighted mean (B, D)
        mean = torch.sum(attn_weights * x, dim=1)

        # Weighted std (B, D)
        std = torch.sqrt(torch.sum(attn_weights * (x - mean.unsqueeze(1)) ** 2, dim=1) + 1e-9)

        # Concatenate mean and std → (B, 2D)
        pooled = torch.cat([mean, std], dim=1)

        return pooled
    
class StatisticsPooling(torch.nn.Module):
    def __init__(self):
        super(StatisticsPooling, self).__init__()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, D) — batch of sequences
        Returns:
            pooled: Tensor of shape (B, 2 * D)
        """
        B, T, D = x.shape

        # Mask zero vectors (padding)
        mask = (x.abs().sum(dim=-1) > 0).float()  # (B, T)
        mask_exp = mask.unsqueeze(-1)  # (B, T, 1)

        valid_counts = mask_exp.sum(dim=1).clamp(min=1e-6)  # (B, 1)

        mean = (x * mask_exp).sum(dim=1) / valid_counts  # (B, D)
        var = ((x - mean.unsqueeze(1)) ** 2 * mask_exp).sum(dim=1) / valid_counts  # (B, D)

        std = torch.sqrt(var + 1e-9)
        pooled = torch.cat([mean, std], dim=1)  # (B, 2D)

        return pooled
        