import torch
from av_hubert_s2s.model.avhubert2text import AV2TextForConditionalGeneration
from privacy.privacy_module import PrivacyModule, PrivacyModuleResidual, PrivacyModuleResidualFiLM, IdentityAdaptor, AWGNAdaptorED
from custom_utils.util import BaseModelOutputWithLatents

class PrivacyModuleWrapper(torch.nn.Module):
    def __init__(self, post_extract_proj, privacy_module):
        '''
        Initialize the PrivacyModuleWrapper with a torch.nn.Sequential layer consisting of the original post_extract_proj and the PrivacyModule.

        Args:
            post_extract_proj (torch.nn.Module): The original post_extract_proj layer from the AV-HuBERT model.
            privacy_module (PrivacyModule): The PrivacyModule to be integrated.
        '''

        super().__init__()
        self.privacy_module = privacy_module
        self.post_extract_proj = post_extract_proj
    def _forward_frontend(self, input_features, attention_mask=None):
        '''
        Perform the forward pass without post_extract_proj of the model with intermediate processing.

        Args:
            input_features (tuple): A tuple containing two tensors:
                - input_features[0] (torch.Tensor): Audio input features with shape (batch_size, seq_len, feature_dim).
                - input_features[1] (torch.Tensor): Video input features with shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Processed features.
        '''
        # Split concatenated features into audio and video parts
        concatenated_features = input_features
        n_features = concatenated_features.size(-1) // 2
        audio_features = concatenated_features[:, :, :n_features]
        video_features = concatenated_features[:, :, n_features:]

        # Process the features using the PrivacyModule
        (
            transformed_audio, 
            transformed_video,
            reconstructed_audio,
            reconstructed_video,
            audio_features,
            video_features,
            mu_audio,
            log_sigma_audio,
            mu_video,
            log_sigma_video
        ) = self.privacy_module(audio_features, video_features, attention_mask=attention_mask)

        # Concatenate the transformed features for the encoder
        processed_features = torch.cat((transformed_audio, transformed_video), dim=-1)
        return processed_features
    
    def forward(self, input_features, attention_mask=None):
        '''
        Perform the forward pass of the model with intermediate processing.

        Args:
            input_features (tuple): A tuple containing two tensors:
                - input_features[0] (torch.Tensor): Audio input features with shape (batch_size, seq_len, feature_dim).
                - input_features[1] (torch.Tensor): Video input features with shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Processed features.
        '''
        # Split concatenated features into audio and video parts
        concatenated_features = input_features
        n_features = concatenated_features.size(-1) // 2
        audio_features = concatenated_features[:, :, :n_features]
        video_features = concatenated_features[:, :, n_features:]

        # Process the features using the PrivacyModule
        (
            transformed_audio, 
            transformed_video,
            reconstructed_audio,
            reconstructed_video,
            audio_features,
            video_features,
            mu_audio,
            log_sigma_audio,
            mu_video,
            log_sigma_video
        ) = self.privacy_module(audio_features, video_features, attention_mask=attention_mask)

        # Concatenate the transformed features for the encoder
        processed_features = torch.cat((transformed_audio, transformed_video), dim=-1)

        processed_features = self.post_extract_proj(processed_features)
        
        return {
        'features': processed_features,
        'latent_audio': transformed_audio,
        'latent_video': transformed_video,
        'reconstructed_audio': reconstructed_audio,
        'reconstructed_video': reconstructed_video,
        'audio_features': audio_features,
        'video_features': video_features,
        'mu_audio': mu_audio,
        'log_sigma_audio': log_sigma_audio,
        'mu_video': mu_video,
        'log_sigma_video': log_sigma_video,
    }

    


class AV2TextWrapper(torch.nn.Module):
    def __init__(self, base_model, features_audio, features_video, n_heads_audio, n_heads_video, residual_type="mult", ib_activation="tanh", adaptor_scheduler=None, power_coefficient=0.1, sensitivity=1.0):
        """
        Initialize the AV2TextWrapper with a PrivacyModule and hook integration.

        Args:
            base_model (torch.nn.Module): The base AV-HuBERT model.
            features_audio (int): Number of features for the audio modality.
            features_video (int): Number of features for the video modality.
            n_heads_audio (int): Number of attention heads for the audio modality.
            n_heads_video (int): Number of attention heads for the video modality.
        """
        super().__init__()
        self.base_model = base_model

        # Initialize PrivacyModule
        if residual_type == 'residual':
            self.processing_module = PrivacyModuleResidual(
                features_audio=features_audio,
                features_video=features_video,
                n_heads_audio=n_heads_audio,
                n_heads_video=n_heads_video,
                residual_type=residual_type,
                ib_activation=ib_activation,
                adaptor_scheduler=adaptor_scheduler
            )
        elif residual_type == 'film':
            self.processing_module = PrivacyModuleResidualFiLM(
                features_audio=features_audio,
                features_video=features_video,
                n_heads_audio=n_heads_audio,
                n_heads_video=n_heads_video,
                residual_type=residual_type,
                ib_activation=ib_activation
            )
        elif residual_type == 'identity':
            self.processing_module = IdentityAdaptor()
        
        elif residual_type == 'awgn':
            self.processing_module = AWGNAdaptorED(
                epsilon=power_coefficient,
                sensitivity=sensitivity
            )
            
        else:
            self.processing_module = PrivacyModule(
                features_audio=features_audio,
                features_video=features_video,
                n_heads_audio=n_heads_audio,
                n_heads_video=n_heads_video,
                ib_activation=ib_activation
            )

        # Register the hook to process and replace features
        self.base_model.model.encoder.post_extract_proj = PrivacyModuleWrapper(
            self.base_model.model.encoder.post_extract_proj,
            self.processing_module
        )

    def forward_frontend_contrastive(self, input_features=None, attention_mask=None):
        source = input_features
        padding_mask = attention_mask
        src_audio, src_video = source[0], source[1]
        features_audio = self.base_model.model.encoder.forward_features(src_audio, modality='audio') # features: [B, F, T]
        
        if len(src_video.shape) == 4:
            src_video = src_video.unsqueeze(1)
        features_video = self.base_model.model.encoder.forward_features(src_video, modality='video')
        features = torch.cat([features_audio, features_video], dim=1)
        features = features.transpose(1, 2)
        features = self.base_model.model.encoder.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.base_model.model.encoder.forward_padding_mask(features, padding_mask)

        features = self.base_model.model.encoder.post_extract_proj._forward_frontend(features)

        return features

    def forward_frontend(self, input_features=None, attention_mask=None):
        source = input_features
        padding_mask = attention_mask
        src_audio, src_video = source['audio'], source['video']
        features_audio = self.base_model.model.encoder.forward_features(src_audio, modality='audio') # features: [B, F, T]
        features_video = self.base_model.model.encoder.forward_features(src_video, modality='video')
        features = torch.cat([features_audio, features_video], dim=1)
        features = features.transpose(1, 2)
        features = self.base_model.model.encoder.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.base_model.model.encoder.forward_padding_mask(features, padding_mask)

        features = self.base_model.model.encoder.post_extract_proj._forward_frontend(features)

        return features
            

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        input_ids=None,
        **kwargs
    ):
        """
        Perform the forward pass of the model with intermediate processing.

        Args:
            input_features (tuple): A tuple containing two tensors:
                - input_features[0] (torch.Tensor): Audio input features with shape (batch_size, seq_len, feature_dim).
                - input_features[1] (torch.Tensor): Video input features with shape (batch_size, seq_len, feature_dim).
            attention_mask (torch.Tensor): Attention mask for the audio input features, with shape (batch_size, seq_len).
            decoder_input_ids (torch.Tensor): Input IDs for the decoder, used in teacher forcing during training, with shape (batch_size, seq_len).
            decoder_attention_mask (torch.Tensor): Attention mask for the decoder inputs, with shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Ground truth token IDs for supervised training, with shape (batch_size, seq_len).
            input_ids (torch.Tensor): Alternative input tensor for the decoder, with shape (batch_size, seq_len).
            **kwargs: Additional keyword arguments for compatibility.

        Returns:
            dict: A dictionary containing the following keys:
                - 'logits' (torch.Tensor): Predicted logits from the language model head, with shape (batch_size, seq_len, vocab_size).
                - 'reconstructed_audio' (torch.Tensor): Reconstructed audio features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
                - 'reconstructed_video' (torch.Tensor): Reconstructed video features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
                - 'audio_features' (torch.Tensor): Audio latent features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
                - 'video_features' (torch.Tensor): Video latent features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
                - 'mu_audio' (torch.Tensor): Mean (μ) of the audio latent distribution, with shape (batch_size, seq_len, feature_dim).
                - 'log_sigma_audio' (torch.Tensor): Logarithm of the standard deviation (σ) of the audio latent distribution, with shape (batch_size, seq_len, feature_dim).
                - 'mu_video' (torch.Tensor): Mean (μ) of the video latent distribution, with shape (batch_size, seq_len, feature_dim).
                - 'log_sigma_video' (torch.Tensor): Logarithm of the standard deviation (σ) of the video latent distribution, with shape (batch_size, seq_len, feature_dim).
                - 'latent_audio' (torch.Tensor): Latent audio features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
                - 'latent_video' (torch.Tensor): Latent video features from the PrivacyModule, with shape (batch_size, seq_len, feature_dim).
        """
        # Forward pass through the entire base model
        encoder_outputs = self.base_model.model.encoder(
            input_features=input_features[0],
            attention_mask=attention_mask,
            video=input_features[1],
        )

        decoder_outputs = self.base_model.model.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
        )
        
        # Compute logits from the decoder outputs
        logits = self.base_model.lm_head(decoder_outputs.last_hidden_state)
        outputs = {'logits': logits, "remaining_features": encoder_outputs.latents}
        return outputs
