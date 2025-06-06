import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_utils.util import mirrored_sigmoid
from privacy.privacy_module import StatisticsPooling

class ASRLoss(nn.Module):
    """
    Automatic Speech Recognition (ASR) loss for AV-HuBERT.
    This loss function computes the cross-entropy loss between the predicted token logits
    and the ground-truth token indices.
    
    Args:
        ignore_index (int): The index to ignore in the loss calculation, typically used for padding tokens.
    """
    def __init__(self, ignore_index=-100):
        super(ASRLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits, labels, target_padding_mask=None):
        """
        Compute the ASR loss.
        
        Args:
            logits (torch.Tensor): Predicted token logits of shape (B, T, V),
                                   where B is batch size, T is sequence length, and V is vocabulary size.
            labels (torch.Tensor): Ground-truth token indices of shape (B, T).
                                   Labels should have the same sequence length as logits.
            target_padding_mask (torch.Tensor): Stupid placeholder variable. Not used.
        
        Returns:
            torch.Tensor: Computed ASR loss.
        """

        logits = logits[..., :-1, :].contiguous()  # Remove the last token (blank token)
        labels = labels[..., 1:].contiguous()  # Remove the first token (blank token)

        # Reshape logits and labels for CrossEntropyLoss
        logits = logits.view(-1, logits.size(-1))  # Flatten (B*T, V)
        labels = labels.view(-1)  # Flatten (B*T)
        
        loss = self.loss_fn(logits, labels)
        return loss
    

class ASRCTCLoss(nn.Module):
    """
    Connectionist Temporal Classification (CTC) loss for ASR.
    Computes the CTC loss between the predicted log probabilities and the target sequences.

    Args:
        blank (int): Index of the blank token. Default is 0.
    """
    def __init__(self, blank=0):
        super(ASRCTCLoss, self).__init__()
        self.blank = blank
        self.loss_fn = nn.CTCLoss(blank=self.blank, zero_infinity=True)

    def forward(self, logits, targets, target_padding_mask):
        """
        Compute the CTC loss.
        
        Args:
            logits (torch.Tensor): Predicted token logits of shape (B, T, V),
                                   where B is batch size, T is sequence length, and V is vocabulary size.
            targets (torch.Tensor): Ground-truth token indices (flattened) of shape (sum(target_lengths)).
            target_padding_mask (torch.Tensor): Boolean mask of shape (B, T), 
                                                where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            torch.Tensor: Computed CTC loss.
        """

        logits = logits[..., :-1, :].contiguous()  # Remove the last token (blank token)
        targets = targets[..., 1:].contiguous()  # Remove the first token (blank token)
        
        # Derive log_probs from logits
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, V)

        # Compute input_lengths
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device) - 1

        # Compute target lengths
        target_lengths = torch.sum(target_padding_mask, dim=1).int() - 1

        # Compute the loss
        loss = self.loss_fn(log_probs, targets, input_lengths, target_lengths)
        return loss

    
class ASRCTCAndCELoss(nn.Module):
    """
    Combined CTC and Cross-Entropy (CE) loss for ASR.
    This loss combines the CTC loss for alignment and the CE loss for token classification.

    Args:
        blank (int): Index of the blank token for CTC. Default is 0.
        ce_ignore_index (int): Index to ignore in CE loss, typically used for padding tokens. Default is -100.
        ctc_weight (float): Weight for the CTC loss component. Default is 0.5.
        ce_weight (float): Weight for the CE loss component. Default is 0.5.
    """
    def __init__(self, blank=0, ce_ignore_index=-100, ctc_weight=0.5, ce_weight=0.5):
        super(ASRCTCAndCELoss, self).__init__()
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.ctc_loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ce_ignore_index)

    def forward(self, logits, labels, target_padding_mask):
        """
        Compute the combined CTC and CE loss.
        
        Args:
            logits (torch.Tensor): Predicted token logits of shape (B, T, V),
                                   where B is batch size, T is sequence length, and V is vocabulary size.
            labels (torch.Tensor): Ground-truth token indices for CE of shape (B, T),
                                   where -100 is used for padding tokens.
            target_padding_mask (torch.Tensor): Boolean mask of shape (B, T),
                                                where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            torch.Tensor: Combined loss (CTC + CE).
        """

        logits = logits[..., :-1, :].contiguous()  # Remove the last token (blank token)
        labels = labels[..., 1:].contiguous()  # Remove the first token (blank token)

        # Derive log_probs from logits for CTC loss
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2) # (T, B, V)

        # Compute input_lengths for CTC loss
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device) - 1

        # Compute target_lengths for CTC loss
        target_lengths = torch.sum(target_padding_mask, dim=1).int() - 1
        
        # Compute CTC loss
        ctc_loss = self.ctc_loss_fn(log_probs, labels, input_lengths, target_lengths)

        # Reshape logits and labels for CE loss
        logits = logits.view(-1, logits.size(-1))  # Flatten (B*T, V)
        labels = labels.view(-1)  # Flatten (B*T)

        # Compute CE loss
        ce_loss = self.ce_loss_fn(logits, labels)

        # Weighted combination of CTC and CE losses
        combined_loss = self.ctc_weight * ctc_loss + self.ce_weight * ce_loss
        return combined_loss


class IBLoss(nn.Module):
    """
    Information Bottleneck (IB) Loss.
    This loss aims to minimize the mutual information between the input modality
    (e.g., audio or video) and its latent representation while preserving task-relevant information.
    
    Minimizes mutual information (via KL Divergence or Variational Approximation).
    
    """
    def __init__(self):
        super(IBLoss, self).__init__()

    def forward(self, prior_mean, prior_logvar):
        """
        Compute the Information Bottleneck loss.
        
        Args:
            latent (torch.Tensor): The latent representation.
            prior_mean (torch.Tensor): Mean of the prior distribution.
            prior_logvar (torch.Tensor): Log variance of the prior distribution.
        
        Returns:
            torch.Tensor: Computed IB loss.
        """
        
        # KL divergence for mutual information minimization
        kl_loss = -0.5 * torch.sum(1 + prior_logvar - prior_mean.pow(2) - prior_logvar.exp())
        kl_loss = kl_loss / (prior_logvar.size(0) * prior_logvar.size(1))  # Normalize by batch size and sequence
        
        return kl_loss

class CMREntropyLoss(nn.Module):
    """
    Cross-Modality Reconstructor (CMR) Entropy Loss.
    
    This loss focuses on **Entropy Maximization**:
    Encourages high uncertainty in the reconstructed modality to minimize predictability between latent representations.

    """
    def __init__(self, exponential_activation=True):
        super(CMREntropyLoss, self).__init__()
        self.exponential_activation = exponential_activation

    def forward(self, logits):
        """
        Compute the CMR entropy maximization loss.
        
        Args:
            logits (torch.Tensor): Logits from the reconstructor for entropy computation.
        
        Returns:
            torch.Tensor: Computed entropy maximization loss.
        """
        # Entropy maximization loss (hence no negative sign)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-4, max=1e4)
        if self.exponential_activation:
            entropy_loss = mirrored_sigmoid(torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1)))
            if torch.isnan(entropy_loss).any():
                print("CMR Entropy is NAN")
        else:
            entropy_loss = torch.log(probs.size(-1)) - torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        
        return entropy_loss
    

class GlobalLoss(nn.Module):
    """
    Global Loss for AdaLoRa and IB Optimization.
    
    Combines multiple loss components:
    - ASR Loss
    - Cross-Modality Reconstructor (CMR) Loss for both directions (video from audio and audio from video)
    - Information Bottleneck (IB) Loss for both audio and video modalities
    
    Args:
        alpha_1 (float): Weight for CMR-Loss (video reconstructed from audio).
        alpha_2 (float): Weight for CMR-Loss (audio reconstructed from video).
        beta_1 (float): Weight for IB-Loss (audio).
        beta_2 (float): Weight for IB-Loss (video).
    """
    def __init__(self, alpha_1=1.0, alpha_2=1.0, beta_1=0.1, beta_2=0.1, exponential_activation=True):
        super(GlobalLoss, self).__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.exponential_activation = exponential_activation

    def forward(self, asr_loss, cmr_loss_video_from_audio, cmr_loss_audio_from_video, coe_loss_audio, coe_loss_video):
        """
        Compute the global loss.
        
        Args:
            asr_loss (torch.Tensor): ASR loss.
            cmr_loss_video_from_audio (torch.Tensor): CMR loss for video reconstructed from audio.
            cmr_loss_audio_from_video (torch.Tensor): CMR loss for audio reconstructed from video.
            ib_loss_audio (torch.Tensor): IB loss for audio modality.
            ib_loss_video (torch.Tensor): IB loss for video modality.
        
        Returns:
            torch.Tensor: Computed global loss.
        """
        total_loss = (
            asr_loss +
            self.alpha_1 * cmr_loss_video_from_audio +
            self.alpha_2 * cmr_loss_audio_from_video +
            self.beta_1 * coe_loss_audio +
            self.beta_2 * coe_loss_video
        )
        return total_loss
    
class CMRReconstructionLoss(nn.Module):
    """
    Cross-Modality Reconstructor (CMR) Reconstruction Loss.
    
    This loss focuses on accurately reconstructing one modality's **latent representations** from another.
    It evaluates the difference between the predicted latent representation and the target latent representation.
    
    Recommended Loss Type:
    - Mean Squared Error (MSE) Loss: Suitable for continuous latent representations.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output ('mean' or 'sum').
    """
    def __init__(self, reduction='mean'):
        super(CMRReconstructionLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, predictions, targets):
        """
        Compute the reconstruction loss.
        
        Args:
            predictions (torch.Tensor): Predicted latent representations.
            targets (torch.Tensor): Ground-truth latent representations.
        
        Returns:
            torch.Tensor: Computed reconstruction loss.
        """
        loss = self.loss_fn(predictions, targets)
        return loss

class CMRGlobalLoss(nn.Module):
    """
    Global Cross-Modality Reconstructor (CMR) Reconstruction Loss.
    
    This loss weights the reconstruction loss for both audio and video modalities.
    
    Args:
        gamma_1 (str): Coefficient for audio to video reconstruction.
        gamma_2 (str): Coefficient for video to audio reconstruction.
    """
    def __init__(self, gamma_1=1.0, gamma_2=1.0):
        super(CMRGlobalLoss, self).__init__()
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def forward(self, audio_recon_loss, video_recon_loss):
        """
        Compute the global CMR reconstruction loss.
        
        Args:
            audio_recon_loss (torch.Tensor): Reconstruction loss for audio modality.
            video_recon_loss (torch.Tensor): Reconstruction loss for video modality.
        
        Returns:
            torch.Tensor: Computed global CMR reconstruction loss.
        """
        total_loss = self.gamma_1 * audio_recon_loss + self.gamma_2 * video_recon_loss
        return total_loss

class ConditionalEntropyLoss(nn.Module):
    """
    Approximates the conditional entropy H(Y|X) for a batch of data.

    Args:
        None
    """
    def __init__(self, exponential_activation=True, target=0.7):
        super(ConditionalEntropyLoss, self).__init__()
        self.exponential_activation = exponential_activation
        self.target = torch.tensor(target)

    def forward(self, x, log_var):
        """
        Compute the conditional entropy H(Y|X).
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            log_var (torch.Tensor): Log variance for the stochastic mask of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Approximated conditional entropy for the batch.
        """
        # Compute variance
        s = x.size()
        # Compute conditional entropy
        if self.exponential_activation:
            conditional_entropy = mirrored_sigmoid(torch.sum(log_var) / (s[0] * s[1] * s[2]))
            if torch.isnan(conditional_entropy).any():
                print("Conditional Entropy is NAN")
        else:
            conditional_entropy = torch.mean(torch.sum(log_var,dim=-1))

        # Return negative since we aim to maximize the entropy
        if self.exponential_activation:
            return conditional_entropy #+ 0.1 * torch.abs(conditional_entropy - self.target)
        else:
            return -1 * conditional_entropy
    
class RelativeConditionalEntropyLoss(nn.Module):
    """
    Approximates the conditional entropy H(Y|X) for a batch of data.

    Args:
        None
    """
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()
        self.constant = torch.tensor(2 * torch.pi * torch.e)

    def forward(self, x, log_var, asr_loss):
        """
        Compute the conditional entropy H(Y|X).
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            log_var (torch.Tensor): Log variance for the stochastic mask of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Approximated conditional entropy for the batch.
        """
        # Compute variance
        var = torch.exp(log_var)

        # Compute conditional entropy
        conditional_entropy = torch.mean(0.5 * torch.sum(  # Sum over all dimensions
            torch.log(self.constant) + torch.log(var + 1e-8) + torch.log(x ** 2 + 1e-8)
        ,dim=-1))

        # Return negative since we aim to maximize the entropy
        return -1 * conditional_entropy
    
class AdditiveConditionalEntropyLoss(nn.Module):
    """
    Approximates the conditional entropy H(Y|X) for a batch of data.

    Args:
        None
    """
    def __init__(self, exponential_activation=True):
        super(AdditiveConditionalEntropyLoss, self).__init__()
        self.exponential_activation = exponential_activation

    def forward(self, log_var):
        """
        Compute the conditional entropy H(Y|X).
        
        Args:
            x (torch.Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            log_var (torch.Tensor): Log variance for the stochastic mask of shape (batch_size, sequence_length, input_dim).
        
        Returns:
            torch.Tensor: Approximated conditional entropy for the batch.
        """
        # Compute variance
        var = torch.exp(log_var)

        # Compute conditional entropy
        conditional_entropy = torch.mean(torch.sum(log_var ,dim=-1))

        # Return negative since we aim to maximize the entropy
        if self.exponential_activation:
            return torch.exp(-1*conditional_entropy)
        return -1 * conditional_entropy

class JSDivergence(nn.Module):

    def __init__(self):
        super(JSDivergence, self).__init__()
    
    def forward(self, p):

        dimension = p.shape[-1]
        return jensen_shannon_divergence(p) / dimension

def kl_divergence(p, q):
    """Computes the KL divergence D_KL[p || q] along the last dimension K."""
    return torch.mean(torch.sum(p * torch.log(p / q), dim=-1))  # Shape: [B, T]

def jensen_shannon_divergence(p):
    """
    Computes the Jensen-Shannon divergence between the given probability distribution p 
    and the uniform distribution for each sequence step.

    Args:
        p (torch.Tensor): Probability distribution of shape [B, T, K]

    Returns:
        torch.Tensor: JS divergence for each batch and time step (shape: [B, T])
    """
    print("P", p)
    B, T, K = p.shape  # Get dimensions
    u = torch.ones_like(p) / K
    m = 0.5 * (p + u)  # Mixture distribution
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(u, m)  # Shape: [B, T]


class InfoNCELoss(nn.Module):
    """
    Implements the InfoNCE loss for contrastive representation learning.
    
    Args:
        temperature (float): Temperature scaling parameter for similarity scores.
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, z, z_positive, z_negative):
        """
        Compute the InfoNCE loss.
        
        Args:
            z_positive (torch.Tensor): Positive sample of shape (batch_size, embedding_dim).
            z_negative (torch.Tensor): Negative samples of shape (batch_size, num_negatives, embedding_dim).
            z (torch.Tensor): Context vectors of shape (batch_size, embedding_dim).
        
        Returns:
            torch.Tensor: InfoNCE loss for the batch.
        """
        # Check for NaN values
        if torch.isnan(z).any():
            print("z is NAN")
        if torch.isnan(z_positive).any():
            print("z_positive is NAN")
        if torch.isnan(z_negative).any():
            print("z_negative is NAN")

        pos_sim = self.cos(z_positive, z)

        # Compute similarity for negative pairs
        neg_sim = self.cos(z_negative, z.unsqueeze(1).repeat(1, z_negative.shape[1], 1, 1))  # (batch_size, num_negatives)

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)

        # Apply temperature scaling
        logits /= self.temperature

        # Compute InfoNCE loss (log-softmax over all logits)
        loss = -F.log_softmax(logits, dim=1)[:, 0]  # Only keep the positive sample's loss

        # Check for NaN values in loss
        if torch.isnan(loss).any():
            print("loss is NAN")
        return loss.mean()
    
class InfoNCELossPooling(nn.Module):
    def __init__(self, temperature=0.2, num_negative_chunks=5):
        super(InfoNCELossPooling, self).__init__()
        self.temperature = temperature
        self.num_chunks = num_negative_chunks
        self.cos = nn.CosineSimilarity(dim=-1)
        self.pool = StatisticsPooling()

    def forward(self, z, z_positive, z_negative):
        """
        Args:
            z:          Tensor of shape (B, T, D)
            z_positive: Tensor of shape (B, T, D)
            z_negative: Tensor of shape (B, N, T, D)

        Returns:
            Scalar loss: averaged InfoNCE loss
        """
        B, N, T, D = z_negative.shape
        chunk_size = T // self.num_chunks
        assert chunk_size > 0, f"Sequence length {T} must be at least {self.num_chunks} for chunking."

        # Pool positive and anchor
        z = self.pool(z)  # (B, 2D)
        z_positive = self.pool(z_positive)  # (B, 2D)

        # Reshape and chunk z_negative → (B*N*num_chunks, chunk_size, D)
        z_negative = z_negative.view(B * N, T, D)
        z_negative_chunks = []

        for i in range(self.num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_chunks - 1 else T  # last chunk takes the remainder
            chunk = z_negative[:, start:end, :]  # (B*N, chunk_len, D)
            z_negative_chunks.append(self.pool(chunk))  # (B*N, 2D)

        # Stack pooled chunks → (B*N*num_chunks, 2D)
        z_negative_pooled = torch.stack(z_negative_chunks, dim=1)  # (B*N, num_chunks, 2D)
        z_negative_pooled = z_negative_pooled.view(B, N * self.num_chunks, -1)  # (B, N*num_chunks, 2D)

        # Compute similarities
        pos_sim = self.cos(z, z_positive)  # (B,)
        neg_sim = self.cos(z.unsqueeze(1).expand(-1, N * self.num_chunks, -1), z_negative_pooled)  # (B, N*num_chunks)

        # Combine and compute InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature  # (B, 1 + N*num_chunks)
        loss = -F.log_softmax(logits, dim=1)[:, 0]

        #if loss.mean().item() < 0.1:
        #    print("loss is too low")

        return loss.mean()
