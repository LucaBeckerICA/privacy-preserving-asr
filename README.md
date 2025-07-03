# Privacy-Preserving Fine-Tuning of Audio-Visual Speech Recognition

## About

This project explores a novel approach to fine-tuning audio-visual speech recognition (AV-ASR) models under strong privacy constraints. The core idea is to transform intermediate representations in a way that preserves recognition performance while preventing information leakage that could be exploited by adversaries.

We introduce feature-wise additive and multiplicative masks trained with a contrastive learning objective. These masks aim to:

- Retain task-relevant information for speech recognition - especially linguistic content,
- Obscure speaker identity.

The system builds on top of a pre-trained [AV-HuBERT](https://github.com/facebookresearch/AV-HuBERT) model, with frozen feature extractors and lightweight fine-tuning via [AdaLoRA](https://github.com/QingruZhang/AdaLoRA). A novel contrastive loss ensures that noisy variants of input sequences remain close to their clean counterparts while becoming less similar to different utterances from the same speaker.
Additional repositories used are [RJCA for Speaker Verification](https://github.com/praveena2j/RJCAforSpeakerVerification) and this pre-trained [AV-HuBERT model](https://huggingface.co/nguyenvulebinh/AV-HuBERT).
This repository contains training and evaluation code, data preprocessing scripts, and example configurations for replicating our results.

## Installation

To set up the environment and run the project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/privacy-av-asr.git
   cd privacy-av-asr
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare datasets**
   Follow the instructions in [`data/README.md`](data/README.md) to download and preprocess the audio-visual datasets (e.g., VoxCeleb2, LRS3).

5. **Run the AV-HuBERT training script**
   ```bash
   python train_avhubert.py --config config/config.yaml
   ```
6. **Run the speaker verification script**
   ```bash
   python train_speaker_verification.py --config config/config.yaml
   ```
   make sure the ```config.yaml```is equal for both training.

> 💡 For GPU support, make sure your system has CUDA-compatible hardware and the appropriate PyTorch build installed.

## License

This project is licensed under the MIT license. 
If you use this work in your research, please cite or reference the following:
```bash
@article{privacy_avsr2025,
  title     = {Contrastive Representation Learning for Privacy-Preserving Fine-Tuning of Audio-Visual Speech Recognition},
  author    = {Becker, Luca and Martin, Rainer},
  journal   = {Conference Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  note      = {Accepted at the Conference Workshop on Applications of Signal Processing to Audio and Acoustics, Tahoe City, USA, October 2025},
  year      = {2025},
  month     = {October},
  status    = {accepted}
}
```
