from train_avhubert import load_main_model
import torch
import yaml
from av_hubert_s2s.model.avhubert2text import AV2TextForConditionalGeneration

def count_parameters(model, keyword="gamma_audio"):
    total_params = 0
    detailed_params = {}

    for name, param in model.named_parameters():
        if keyword in name:
            param_count = param.numel()
            detailed_params[name] = param_count
            total_params += param_count

    return total_params, detailed_params

config_path = "config/ba03bb03.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, adaptor_scheduler = load_main_model(device, config['av_hubert'])

path = "nguyenvulebinh/AV-HuBERT"
orig_model = AV2TextForConditionalGeneration.from_pretrained(path)
print("FIN")