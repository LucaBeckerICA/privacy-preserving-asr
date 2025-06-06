import torch
from peft import AdaLoraConfig, get_peft_model

def get_lora_trainable_modules(model):
    """
    Get the modules that are trainable for LoRA in the given model.

    Args:
        model (torch.nn.Module): The model to inspect.
    """
    trainable_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            trainable_modules.append(name)

    return trainable_modules

class AV2TextAdaLoRaTrainer(torch.nn.Module):
    def __init__(self, model, init_r, target_r, train_privacy_module_only=False):
        """
        Initialize the trainer with AdaLoRa configuration using the PEFT package.

        Args:
            model (AV2TextWrapper): The wrapped AV2Text model.
            init_r (int): Initial low-rank approximation rank.
            target_r (int): Target rank for adaptive LoRA.
            train_privacy_module_only (bool): Whether to train only the PrivacyModule.
        """
        super().__init__()
        self.model = model
        self.train_privacy_module_only = train_privacy_module_only

        # Get the modules that are trainable for LoRA (except processing_module)
        lora_trainable_modules = get_lora_trainable_modules(model)
        lora_trainable_modules = [name for name in lora_trainable_modules if "privacy_module" not in name]
        lora_trainable_modules = [name for name in lora_trainable_modules if "feature_extractor" not in name]

        # DEBUG
        lora_trainable_modules = [name for name in lora_trainable_modules if "layers.0" not in name]
        

        # Configure AdaLoRa using PEFT
        self.adalora_config = AdaLoraConfig(
            init_r=init_r,
            target_r=target_r,
            target_modules=lora_trainable_modules  # Apply AdaLoRa to these modules
        )

        # Apply AdaLoRa to the model using PEFT
        self.model = get_peft_model(self.model, self.adalora_config)
        self._freeze()

    def _freeze(self):
        """
        Freeze all layers except the PrivacyModule (and optinally Lora-Layers) to ensure it is trained from scratch.
        """
        if self.train_privacy_module_only:
            for name, param in self.model.named_parameters():
                if "privacy_module" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if "privacy_module" in name or "lora_" in name:# or "layers.0" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_attention_mask=None,
        input_ids=None
    ):
        """
        Perform a forward pass on the model.

        Args:
            batch (dict): A batch of input data containing:
                - input_ids (torch.Tensor): Input token IDs.
                - attention_mask (torch.Tensor): Attention mask for the input.
                - decoder_input_ids (torch.Tensor): Decoder input token IDs.
                - decoder_attention_mask (torch.Tensor): Attention mask for the decoder input.
                - labels (torch.Tensor): Ground-truth labels for the output.

        Returns:
            torch.Tensor: The loss computed by the model.
        """
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            input_ids=input_ids
        )
        return outputs
