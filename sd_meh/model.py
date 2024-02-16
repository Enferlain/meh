import logging
import os
from dataclasses import dataclass
from typing import Dict

import safetensors
import torch
from tensordict import TensorDict

# Configure logging for SDModel
logger = logging.getLogger("SDModel")
logger.setLevel(logging.DEBUG)  # Adjust the log level as needed for your use case

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class SDModel:
    model_path: os.PathLike
    device: str

    def load_model(self):
        model_type = "SafeTensors" if self.model_path.suffix == ".safetensors" else "standard"
        logger.info(f"Loading {model_type} model from: {self.model_path} on device: {self.device}")

        try:
            if model_type == "SafeTensors":
                ckpt = safetensors.torch.load_file(self.model_path, device=self.device)
            else:
                ckpt = torch.load(self.model_path, map_location=self.device)

            # Process the state dictionary without explicitly passing the sdxl flag
            state_dict = get_state_dict_from_checkpoint(ckpt)
            
            logger.info("Model loaded and processed successfully.")
            return TensorDict.from_dict(state_dict)

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)  # Log with exception information
            raise


# TODO: tidy up
# from: stable-diffusion-webui/modules/sd_models.py
def get_state_dict_from_checkpoint(pl_sd, sdxl: bool = False):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    # Check for the presence of the specific key and set sdxl to True if found
    specific_key = "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.weight"
    if specific_key in pl_sd:
        sdxl = True
        logger.info(f"Specific key '{specific_key}' found. Setting SDXL to True.")

    sd = {}
    for k, v in pl_sd.items():
        # Apply key transformations only if sdxl is False
        new_key = transform_checkpoint_dict_key(k) if not sdxl else k
        sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


checkpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k