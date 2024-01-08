import logging
import os
from dataclasses import dataclass
from typing import Dict

import safetensors
import torch
from tensordict import TensorDict

logger = logging.getLogger("SDModel")
logger.setLevel(logging.DEBUG)  # Adjust level as needed
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
            ckpt = safetensors.torch.load_file(self.model_path, device=self.device) if model_type == "SafeTensors" \
                   else torch.load(self.model_path, map_location=self.device)
            
            logger.info("Model loaded successfully.")
            return TensorDict.from_dict(get_state_dict_from_checkpoint(ckpt))

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def sdxl_model(model: Dict) -> bool:
    # Example key unique to SDXL models
    sdxl_unique_key = "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.weight"
    return sdxl_unique_key in model.keys()

# TODO: tidy up
# from: stable-diffusion-webui/modules/sd_models.py
def get_state_dict_from_checkpoint(pl_sd):
    sdxl = sdxl_model(pl_sd)
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k, sdxl)
        sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


checkpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k, sdxl: bool):
    if sdxl:
        # If there's no specific renaming needed for SDXL models, just return the key as it is.
        return k
    else:
        for text, replacement in checkpoint_dict_replacements.items():
            if k.startswith(text):
                return replacement + k[len(text):]
    return k  # Return the original key if no transformation is needed