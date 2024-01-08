import gc
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from tqdm import tqdm

from sd_meh import merge_methods
from sd_meh.model import SDModel
from sd_meh.rebasin import (
    apply_permutation,
    step_weights_and_bases,
    update_model_a,
    weight_matching,
)
from sd_meh.merge_PermSpec import sdunet_permutation_spec
from sd_meh.merge_PermSpec_SDXL import sdxl_permutation_spec

logger = logging.getLogger("merge_models")
logger.setLevel(logging.INFO)  # Adjust level as needed
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_TOKENS = 77
NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

NUM_INPUT_BLOCKS_XL = 9
NUM_OUTPUT_BLOCKS_XL = 9
NUM_TOTAL_BLOCKS_XL = NUM_INPUT_BLOCKS_XL + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS_XL



KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


NAI_KEYS = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def fix_clip(model: Dict) -> Dict:
    if KEY_POSITION_IDS in model.keys():
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))],
            dtype=torch.int64,
            device=model[KEY_POSITION_IDS].device,
        )

    return model


def fix_key(model: Dict, key: str, sdxl: bool):
    if sdxl:
        # Add specific conditions for SDXL model keys if necessary
        if key.startswith("conditioner.embedders."):
            pass  # Currently, do nothing for SDXL model keys
    else:
        for nk in NAI_KEYS:
            if key.startswith(nk):
                model[key.replace(nk, NAI_KEYS[nk])] = model[key]
                del model[key]
    return model



# https://github.com/j4ded/sdweb-merge-block-weighted-gui/blob/master/scripts/mbw/merge_block_weighted.py#L115
def fix_model(model: Dict, sdxl: bool):
    for k in list(model.keys()):
        model = fix_key(model, k, sdxl)
    return fix_clip(model)


def load_sd_model(model: os.PathLike | str, device: str = "cpu") -> Dict:
    if isinstance(model, str):
        model = Path(model)

    return SDModel(model, device).load_model()


def prune_sd_model(model: Dict, sdxl: bool) -> Dict:
    keys = list(model.keys())
    if sdxl:
        for k in keys:
            if (
                not k.startswith("model.diffusion_model.")
                #vae?
                #and not k.startswith("conditioner.embedders.")
            ):
                del model[k]
    else:
        for k in keys:
            if (
                not k.startswith("model.diffusion_model.")
                #and not k.startswith("first_stage_model.")
                #and not k.startswith("cond_stage_model.")
            ):
                del model[k]
    return model


def restore_sd_model(original_model: Dict, merged_model: Dict) -> Dict:
    for k in original_model:
        if k not in merged_model:
            merged_model[k] = original_model[k]
    return merged_model


def log_vram(txt=""):
    alloc = torch.cuda.memory_allocated(0)
    logger.debug(f"{txt} VRAM: {alloc*1e-9:5.3f}GB")


def load_thetas(
    models: Dict[str, os.PathLike | str],
    prune: bool,
    device: str,
    precision: int,
    sdxl: bool,
) -> Dict:
    log_vram("before loading models")
    logger.info("Loading models with the following parameters:")
    logger.info(f"Prune: {prune}, Device: {device}, Precision: {precision}, SDXL: {sdxl}")
    
    # Check if device is valid
    valid_devices = ["cpu", "cuda"]
    if device not in valid_devices:
        raise ValueError(f"Invalid device setting: '{device}'. Valid options are {valid_devices}.")
    
    if prune:
        logger.info("Pruning enabled. Loading and pruning models.")
        thetas = {k: prune_sd_model(load_sd_model(m, "cpu"), sdxl) for k, m in models.items()}
    else:
        logger.info("Pruning disabled. Loading models without pruning.")
        thetas = {k: load_sd_model(m, device) for k, m in models.items()}

    if device.startswith("cuda"):
        logger.info(f"Transferring models to CUDA device: {device}")
        for model_key, model in thetas.items():
            logger.debug(f"Processing model: {model_key}")
            for key, block in model.items():
                try: 
                    if precision == 16:
                        logger.debug(f"Transferring block '{key}' to device '{device}' with precision 16.")
                        thetas[model_key].update({key: block.to(device).half()})
                    else:
                        logger.debug(f"Transferring block '{key}' to device '{device}' with full precision.")
                        thetas[model_key].update({key: block.to(device)})
                except Exception as e:
                    logger.error(f"Error transferring block '{key}' to device '{device}': {e}")
                    raise
    else:
        logger.info(f"No CUDA transfer needed, using device: {device}")

    log_vram("models loaded")
    return thetas


def merge_models(
    models: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    re_basin: bool = False,
    iterations: int = 1,
    device: str = "cpu",
    work_device: Optional[str] = None,
    prune: bool = False,
    threads: int = 1,
    sdxl: bool = False,
) -> Dict:
    logger.info(f"Initializing merge process. Mode: {merge_mode}, SDXL: {sdxl}, Device: {device}")
    thetas = load_thetas(models, prune, device, precision, sdxl)
    logger.debug(f"Thetas loaded. Keys: {list(thetas['model_a'].keys())}")

    sdxl = (
    "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.weight"
    in thetas["model_a"].keys()
    )
    logger.info(f"sdxl: {sdxl}")

    logger.info(f"start merging with {merge_mode} method")
    if re_basin:
        logger.info("Rebasin merge initiated.")
        merged = rebasin_merge(
            thetas,
            weights,
            bases,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
            iterations=iterations,
            device=device,
            work_device=work_device,
            threads=threads,
            sdxl=sdxl,
        )
    else:
        logger.info("Simple merge initiated.")
        merged = simple_merge(
            thetas,
            weights,
            bases,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
            device=device,
            work_device=work_device,
            threads=threads,
            sdxl=sdxl,
        )
    logger.info("Unpruning merged model.")
    unpruned_model = un_prune_model(merged, thetas, models, device, prune, precision, sdxl)
    logger.info("Merge process completed.")
    return unpruned_model


def un_prune_model(
    merged: Dict,
    thetas: Dict,
    models: Dict,
    device: str,
    prune: bool,
    precision: int,
    sdxl: bool = False,
) -> Dict:
    if prune:
        logger.info("Un-pruning merged model")
        del thetas
        gc.collect()
        log_vram("remove thetas")

        # Un-prune model A
        original_a = load_sd_model(models["model_a"], device)
        for key in tqdm(original_a.keys(), desc="un-prune model a"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_a[key]})
                if precision == 16:
                    merged.update({key: merged[key].half()})
        del original_a
        gc.collect()
        log_vram("remove original_a")

        # Un-prune model B
        original_b = load_sd_model(models["model_b"], device)
        for key in tqdm(original_b.keys(), desc="un-prune model b"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_b[key]})
                if precision == 16:
                    merged.update({key: merged[key].half()})
        
        # Delete original_b after the loop is completed
        del original_b
        gc.collect()
        log_vram("remove original_b")
        logger.debug("Un-pruning for model B completed.")

    return fix_model(merged, sdxl)



def simple_merge(
    thetas: Dict[str, Dict],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    device: str = "cpu",
    work_device: Optional[str] = None,
    threads: int = 1,
    sdxl: bool = False,
) -> Dict:
    futures = []
    with tqdm(thetas["model_a"].keys(), desc="stage 1") as progress:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for key in thetas["model_a"].keys():
                future = executor.submit(
                    simple_merge_key,
                    progress,
                    key,
                    thetas,
                    weights,
                    bases,
                    merge_mode,
                    precision,
                    weights_clip,
                    device,
                    work_device,
                    sdxl,
                )
                futures.append(future)

        for res in futures:
            res.result()

    log_vram("after stage 1")

    for key in tqdm(thetas["model_b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["model_a"].keys():
            thetas["model_a"].update({key: thetas["model_b"][key]})
            if precision == 16:
                thetas["model_a"].update({key: thetas["model_a"][key].half()})

    log_vram("after stage 2")

    return fix_model(thetas["model_a"], sdxl)


def rebasin_merge(
    thetas: Dict[str, os.PathLike | str],
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    iterations: int = 1,
    device="cpu",
    work_device=None,
    threads: int = 1,
    sdxl: bool = False,
):
    logger.info("Initiating rebasin merging process.")
    logger.debug(f"Rebasin parameters: SDXL={sdxl}, precision={precision}, device={device}")

    sdxl = "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.weight" in thetas["model_a"].keys()
    perm_spec = sdxl_permutation_spec() if sdxl else sdunet_permutation_spec()
    logger.debug(f"Permutation specification selected: {'SDXL' if sdxl else 'SDUnet'}")

    model_a = thetas["model_a"].clone()
    logger.info(f"Starting rebasin iterations: total iterations={iterations}")

    for it in range(iterations):
        logger.info(f"Processing rebasin iteration: {it+1}/{iterations}")
        log_vram(f"Before iteration {it+1}")
        # Updating weights and bases
        new_weights, new_bases = step_weights_and_bases(
            weights,
            bases,
            it,
            iterations,
        )
        log_vram("After updating weights and bases")

        # normal block merge we already know and love
        thetas["model_a"] = simple_merge(
            thetas,
            new_weights,
            new_bases,
            merge_mode,
            precision,
            False,
            device,
            work_device,
            threads,
            sdxl,
        )
        log_vram("Block merge completed")

        # Weight matching and permutation application
        perm_1, y = weight_matching(
            perm_spec,
            model_a,
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
            sdxl=sdxl
        )
        thetas["model_a"] = apply_permutation(perm_spec, perm_1, thetas["model_a"])
        log_vram("First weight matching and permutation applied")

        perm_2, z = weight_matching(
            perm_spec,
            thetas["model_b"],
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
            sdxl=sdxl
        )
        new_alpha = torch.nn.functional.normalize(
            torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0
        ).tolist()[0]
        thetas["model_a"] = update_model_a(
            perm_spec, perm_2, thetas["model_a"], new_alpha
        )
        log_vram(f"Second weight matching and model update completed for iteration {it+1}")

    if weights_clip:
        clip_thetas = thetas.copy()
        clip_thetas["model_a"] = model_a
        thetas["model_a"] = clip_weights(thetas, thetas["model_a"])
        logger.debug("Weights clipping applied to the merged model.")
        
    logger.info("Rebasin merging process completed successfully.")
    return thetas["model_a"]

def simple_merge_key(progress, key, thetas, *args, **kwargs):
    with merge_key_context(key, thetas, *args, **kwargs) as result:
        if result is not None:
            thetas["model_a"].update({key: result.detach().clone()})

        progress.update()


def merge_key(
    key: str,
    thetas: Dict,
    weights: Dict,
    bases: Dict,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    device: str = "cpu",
    work_device: Optional[str] = None,
    sdxl: bool = False,
) -> Optional[Tuple[str, Dict]]:
    logger.debug(f"Starting merge_key function for key: {key}")
    if work_device is None:
        work_device = device

    if KEY_POSITION_IDS in key:
        return

    for theta in thetas.values():
        if key not in theta.keys():
            return
            return thetas["model_a"][key]

    if "model" in key:
        current_bases = bases

        if "model.diffusion_model." in key:
            weight_index = -1

            re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # 12, 9 for sdxl
            re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # 1
            re_out = re.compile(r"\.output_blocks\.(\d+)\.")  # 12, 9 for sdxl

            if "time_embed" in key and sdxl:
                # Position of time_embed in SDXL models
                weight_index = NUM_TOTAL_BLOCKS_XL - 1
            elif "label_emb" in key and sdxl:    
                # Position of label_emb layers in SDXL models
                # Set to index just before middle block
                weight_index = NUM_INPUT_BLOCKS_XL + 1  # Adjust based on actual model structure
            elif ".out." in key:
                weight_index = (
                    NUM_TOTAL_BLOCKS_XL - 1 if sdxl else NUM_TOTAL_BLOCKS - 1
                )  # after output blocks
            elif m := re_inp.search(key):
                weight_index = int(m.groups()[0])
            elif re_mid.search(key):
                weight_index = NUM_INPUT_BLOCKS_XL if sdxl else NUM_INPUT_BLOCKS
            elif m := re_out.search(key):
                weight_index = (
                    (NUM_INPUT_BLOCKS_XL if sdxl else NUM_INPUT_BLOCKS)
                    + NUM_MID_BLOCK
                    + int(m.groups()[0])
                )

            if weight_index >= (NUM_TOTAL_BLOCKS_XL if sdxl else NUM_TOTAL_BLOCKS):
                raise ValueError(f"illegal block index {weight_index} for key {key}")
                
            logging.debug(f"key: {key}, weight_index: {weight_index}, sdxl: {sdxl}")    

            if weight_index >= 0:
                current_bases = {k: w[weight_index] for k, w in weights.items()}

        try:
            merge_method = getattr(merge_methods, merge_mode)
        except AttributeError as e:
            raise ValueError(f"{merge_mode} not implemented, aborting merge!") from e

        merge_args = get_merge_method_args(current_bases, thetas, key, work_device)

        # dealing wiht pix2pix and inpainting models
        if (a_size := merge_args["a"].size()) != (b_size := merge_args["b"].size()):
            logger.debug(f"Size mismatch in merge_args: a_size={a_size}, b_size={b_size}")
            if a_size[1] > b_size[1]:
                merged_key = merge_args["a"]
            else:
                merged_key = merge_args["b"]
        else:
            merged_key = merge_method(**merge_args).to(device)

        if weights_clip:
            merged_key = clip_weights_key(thetas, merged_key, key)

        if precision == 16:
            merged_key = merged_key.half()

        return merged_key


def clip_weights(thetas, merged):
    for k in thetas["model_a"].keys():
        if k in thetas["model_b"].keys():
            merged.update({k: clip_weights_key(thetas, merged[k], k)})
    return merged


def clip_weights_key(thetas, merged_weights, key):
    t0 = thetas["model_a"][key]
    t1 = thetas["model_b"][key]
    maximums = torch.maximum(t0, t1)
    minimums = torch.minimum(t0, t1)
    return torch.minimum(torch.maximum(merged_weights, minimums), maximums)


@contextmanager
def merge_key_context(*args, **kwargs):
    result = merge_key(*args, **kwargs)
    try:
        yield result
    finally:
        if result is not None:
            del result


def get_merge_method_args(
    current_bases: Dict,
    thetas: Dict,
    key: str,
    work_device: str,
) -> Dict:
    merge_method_args = {
        "a": thetas["model_a"][key].to(work_device),
        "b": thetas["model_b"][key].to(work_device),
        **current_bases,
    }

    if "model_c" in thetas:
        merge_method_args["c"] = thetas["model_c"][key].to(work_device)

    return merge_method_args


def save_model(model, output_file, file_format) -> None:
    logger.info(f"Saving model to '{output_file}' in format: {file_format}")
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model if type(model) == dict else model.to_dict(),
            f"{output_file}.safetensors",
            metadata={"format": "pt"},
        )
        logger.debug("Model saved in SafeTensors format.")
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
        logger.debug("Model saved in .ckpt format.")