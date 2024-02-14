import inspect
import logging

from sd_meh import merge_methods
from sd_meh.merge import NUM_TOTAL_BLOCKS, NUM_TOTAL_BLOCKS_XL
from sd_meh.presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS

MERGE_METHODS = dict(inspect.getmembers(merge_methods, inspect.isfunction))
BETA_METHODS = [
    name
    for name, fn in MERGE_METHODS.items()
    if "beta" in inspect.getfullargspec(fn)[0]
]


def compute_weights(weights, base, sdxl: bool = False):
    # Define the number of total blocks based on the sdxl flag
    num_blocks = NUM_TOTAL_BLOCKS_XL if sdxl else NUM_TOTAL_BLOCKS

    logging.info(f"compute_weights called with weights: {weights}, sdxl: {sdxl}")

    # If weights is a comma-separated string, convert to a list of floats
    if isinstance(weights, str) and "," in weights:
        return list(map(float, weights.split(",")))

    # If weights is not provided, use the base value for all blocks
    if not weights:
        return [base] * num_blocks

    # If weights is already a list, return as is (assuming it's a list of floats)
    if isinstance(weights, list):
        return weights

    # If weights is a single number (not a list or string), create a list of that value for all blocks
    return [float(weights)] * num_blocks




def assemble_weights_and_bases(preset, weights, base, greek_letter, sdxl: bool = False):
    logging.info(f"Assembling {greek_letter} w&b")
    if preset:
        logging.info(f"Using {preset} preset")
        if sdxl:
            base, *weights = SDXL_BLOCK_WEIGHTS_PRESETS[preset]
        else:
            base, *weights = BLOCK_WEIGHTS_PRESETS[preset]
    bases = {greek_letter: base}
    weights = {greek_letter: compute_weights(weights, base, sdxl)}

    logging.info(f"base_{greek_letter}: {bases[greek_letter]}")
    logging.info(f"{greek_letter} weights: {weights[greek_letter]}")

    logging.info(f"Final weights in assemble_weights_and_bases: {weights}")
    logging.info(f"Final bases in assemble_weights_and_bases: {bases}")
    return weights, bases



def interpolate_presets(
    weights, bases, weights_b, bases_b, greek_letter, presets_lambda
):
    logging.info(f"Interpolating {greek_letter} w&b")
    for i, e in enumerate(weights[greek_letter]):
        weights[greek_letter][i] = (
            1 - presets_lambda
        ) * e + presets_lambda * weights_b[greek_letter][i]

    bases[greek_letter] = (1 - presets_lambda) * bases[
        greek_letter
    ] + presets_lambda * bases_b[greek_letter]

    logging.info(f"Interpolated base_{greek_letter}: {bases[greek_letter]}")
    logging.info(f"Interpolated {greek_letter} weights: {weights[greek_letter]}")

    return weights, bases


def weights_and_bases(
    merge_mode,
    weights_alpha,
    base_alpha,
    block_weights_preset_alpha,
    weights_beta,
    base_beta,
    block_weights_preset_beta,
    block_weights_preset_alpha_b,
    block_weights_preset_beta_b,
    sdxl_block_weights_preset_alpha,
    sdxl_block_weights_preset_beta,
    sdxl_block_weights_preset_alpha_b,
    sdxl_block_weights_preset_beta_b,
    presets_alpha_lambda,
    presets_beta_lambda,
    sdxl: bool = False,
):
    # Use sdxl presets if sdxl flag is True, else use standard presets
    alpha_preset = sdxl_block_weights_preset_alpha if sdxl else block_weights_preset_alpha
    beta_preset = sdxl_block_weights_preset_beta if sdxl else block_weights_preset_beta
    alpha_b_preset = sdxl_block_weights_preset_alpha_b if sdxl else block_weights_preset_alpha_b
    beta_b_preset = sdxl_block_weights_preset_beta_b if sdxl else block_weights_preset_beta_b

    weights, bases = assemble_weights_and_bases(
        block_weights_preset_alpha,
        weights_alpha,
        base_alpha,
        "alpha",
        sdxl,
    )

    if block_weights_preset_alpha_b:
        logging.info("Computing w&b for alpha interpolation")
        weights_b, bases_b = assemble_weights_and_bases(
            block_weights_preset_alpha_b,
            None,
            None,
            "alpha",
            sdxl,
        )
        weights, bases = interpolate_presets(
            weights,
            bases,
            weights_b,
            bases_b,
            "alpha",
            presets_alpha_lambda,
        )

    if merge_mode in BETA_METHODS:
        weights_beta, bases_beta = assemble_weights_and_bases(
            block_weights_preset_beta,
            weights_beta,
            base_beta,
            "beta",
            sdxl,
        )

        if block_weights_preset_beta_b:
            logging.info("Computing w&b for beta interpolation")
            weights_b, bases_b = assemble_weights_and_bases(
                block_weights_preset_beta_b,
                None,
                None,
                "beta",
                sdxl,
            )
            weights, bases = interpolate_presets(
                weights,
                bases,
                weights_b,
                bases_b,
                "beta",
                presets_beta_lambda,
            )

        weights |= weights_beta
        bases |= bases_beta

    print(f"Weights after assembly: {weights}")
    print(f"Bases after assembly: {bases}")
    return weights, bases
