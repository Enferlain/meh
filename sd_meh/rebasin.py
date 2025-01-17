# https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion
import logging
from collections import defaultdict
from random import shuffle
from typing import Dict, NamedTuple, Tuple
from tqdm import tqdm

import torch
from scipy.optimize import linear_sum_assignment

#.getLogger("sd_meh").addHandler(logging.NullHandler())
logger = logging.getLogger("merge_models")
logging.basicConfig(level=logging.INFO)

SPECIAL_KEYS = [
    "first_stage_model.decoder.norm_out.weight",
    "first_stage_model.decoder.norm_out.bias",
    "first_stage_model.encoder.norm_out.weight",
    "first_stage_model.encoder.norm_out.bias",
#    "model.diffusion_model.out.0.weight",
#    "model.diffusion_model.out.0.bias",
]


def step_weights_and_bases(
    weights: Dict, bases: Dict, it: int = 0, iterations: int = 1
) -> Tuple[Dict, Dict]:
    new_weights = {
        gl: [
            1 - (1 - (1 + it) * v / iterations) / (1 - it * v / iterations)
            if it > 0
            else v / iterations
            for v in w
        ]
        for gl, w in weights.items()
    }

    new_bases = {
        k: 1 - (1 - (1 + it) * v / iterations) / (1 - it * v / iterations)
        if it > 0
        else v / iterations
        for k, v in bases.items()
    }

    return new_weights, new_bases


def flatten_params(model):
    return model["state_dict"]


rngmix = lambda rng, x: random.fold_in(rng, hash(x))


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p:
            w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def update_model_a(ps: PermutationSpec, perm, model_a, new_alpha):
    for k in model_a:
        try:
            perm_params = get_permuted_param(
                ps, perm, k, model_a
            )
            model_a[k] = model_a[k] * (1 - new_alpha) + new_alpha * perm_params
        except RuntimeError:  # dealing with pix2pix and inpainting models
            continue
    return model_a


def inner_matching(
    n,
    ps,
    p,
    params_a,
    params_b,
    usefp16,
    progress,
    number,
    linear_sum,
    perm,
    device,
    sdxl,
):
    """
    Performs inner weight matching for a specific layer.

    Args:
        n: Size of the weight tensors.
        ps: PermutationSpec object.
        p: Layer key.
        params_a: Dictionary containing weights from model A.
        params_b: Dictionary containing weights from model B.
        usefp16: Boolean flag indicating if using fp16 data type.
        progress: Boolean flag indicating progress made (optional).
        number: Number of successful improvement steps (optional).
        linear_sum: Cumulative improvement score (optional).
        perm: Current permutation dictionary.
        device: Device to use for computations (e.g., "cpu" or "cuda").
        sdxl: Boolean flag for enabling detailed logging (optional).

    Returns:
        linear_sum: Updated cumulative improvement score.
        number: Updated number of successful improvement steps.
        perm: Updated permutation dictionary with improved alignment for layer p.
        progress: Boolean flag indicating if improvement was made.
    """

    A = torch.zeros((n, n), dtype=torch.float16 if usefp16 else torch.zeros((n, n)))
    A = A.to(device)

    logging.debug(f"Processing weight key for layer {p} within inner_matching")

    # Check if ps.perm_to_axes[p] is a dictionary
    if isinstance(ps.perm_to_axes[p], dict):
        for wk, axis in ps.perm_to_axes[p].items():
            if wk.startswith("first_stage_model"):
                continue  # Skip weight key if it starts with "first_stage_model"
            w_a = params_a[wk]
            w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
            w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
            w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)

            if usefp16:
                w_a = w_a.half().to(device)
                w_b = w_b.half().to(device)

            try:
                A += torch.matmul(w_a, w_b)
            except RuntimeError:
                A += torch.matmul(torch.dequantize(w_a), torch.dequantize(w_b))

    A = A.cpu()
    ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
    A = A.to(device)

    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    eye_tensor = torch.eye(n).to(device)

    oldL = torch.vdot(
        torch.flatten(A).float(), torch.flatten(eye_tensor[perm[p].long()])
    )
    newL = torch.vdot(torch.flatten(A).float(), torch.flatten(eye_tensor[ci, :]))

    if usefp16:
        oldL = oldL.half()
        newL = newL.half()

    if newL - oldL != 0:
        linear_sum += abs((newL - oldL).item())
        number += 1
        logging.debug(f"Merge Rebasin permutation: {p}={newL-oldL}")

    progress = progress or newL > oldL + 1e-12

    perm[p] = torch.Tensor(ci).to(device)

    return linear_sum, number, perm, progress

def weight_matching(
    ps: PermutationSpec,
    params_a,
    params_b,
    max_iter=1,
    init_perm=None,
    usefp16=False,
    device="cpu",
    sdxl: bool = False,
):
    logging.debug(f"Starting weight matching: {params_a.keys()} vs {params_b.keys()}")
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perm_to_axes.items()
        if axes[0][0] in params_a.keys()
    }
    perm = {}
    perm = (
        {p: torch.arange(n).to(device) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )

    linear_sum = 0
    number = 0

    # Remove special_layers
    
    for _i in range(max_iter):
        progress = False
        # Loop through all layers in perm_sizes
        for p, n in perm_sizes.items():
          linear_sum, number, perm, progress = inner_matching(
              n,
              ps,
              p,
              params_a,
              params_b,
              usefp16,
              progress,
              number,
              linear_sum,
              perm,
              device,
              sdxl,
          )
        if not progress:
          break

    average = linear_sum / number if number > 0 else 0
    return (perm, average)
