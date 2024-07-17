import functools
import math
import operator
import textwrap
import torch
import time
import numpy as np
import cupy as cp
import ot
import os
import gc
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch import Tensor
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
from came_pytorch import CAME
from pytorch_optimizer import REXScheduler
#from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "euclidean_add_difference",
    "multiply_difference",
    "top_k_tensor_sum",
    "similarity_add_difference",
    "distribution_crossover",
    "ties_add_difference",
    "rotate",
    "train_difference",
    "add_perpendicular",
    "vector_rejection",
    "slerp",
    "geometric sum",
    "add_cosine_a",
    "add_cosine_b",
    "neuron_train_difference",
    "neuron_train_difference_jaccard",
    "neuron_add_difference_similarity",
    "neuron_train_difference_euclidean",
    "neuron_train_difference_cosine",
    "neuron_train_difference_mahalanobis",
    "neuron_train_difference_correlation",
    "neuron_train_difference_proj",
    "crossover",
    "anchored_guided_alignment",
    "literal_train_difference",
    "merge_with_transformers"
]


EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero


def weighted_sum(
    a: Tensor, b: Tensor, alpha: float, **kwargs
) -> Tensor:
    return (1 - alpha) * a + alpha * b


def weighted_subtraction(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON

    return (a - alpha * beta * b) / (1 - alpha * beta)


def tensor_sum(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    if a.shape == ():
        if alpha > 0.5:
            return b
        return a

    start_i, end_i, region_is_inverted = ratio_to_region(alpha, beta, a.size(0))
    if region_is_inverted:
        b[start_i:end_i] = a[start_i:end_i]
        return b
    else:
        a[start_i:end_i] = b[start_i:end_i]
        return a


def add_cosine_a(
    a: Tensor, b: Tensor, *, alpha: float, **kwargs
) -> Tensor:
    a_norm = torch.nn.functional.normalize(a, dim=0)
    b_norm = torch.nn.functional.normalize(b, dim=0)
    similarity = torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0)
    return add_cosine_generic(a, b, alpha, similarity)


def add_cosine_generic(
    a: Tensor, b: Tensor, alpha: float, similarity: Tensor
) -> Tensor:
    k = 1 - torch.clamp(similarity - alpha, 0, 1)
    return weighted_sum(a, b, alpha=k)


def add_cosine_b(
    a: Tensor, b: Tensor, *, alpha: float, **kwargs
) -> Tensor:
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    dot_product = torch.sum(a * b)
    magnitude_similarity = dot_product / (torch.norm(a) * torch.norm(b))
    combined_similarity = (similarity + magnitude_similarity) / 2.0
    return add_cosine_generic(a, b, alpha, combined_similarity)


def geometric_sum(
    a: Tensor, b: Tensor, *, alpha: float, **kwargs
) -> Tensor:
    a = torch.complex(a, torch.zeros_like(a))
    b = torch.complex(b, torch.zeros_like(b))
    res = a ** (1 - alpha) * b ** alpha
    return res.real


def perpendicular_component(
    a: Tensor, b: Tensor, **kwargs
) -> Tensor:
    norm_a = torch.linalg.norm(a)
    res = b - a * (a / norm_a * (b / norm_a)).sum()
    if res.isnan().any():
        return torch.zeros_like(a)
    return res

def add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    return a + alpha * (b - c)


def sum_twice(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c


def triple_sum(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    return (1 - alpha - beta) * a + alpha * b + beta * c


def euclidean_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    a_diff = a.float() - c.float()
    b_diff = b.float() - c.float()
    a_diff = torch.nan_to_num(a_diff / torch.linalg.norm(a_diff))
    b_diff = torch.nan_to_num(b_diff / torch.linalg.norm(b_diff))

    distance = (1 - alpha) * a_diff**2 + alpha * b_diff**2
    distance = torch.sqrt(distance)
    sum_diff = weighted_sum(a.float(), b.float(), alpha) - c.float()
    distance = torch.copysign(distance, sum_diff)

    target_norm = torch.linalg.norm(sum_diff)
    return c + distance / torch.linalg.norm(distance) * target_norm


def multiply_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    diff_a = torch.pow(torch.abs(a.float() - c), (1 - alpha))
    diff_b = torch.pow(torch.abs(b.float() - c), alpha)
    difference = torch.copysign(diff_a * diff_b, weighted_sum(a, b, beta) - c)
    return c + difference.to(c.dtype)


def top_k_tensor_sum(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    a_flat = torch.flatten(a)
    a_dist = torch.msort(a_flat)
    b_indices = torch.argsort(torch.flatten(b), stable=True)
    redist_indices = torch.argsort(b_indices)

    start_i, end_i, region_is_inverted = ratio_to_region(alpha, beta, torch.numel(a))
    start_top_k = kth_abs_value(a_dist, start_i)
    end_top_k = kth_abs_value(a_dist, end_i)

    indices_mask = (start_top_k <= torch.abs(a_dist)) & (torch.abs(a_dist) <= end_top_k)
    if region_is_inverted:
        indices_mask = ~indices_mask
    indices_mask = torch.gather(indices_mask.float(), 0, redist_indices)

    a_redist = torch.gather(a_dist, 0, redist_indices)
    a_redist = (1 - indices_mask) * a_flat + indices_mask * a_redist
    return a_redist.reshape_as(a)


def kth_abs_value(a: Tensor, k: int) -> Tensor:
    if k <= 0:
        return torch.tensor(-1, device=a.device)
    else:
        return torch.kthvalue(torch.abs(a.float()), k)[0]


def ratio_to_region(alpha: float, beta: float, n: int) -> Tuple[int, int, bool]:
    if alpha < 0:
        beta += alpha
        alpha = -alpha
    alpha = min(alpha, 1)

    if beta < 0:
        beta = 1 + beta - int(beta)
    beta = math.fmod(beta, 1.0)

    if alpha + beta <= 1:
        inverted = False
        start = beta * n
        end = (alpha + beta) * n
    else:
        inverted = True
        start = (alpha + beta - 1) * n
        end = beta * n

    return round(start), round(end), inverted


def similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity * beta, nan=beta)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return (1 - similarity) * ab_diff + similarity * ab_sum


def distribution_crossover(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
):
    if a.shape == ():
        return alpha * a + (1 - alpha) * b

    c_indices = torch.argsort(torch.flatten(c))
    a_dist = torch.gather(torch.flatten(a), 0, c_indices)
    b_dist = torch.gather(torch.flatten(b), 0, c_indices)

    a_dft = torch.fft.rfft(a_dist.float())
    b_dft = torch.fft.rfft(b_dist.float())

    dft_filter = torch.arange(0, torch.numel(a_dft), device=a_dft.device).float()
    dft_filter /= torch.numel(a_dft)
    if beta > EPSILON:
        dft_filter = (dft_filter - alpha) / beta + 1 / 2
        dft_filter = torch.clamp(dft_filter, 0.0, 1.0)
    else:
        dft_filter = (dft_filter >= alpha).float()

    x_dft = (1 - dft_filter) * a_dft + dft_filter * b_dft
    x_dist = torch.fft.irfft(x_dft, a_dist.shape[0])
    x_values = torch.gather(x_dist, 0, torch.argsort(c_indices))
    return x_values.reshape_as(a)


def ties_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    deltas = []
    signs = []
    for m in [a, b]:
        deltas.append(filter_top_k(m - c, beta))
        signs.append(torch.sign(deltas[-1]))

    signs = torch.stack(signs, dim=0)
    final_sign = torch.sign(torch.sum(signs, dim=0))
    delta_filters = (signs == final_sign).float()

    res = torch.zeros_like(c, device=c.device)
    for delta_filter, delta in zip(delta_filters, deltas):
        res += delta_filter * delta

    param_count = torch.sum(delta_filters, dim=0)
    return c + alpha * torch.nan_to_num(res / param_count)
    
def ties_neuron_sum(
    a: Tensor, b: Tensor, c: Tensor, **kwargs
) -> Tensor:
    if not models:
        return torch.tensor(0)

    key = kwargs["key"]
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = models[0].shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_models = [
                model[t_start:t_end] for model in models
            ]
            vs.append(ties_neuron_sum(*k_models, **k_kwargs))
        return torch.concatenate(vs)

    original_shape = models[0].shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(original_shape) == 1:
        shape_2d = (original_shape[0], 1)
    elif len(original_shape) == 4:
        shape_2d = (original_shape[0], functools.reduce(operator.mul, original_shape[1:]))
    else:
        shape_2d = original_shape

    models = [
        model.reshape(shape_2d)
        for model in models
    ]

    deltas = []
    for m in models:
        deltas.append(filter_top_k(m, k))

    deltas = torch.stack(deltas, dim=0)
    normalized_deltas = deltas / deltas.norm(dim=2, keepdim=True)

    mean_delta = normalized_deltas.mean(dim=0, keepdim=True)  # alt: deltas.mean
    mean_delta = torch.nan_to_num(mean_delta / mean_delta.norm(dim=2, keepdim=True))
    # delta_filters = (1 - torch.cos((mean_delta * normalized_deltas).sum(dim=2, keepdim=True).clamp(0)*torch.pi)) / 2
    delta_filters = (mean_delta * normalized_deltas).sum(dim=2, keepdim=True).clamp(0)
    filtered_delta = (deltas * delta_filters).sum(dim=0)

    param_counts = torch.sum(delta_filters, dim=0)
    return torch.nan_to_num(filtered_delta / param_counts).reshape(original_shape)


def filter_top_k(a: Tensor, k: float):
    k = max(int((1 - k) * torch.numel(a)), 1)
    k_value, _ = torch.kthvalue(torch.abs(a.flatten()).float(), k)
    top_k_filter = (torch.abs(a) >= k_value).float()
    return a * top_k_filter


def rotate(a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs):
    if alpha == 0 and beta == 0:
        return a

    is_conv = len(a.shape) == 4 and a.shape[-1] != 1
    if len(a.shape) == 0 or is_conv or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, beta)

    if len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a_neurons = a.reshape(*shape_2d).double()
    b_neurons = b.reshape(*shape_2d).double()

    a_centroid = a_neurons.mean(0)
    b_centroid = b_neurons.mean(0)
    new_centroid = weighted_sum(a_centroid, b_centroid, alpha)
    if len(a.shape) == 1 or len(a.shape) == 2 and a.shape[0] == 1:
        return new_centroid.reshape_as(a)

    a_neurons -= a_centroid
    b_neurons -= b_centroid

    alpha_is_float = alpha != round(alpha)

    if kwargs["cache"] is not None and "rotation" in kwargs["cache"]:
        rotation = transform = kwargs["cache"]["rotation"].to(a.device)
    else:
        svd_driver = "gesvd" if a.is_cuda else None
        u, _, v_t = torch.linalg.svd(a_neurons.T @ b_neurons, driver=svd_driver)

        if alpha_is_float:
            # cancel reflection. without this, eigenvalues often have a complex component
            #   and then we can't obtain a valid dtype for the merge
            u[:, -1] /= torch.det(u) * torch.det(v_t)

        rotation = transform = u @ v_t
        if not torch.isfinite(u).all():
            raise ValueError(
                textwrap.dedent(
                    f"""determinant error: {torch.det(rotation)}.
                This can happen when merging on the CPU with the "rotate" method.
                Consider merging on a cuda device, or try setting alpha to 1 for the problematic blocks.
                See this related discussion for more info: https://github.com/s1dlx/meh/pull/50#discussion_r1429469484"""
                )
            )

        if kwargs["cache"] is not None:
            kwargs["cache"]["rotation"] = rotation.cpu()

    if alpha_is_float:
        transform = fractional_matrix_power(transform, alpha, kwargs["cache"])
    elif alpha == 0:
        transform = torch.eye(
            len(transform),
            dtype=transform.dtype,
            device=transform.device,
        )
    elif alpha != 1:
        transform = torch.linalg.matrix_power(transform, round(alpha))

    if beta != 0:
        # interpolate the relationship between the neurons
        a_neurons = weighted_sum(a_neurons, b_neurons @ rotation.T, beta)

    a_neurons @= transform
    a_neurons += new_centroid
    return a_neurons.reshape_as(a).to(a.dtype)


def fractional_matrix_power(matrix: Tensor, power: float, cache: dict):
    if cache is not None and "eigenvalues" in cache:
        eigenvalues = cache["eigenvalues"].to(matrix.device)
        eigenvectors = cache["eigenvectors"].to(matrix.device)
        eigenvectors_inv = cache["eigenvectors_inv"].to(matrix.device)
    else:
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        if cache is not None:
            cache["eigenvalues"] = eigenvalues.cpu()
            cache["eigenvectors"] = eigenvectors.cpu()
            cache["eigenvectors_inv"] = eigenvectors_inv.cpu()

    eigenvalues.pow_(power)
    result = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors_inv
    return result.real.to(dtype=matrix.dtype)
    
    
def train_difference(
    a: Tensor, b: Tensor, c: Tensor, *, alpha: float, **kwargs
) -> Tensor:
    threshold = torch.maximum(torch.abs(a - c), torch.abs(b - c))
    dissimilarity = torch.clamp(torch.nan_to_num((c - a) * (b - c) / threshold**2, nan=0), 0)

    return a + (b - c) * alpha * dissimilarity

    
def neuron_train_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs["key"]
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.concatenate(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    threshold = torch.max((a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True))
    dissimilarity = (1 - torch.nan_to_num(((a - c) * (b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0)) / 2

    res = a + (b - c) * alpha * dissimilarity
    return res.reshape(original_shape)
  

def neuron_train_difference_jaccard(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_jaccard(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    dissimilarity = 1 - jaccard_similarity(a, b, c)
    
    res = a + (b - c) * alpha * dissimilarity
    return res.reshape(original_shape)

def jaccard_similarity(a, b, c):
    a = set(torch.where((a - c) != 0)[0].tolist())
    b = set(torch.where((b - c) != 0)[0].tolist())
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union


def neuron_train_difference_euclidean(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_euclidean(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    dissimilarity = torch.norm((a - c) - (b - c), dim=1, keepdim=True)

    res = a + (b - c) * alpha * dissimilarity
    return res.reshape(original_shape)


def neuron_train_difference_cosine(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_cosine(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    dissimilarity = 1 - F.cosine_similarity(a - c, b - c, dim=1, eps=EPSILON).unsqueeze(-1)

    res = a + (b - c) * alpha * dissimilarity
    return res.reshape(original_shape)


def neuron_train_difference_mahalanobis(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_mahalanobis(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d).float()
    b = b.reshape(*shape_2d).float()
    c = c.reshape(*shape_2d).float()

    mean_B = torch.mean(b, dim=0)
    B_centered = b - mean_B
    cov_B = torch.mm(B_centered.T, B_centered) / (b.shape[0] - 1)
    regularization = torch.eye(cov_B.shape[0]) * 1e-5
    cov_B_inv = torch.inverse(cov_B + regularization)

    # Step 2: Center A using the mean of B
    A_centered = a - mean_B

    # Step 3: Apply the inverse covariance matrix to each centered vector of A
    # We reshape A_centered to [50,000, 1, 768] and cov_B_inv to [1, 768, 768] to use torch.bmm for batched matrix multiplication
    temp = torch.bmm(A_centered.unsqueeze(1), cov_B_inv.unsqueeze(0).expand(a.shape[0], -1, -1))
    # The output temp has shape [50,000, 1, 768]. We then perform another bmm with A_centered
    mahalanobis_dists_squared = torch.bmm(temp, A_centered.unsqueeze(-1)).squeeze()
    # mahalanobis_dists now contains the squared Mahalanobis distances
    mahalanobis_dists = torch.sqrt(mahalanobis_dists_squared.clamp(0)).reshape(a.shape[0], -1)

    res = a + (b - c) * alpha * mahalanobis_dists
    return res.reshape(original_shape).half()


def neuron_train_difference_correlation(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_correlation(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    a = (a - torch.mean(a)) / torch.std(a)
    b = (b - torch.mean(b)) / torch.std(b)
    c = (c - torch.mean(c)) / torch.std(c)

    threshold = torch.max((a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True))

    a_diff = (a - c) / threshold
    b_diff = (b - c) / threshold

    mean_a_diff = torch.mean(a_diff, dim=1, keepdim=True)
    mean_b_diff = torch.mean(b_diff, dim=1, keepdim=True)

    numerator = torch.sum((a_diff - mean_a_diff) * (b_diff - mean_b_diff), dim=1, keepdim=True)
    denominator = torch.sqrt(torch.sum((a_diff - mean_a_diff)**2, dim=1, keepdim=True)) * torch.sqrt(torch.sum((b_diff - mean_b_diff)**2, dim=1, keepdim=True)) + EPSILON

    corr_coef = numerator / denominator

    dissimilarity = torch.abs(1 - corr_coef)

    res = a + (b - c) * alpha * dissimilarity
    return res.reshape(original_shape)


#def neuron_train_difference_proj( a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_proj(k_a, k_b, k_c, alpha=alpha, beta=beta, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d) 
    
    a_diff_norm = (a - c) / torch.norm(a - c, dim=1, keepdim=True)
    b_diff_norm = (b - c) / torch.norm(b - c, dim=1, keepdim=True)

    proj_coeff = (a_diff_norm * b_diff_norm).sum(dim=1, keepdim=True) / ((a_diff_norm * a_diff_norm).sum(dim=1, keepdim=True) + EPSILON)
    b_diff_proj = proj_coeff * a_diff_norm

    transfer_strength = torch.norm(a - c, dim=1, keepdim=True)
    beta = beta * (1.0 / (1.0 + torch.exp(-transfer_strength)))

    a_new = a + alpha * (beta * b_diff_proj)

    return a_new.reshape(original_shape)
    
    
def neuron_train_difference_proj(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_train_difference_proj(k_a, k_b, k_c, alpha=alpha, beta=beta, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d) 

    threshold = torch.max((a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True))
    dissimilarity = (1 - torch.nan_to_num(((a - c) * (b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0))

    b_diff_proj = dissimilarity * (b - c)

    transfer_strength = torch.norm(a - c, dim=1, keepdim=True)
    beta = beta * (1.0 / (1.0 + torch.exp(-transfer_strength)))

    a_new = a + alpha * (beta * b_diff_proj)

    return a_new.reshape(original_shape)    
    

def add_perpendicular(
    a: Tensor, b: Tensor, alpha: float, c: Tensor = None, **kwargs
) -> Tensor:
    a_diff = a.float() - c.float()
    b_diff = b.float() - c.float()
    a_ortho = a_diff * (a_diff / torch.linalg.norm(a_diff) * (b_diff / torch.linalg.norm(a_diff))).sum()
    b_perp = b_diff - a_ortho
    res = a + alpha * b_perp
    if torch.isnan(res).any():
        return a
    return res.to(a.dtype)


def rej(a: Tensor, b: Tensor):
    return b * torch.tensordot(a, b, dims=len(a.shape)) / torch.tensordot(b, b, dims=len(a.shape))

def vector_rejection(a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs) -> Tensor:
    ac_diff = a - c
    bc_diff = b - c
    merged = rej(ac_diff, bc_diff)
    return a + merged * alpha
    
    
def slerp(
a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    a_normalized = a / a.norm()
    b_normalized = b / b.norm()

    ab_dot = (a_normalized * b_normalized).sum().clip(-1, 1)

    omega = torch.arccos(ab_dot)
    a_contrib = a_normalized * torch.sin((1-alpha)*omega)
    b_contrib = b_normalized * torch.sin(alpha*omega)
    res = (a_contrib + b_contrib) / torch.sin(omega)
    res *= weighted_sum(a.norm(), b.norm(), alpha=alpha)
    if torch.isnan(res).any():
        return weighted_sum(a, b, alpha=alpha)
    return res


def crossover(
    a: Tensor, b: Tensor, alpha: float, beta: float, **kwargs
) -> Tensor:
    if alpha == 0:
        return a
    if beta == 1:
        return weighted_sum(a, b, alpha=alpha)

    if len(a.shape) == 0 or torch.allclose(a.half(), b.half()):
        return weighted_sum(a, b, alpha=alpha)

    shape = a.shape

    a_dft = torch.fft.rfftn(a, s=shape)
    b_dft = torch.fft.rfftn(b, s=shape)

    dft_filter = create_filter(a_dft.shape, alpha, beta, device=a.device)

    x_dft = (1 - dft_filter)*a_dft + dft_filter*b_dft
    return torch.fft.irfftn(x_dft, s=shape)


def create_filter(shape: Tuple[int, ...] | torch.Size, alpha: float, beta: float, device=None):
    """
    Create a crossover filter. The cut is first tilted, then slid along its normal to match the mean.
    :param shape: shape of the filter
    :param alpha: the frequency of the filter as a ratio, in [0, 1]
    :param beta: tilt of the filter. 0 = vertical filter, 0.5 = 45 degrees, 1 = degenerates to a weighted sum with alpha=mean
    :param steps: maximum number of optimization steps to apply over the mean until the filter converges
    :param precision: the accepted loss between the requested mean and the effective mean of the filter
    :param device: device of the filter
    :return:
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    gradients = list(reversed([
        torch.linspace(0, 1, s, device=device)
        if i == 0 or s == 1 else
        # negative frequencies are in the second half of the dimension
        torch.cat([torch.linspace(0, (s - 1) // 2, s - s // 2, device=device), torch.linspace(s // 2, 1, s // 2, device=device)]) / (s // 2)
        for i, s in enumerate(reversed(shape))
    ]))

    if len(shape) > 1:
        grids = torch.meshgrid(*(g**2 for g in gradients), indexing='ij')
        mesh = (torch.stack(grids).sum(dim=0) / len(shape)).sqrt()
    else:
        mesh = gradients[0]

    if beta < EPSILON:
        dft_filter = (mesh > 1 - alpha).float()
    else:
        tilt_cot = 1 / math.tan(math.pi * beta / 2)
        dft_filter = torch.clamp(mesh*tilt_cot + alpha*tilt_cot + alpha - tilt_cot, 0, 1)

    return dft_filter


def neuron_similarity_add_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, **kwargs
) -> Tensor:
    key = kwargs.get("key", "")
    if key.endswith(("in_proj_weight", "in_proj_bias")):
        # workaround for concatenated attention projection layers
        vs = []
        for i, k in enumerate(("to_q", "to_k", "to_v")):
            k_kwargs = kwargs.copy()
            k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
            dim = a.shape[0] // 3
            t_start = dim*i
            t_end = dim*(i+1)
            k_a = a[t_start:t_end]
            k_b = b[t_start:t_end]
            k_c = c[t_start:t_end]
            vs.append(neuron_similarity_add_difference(k_a, k_b, k_c, alpha=alpha, **k_kwargs))
        return torch.cat(vs)

    original_shape = a.shape

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    threshold = torch.maximum(torch.abs(a), torch.abs(b))
    similarity = ((a * b / threshold**2) + 1) / 2
    similarity = torch.nan_to_num(similarity, nan=0)

    ab_diff = a + alpha * (b - c)
    ab_sum = (1 - alpha / 2) * a + (alpha / 2) * b
    return ((1 - similarity) * ab_diff + similarity * ab_sum).reshape(original_shape)


def score_target_tokens(attn_scores, target_positions, model_b):
  """
  Analyzes attention scores for target tokens and returns a score.

  Args:
      attn_scores: Attention scores from the merged model.
      target_positions: List of positions corresponding to target tokens.
      model_b: The second model used in the merge (potentially containing 
               the desired behavior for target tokens).

  Returns:
      A score (float) representing how well the merged model incorporates 
      the target tokens.
  """

  # Calculate average attention for target tokens from model B
  target_scores = []
  for pos in target_positions:
    # Extract attention scores for the target position across all heads
    head_scores = attn_scores[:, :, pos, :]  # (batch_size, num_heads, seq_len)
    # Get attention weights from model B specifically
    model_b_scores = head_scores[:, :, model_b.get_token_id("TARGET_TOKEN")]  # (batch_size, num_heads)
    # Average attention across heads for this position
    avg_score = torch.mean(model_b_scores, dim=1)  # (batch_size,)
    target_scores.append(avg_score)

  # Combine target scores into a single score (consider different strategies here)
  final_score = torch.mean(torch.stack(target_scores))  # Average score across targets
  return final_score

def get_attn(emb, ret):
    def hook(self, sin, sout):
        h = self.heads
        q = self.to_q(sin[0])
        context = emb
        k = self.to_k(context)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        ret["out"] = attn
    return hook


def merge_token_embeddings(model_a, model_b, base_model="model_a"):
    """
    Merges token embeddings of two models using tokenizer_permute.py from mergekit.

    Args:
        model_a: The first model (PyTorch object).
        model_b: The second model (PyTorch object).
        base_model (str): Which model to use as the base ("model_a" or "model_b").

    Returns:
        A dictionary of merged tensors.
    """

    build_tokenizer = BuildTokenizer(model_a.tokenizer, model_b.tokenizer)
    merge_method = TokenizerPermutationMerge(tokenizer_task=build_tokenizer)

    model_references = {
        "model_a": model_a,
        "model_b": model_b,
    }

    tensors = {}
    for model_ref, model in model_references.items():
        tensors[model_ref] = {
            # Filter for relevant tensors (e.g., embeddings)
            tensor_name: getattr(model, tensor_name)
            for tensor_name in model.state_dict()
            if "token_embedding" in tensor_name  # Example filter
        }

    task = merge_method.make_task(
        tensors=tensors,
        parameters={"embed_slerp": True, "t": 0.5},  
        tensor_parameters={},
        base_model=base_model,
    )
    merged_tensors = task.execute()

    return merged_tensors

# Now you have 'merged_tensors', a dictionary containing the merged tensors
# You can use these as input to your neuron_train_difference method or any other 
# part of your code


#def anchored_guided_alignment(a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, **kwargs) -> Tensor:
    original_shape = a.shape
    shape_2d = (-1, a.numel() // a.shape[0])

    # Align to anchor (simplified mean-based alignment)
    a_neurons = a.reshape(*shape_2d).mean(0) - c.reshape(*shape_2d).mean(0)
    b_neurons = b.reshape(*shape_2d).mean(0) - c.reshape(*shape_2d).mean(0)

    rotation = torch.eye(a_neurons.shape[0], device=a_neurons.device)  # Identity rotation

    aligned_a = (a.reshape(*shape_2d) - a_neurons) @ rotation + a_neurons
    aligned_b = (b.reshape(*shape_2d) - b_neurons) @ rotation + b_neurons

    # Refinement (neuron train difference)
    threshold = torch.max(
        (aligned_a - c).norm(dim=1, keepdim=True), (aligned_b - c).norm(dim=1, keepdim=True)
    )
    dissimilarity = (1 - torch.nan_to_num(((aligned_a - c) * (aligned_b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0)) / 2

    # Merging with weighted sum and dissimilarity adjustment (inline calculation)
    merged_tensor = aligned_a * alpha + aligned_b * (1 - alpha) + (aligned_b - c) * beta * dissimilarity

    return merged_tensor.reshape(original_shape)


def anchored_guided_alignment(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, cache:dict, EPSILON = 1e-8, **kwargs
) -> Tensor:
    """Merges tensors A and B using anchored neuron train difference with simplified alignment and slerp interpolation.

    Args:
        a (Tensor): The first tensor (assumed to be fp16).
        b (Tensor): The second tensor (assumed to be fp16).
        c (Tensor): The anchor tensor (assumed to be fp16).
        alpha (float): The alpha parameter for slerp interpolation (0 <= alpha <= 1).
        beta (float): The beta parameter for dissimilarity adjustment (0 <= beta <= 1).

    Returns:
        Tensor: The merged tensor (in fp16).
    """
    original_shape = a.shape

    # Reshape tensors to 2D
    if not original_shape: # Empty Tensor
        shape_2d = (1, 1)
    elif len(a.shape) == 4: # 4D Tensor (Convolutional)
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    elif len(a.shape) == 1: # 1D Tensor
        shape_2d = (a.shape[0], 1) 
    elif len(a.shape) == 0:  # Scalar (Logit Scale)
        shape_2d = (1, 1)  # Or handle differently as needed
    else: # 2D Tensor
        shape_2d = (-1, a.shape[-1])

    # Convert to float32 for calculations
    a = a.reshape(*shape_2d)
    b = b.reshape(*shape_2d)
    c = c.reshape(*shape_2d)

    # Align to anchor using frequency domain alignment with anchor guidance
    b = mine_alignment(a, b, cache)

    a = a.float()
    b = b.float()
    c = c.float()

    # Refinement (neuron train difference)
    threshold = torch.max(
        (a - c).norm(dim=1, keepdim=True), (b - c).norm(dim=1, keepdim=True)
    )
    dissimilarity = (1 - torch.nan_to_num(((a - c) * (b - c)).sum(dim=1, keepdim=True) / threshold**2, nan=0)) / 2

    # Merging with slerp (or nlerp) and dissimilarity adjustment
    a_norm = a / (a.norm(dim=1, keepdim=True) + EPSILON)
    b_norm = b / (b.norm(dim=1, keepdim=True) + EPSILON)

    ab_dot = (a_norm * b_norm).sum(dim=1, keepdim=True).clip(-1 + EPSILON, 1 - EPSILON)
    omega = torch.acos(ab_dot).clip(EPSILON, math.pi - EPSILON) 

    sin_omega = torch.sin(omega) + EPSILON

    a_contrib = a_norm * torch.sin((1 - alpha) * omega) / sin_omega  
    b_contrib = b_norm * torch.sin(alpha * omega) / sin_omega

    # Use nlerp if vectors are close to parallel or antiparallel
    if torch.all(1 - torch.abs(ab_dot) < EPSILON):
        merged_tensor = nlerp(aligned_a_norm, b_norm, alpha=alpha) * (
            torch.norm(a, dim=1, keepdim=True) * (1 - alpha)
            + torch.norm(b, dim=1, keepdim=True) * alpha    
        )
    else:
        merged_tensor = (a_contrib + b_contrib) * (
            torch.norm(a, dim=1, keepdim=True) * (1 - alpha) 
            + torch.norm(b, dim=1, keepdim=True) * alpha    
        )

    merged_tensor += (b - c) * beta * dissimilarity  

    return merged_tensor.reshape(original_shape).half()

    
def frequency_selective_alignment(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Aligns tensor 'b' to 'a' in specific frequency bands, guided by 'c', by directly adjusting magnitudes.

    Args:
        a (Tensor): The first tensor (2D).
        b (Tensor): The tensor to be aligned (2D).
        c (Tensor): The anchor tensor (2D).

    Returns:
        Tensor: The aligned tensor 'b'.
    """

    # Reshape to 1D
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    c_flat = c.reshape(-1).float()

    # Apply FFT along the feature dimension (dim=1)
    a_dft = torch.fft.rfft(a_flat)
    b_dft = torch.fft.rfft(b_flat)
    c_dft = torch.fft.rfft(c_flat)

    # Calculate spectral centroids 
    a_centroid = calculate_spectral_centroid(a_dft)
    b_centroid = calculate_spectral_centroid(b_dft)
    c_centroid = calculate_spectral_centroid(c_dft)

    # Dynamic beta based on spectral centroid distance
    dissimilarity = abs(a_centroid - b_centroid)
    normalized_dissimilarity = dissimilarity / (c_centroid + 1e-8)
    dynamic_beta = 1 - normalized_dissimilarity

    # Use spectral centroids to define passband and stopband
    passband = (0, int(min(a_centroid, c_centroid) * a_dft.shape[0]))
    stopband = (int(max(a_centroid, c_centroid) * a_dft.shape[0]), a_dft.shape[0])

    # Define transition_start and transition_end here
    transition_start = passband[1]
    transition_end = stopband[0]
    
    # Calculate magnitude difference between 'a' and 'c'
    a_dft_magnitude = torch.abs(a_dft[passband[0]:passband[1]])
    b_dft_magnitude = torch.abs(b_dft[passband[0]:passband[1]])
    c_dft_magnitude = torch.abs(c_dft[passband[0]:passband[1]])

    # Weighted average of 'a' and 'b' magnitudes using dynamic_beta
    weighted_magnitude = (1 - dynamic_beta) * a_dft_magnitude + dynamic_beta * b_dft_magnitude

    magnitude_difference = weighted_magnitude - c_dft_magnitude

    # Apply smooth magnitude adjustment using sigmoid
    transition_width = transition_end - transition_start  
    if transition_width > 0: # Prevent division by zero
        transition_slope = 1.0 / (transition_width + 1e-8)  # Add epsilon
        smooth_adjustment = torch.sigmoid(transition_slope * (torch.arange(magnitude_difference.shape[0]) - transition_start)) 
        b_dft[passband[0]:passband[1]] += magnitude_difference * smooth_adjustment

    # Apply inverse FFT to get the aligned tensor in the time domain
    aligned_b = torch.fft.irfft(b_dft, a_flat.shape[0]) 

    return aligned_b.reshape(a.shape)

def calculate_spectral_centroid(dft: torch.Tensor) -> float:
    """
    Calculates the spectral centroid of a tensor in the frequency domain.

    Args:
        dft (torch.Tensor): The tensor's Fourier Transform.

    Returns:
        float: The spectral centroid.
    """
    frequencies = torch.arange(dft.shape[0])
    magnitudes = torch.abs(dft)
    centroid = (frequencies * magnitudes).sum() / magnitudes.sum() 
    return centroid.item()
 
def adversarial_domain_adaptation(b: Tensor, a: Tensor, cache: dict, num_epochs: int = 2) -> Tensor:
    """Aligns tensor 'b' to the target distribution of 'a' using adversarial domain adaptation.

    Args:
        b (Tensor): The tensor to be aligned (2D).
        a (Tensor): The first tensor representing the target distribution (2D).
        num_epochs (int): The number of training epochs.

    Returns:
        Tensor: The aligned tensor 'b'.
    """
    # Check if aligned_b is in the cache
    if cache is not None and "aligned_b" in cache:
        return cache["aligned_b"]
        
    class Discriminator(nn.Module):
        def __init__(self, input_size):
            super(Discriminator, self).__init__()
            self.fc1 = nn.utils.spectral_norm(nn.Linear(input_size, 256))  # Spectral norm before dropout
            self.leaky_relu1 = nn.LeakyReLU(0.2)
            self.dropout1 = nn.Dropout(p=0.5)  
            self.fc2 = nn.utils.spectral_norm(nn.Linear(256, 128))  # Spectral norm before dropout
            self.leaky_relu2 = nn.LeakyReLU(0.2)
            self.dropout2 = nn.Dropout(p=0.5) 
            self.fc3 = nn.utils.spectral_norm(nn.Linear(128, 64))  # Spectral norm before dropout
            self.leaky_relu3 = nn.LeakyReLU(0.2)
            self.dropout3 = nn.Dropout(p=0.5)  
            self.fc4 = nn.utils.spectral_norm(nn.Linear(64, 1))  # Spectral norm before dropout
            self.sigmoid = nn.Sigmoid()

            # He initialization
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)
            torch.nn.init.xavier_uniform_(self.fc4.weight)

        def forward(self, x):
            x = self.leaky_relu1(self.fc1(x))
            x = self.dropout1(x)  # Apply dropout
            x = self.leaky_relu2(self.fc2(x))
            x = self.dropout2(x)  # Apply dropout
            x = self.leaky_relu3(self.fc3(x))
            x = self.dropout3(x)  # Apply dropout
            x = self.sigmoid(self.fc4(x))
            return x


    class Encoder(nn.Module):
        def __init__(self, input_size):
            super(Encoder, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=0.5)  # Dropout after the first layer
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=0.5)  # Dropout after the second layer
            self.fc3 = nn.Linear(128, input_size)

            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.dropout1(x)  # Apply dropout
            x = self.relu2(self.fc2(x))
            x = self.dropout2(x)  # Apply dropout
            x = self.fc3(x)
            return x
 
    # Initialize networks and optimizer
    input_size = b.shape[1]
    discriminator = Discriminator(input_size).to('cuda').bfloat16()  # Cast to bfloat16
    encoder = Encoder(input_size).to('cuda').bfloat16()            # Cast to bfloat16
    optimizer = CAME(
        list(discriminator.parameters()) + list(encoder.parameters()),
        lr=1e-5,
        weight_decay=1e-1,
        betas=(0.9, 0.999, 0.9995),
        eps=(1e-30, 1e-16)
    )

    # Calculate total scheduler steps based on epochs and batch size
    batch_size = 64
    min_effective_steps = 200
    accumulation_steps = 1  # Accumulate gradients over 4 batches
    total_steps = num_epochs * (len(b) // (batch_size * accumulation_steps)) 
    total_steps = max(1, total_steps)
    scheduler = REXScheduler(optimizer, total_steps, max_lr=1e-5, min_lr=1e-6)
  
    # Open the log file for writing
    log_file = open("training_log.txt", "a")

    # Log summaries of 'a' and 'b' before batching
    log_file.write(f"a - Mean: {torch.mean(a).item():.4f}, Std: {torch.std(a).item():.4f}, Min: {torch.min(a).item():.4f}, Max: {torch.max(a).item():.4f}\n")
    log_file.write(f"b - Mean: {torch.mean(b).item():.4f}, Std: {torch.std(b).item():.4f}, Min: {torch.min(b).item():.4f}, Max: {torch.max(b).item():.4f}\n")
  
    step_counter = 0
    effective_step_counter = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for batch_start in range(0, len(b), batch_size):
            
            actual_batch_size = min(batch_size, len(b) - batch_start)
            batch_end = batch_start + actual_batch_size

            # Slice batches directly from the 2D tensors
            b_batch = b[batch_start:batch_end].bfloat16() 
            a_batch = a[batch_start:batch_end].bfloat16()

            # Reshape to 2D
            b_batch = b_batch.reshape(actual_batch_size, -1)  # Add a dimension for features
            a_batch = a_batch.reshape(actual_batch_size, -1)  # Add a dimension for features

            # Encode features from 'b'
            encoded_b = encoder(b_batch)

            # Discriminator predictions
            preds_encoded_b = discriminator(encoded_b)
            preds_a = discriminator(a_batch)

            alpha = 0.5  # Initial small weight for reconstruction loss

            # Additional reconstruction loss
            reconstruction_loss = torch.mean((b_batch - encoded_b)**2)

            # Combined loss
            loss = 0.5 * torch.mean((preds_a - 1)**2) + 0.5 * torch.mean(preds_encoded_b**2) + alpha * reconstruction_loss
            
            # Gradient Accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (step_counter + 1) % accumulation_steps == 0:
                # Clip accumulated gradients
                torch.nn.utils.clip_grad_norm_(list(discriminator.parameters()) + list(encoder.parameters()), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # Increment effective step counter
                effective_step_counter += 1

            # Log summaries every 10 steps
            if step_counter % 10 == 0:
                log_file.write(f"Epoch {epoch+1}/{num_epochs}, Step {step_counter}\n")
                log_file.write(f"b_batch - Mean: {torch.mean(b_batch).item():.4f}, Std: {torch.std(b_batch).item():.4f}, Min: {torch.min(b_batch).item():.4f}, Max: {torch.max(b_batch).item():.4f}\n")
                log_file.write(f"a_batch - Mean: {torch.mean(a_batch).item():.4f}, Std: {torch.std(a_batch).item():.4f}, Min: {torch.min(a_batch).item():.4f}, Max: {torch.max(a_batch).item():.4f}\n")
                log_file.write(f"encoded_b - Mean: {torch.mean(encoded_b).item():.4f}, Std: {torch.std(encoded_b).item():.4f}, Min: {torch.min(encoded_b).item():.4f}, Max: {torch.max(encoded_b).item():.4f}\n")
                log_file.write(f"preds_encoded_b - Mean: {torch.mean(preds_encoded_b).item():.4f}, Std: {torch.std(preds_encoded_b).item():.4f}, Min: {torch.min(preds_encoded_b).item():.4f}, Max: {torch.max(preds_encoded_b).item():.4f}\n")
                log_file.write(f"preds_a - Mean: {torch.mean(preds_encoded_b).item():.4f}, Std: {torch.std(preds_encoded_b).item():.4f}, Min: {torch.min(preds_encoded_b).item():.4f}, Max: {torch.max(preds_encoded_b).item():.4f}\n")
                log_file.write(f"loss: {loss.item():.4f}\n")  # Log the loss
                log_file.write("------------------------------------\n")

            # Increment step counter
            step_counter += 1

            # Check and ensure minimum effective steps
            if effective_step_counter < min_effective_steps:
                print(f"Epoch {epoch+1}/{num_epochs}, Effective Step {effective_step_counter}: "
                      f"Continuing to ensure minimum effective steps.")
                continue

        print(f"Epoch {epoch+1}/{num_epochs}, Total Steps: {step_counter}, Effective Steps: {effective_step_counter}")
        # Reset counters for the next epoch if necessary
        effective_step_counter = 0
                
    # Align 'b' using the trained encoder
    aligned_b = encoder(b.bfloat16()).float()  

    # Store aligned_b in the cache
    if cache is not None:
        cache["aligned_b"] = aligned_b.to("cpu")

    # Manually clear the cache
    torch.cuda.empty_cache()
    
    # Close the log file
    log_file.close()

    return aligned_b
    
def gradient_penalty(discriminator, a_batch, fake_data, lambda_gp=10):
    """Calculates the gradient penalty.

    Args:
        discriminator: The discriminator network.
        a_batch: Real data batch.
        fake_data: Generated data batch.
        lambda_gp: Gradient penalty coefficient.

    Returns:
        Tensor: The gradient penalty.
    """
    alpha = torch.rand(a_batch.size(0), 1, device=a_batch.device).expand_as(a_batch).bfloat16()
    interpolated = alpha * a_batch + (1 - alpha) * fake_data

    # Calculate discriminator output
    disc_interpolates = discriminator(interpolated)

    # Calculate gradients of the discriminator output w.r.t. the interpolated data
    gradients = grad(
        outputs=disc_interpolates,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0].view(a_batch.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


def cca_slerp_merge(a: Tensor, b: Tensor, alpha: float, k: int = 128, **kwargs) -> Tensor:
    """Merges tensors A and B using CCA for alignment and Slerp for blending.

    Args:
        a (Tensor): The first tensor (2D).
        b (Tensor): The second tensor (2D).
        alpha (float): The alpha parameter for Slerp interpolation.
        k (int): The number of dimensions to align using CCA.

    Returns:
        Tensor: The merged tensor (float32).
    """
    print("Entering cca_slerp_merge function")  # Check if function is called
    original_shape = a.shape
    print(f"Original Shape: {original_shape}") # Print original shape of tensor

    if not original_shape:
        shape_2d = (1, 1)
    elif len(a.shape) == 4:
        shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
    else:
        shape_2d = (-1, a.shape[-1])

    a = a.reshape(*shape_2d).float()
    b = b.reshape(*shape_2d).float()
    
    print("Calling cca_alignment function") # Check if function is called

    # Align the features using CCA
    aligned_a, aligned_b = cca_alignment(a, b, k=k) 

    print("Finished cca_alignment function")  # Check if function completes

    # Reshape aligned tensors back to original shape before Slerp
    aligned_a = aligned_a.reshape(original_shape) 
    aligned_b = aligned_b.reshape(original_shape)

    print("Calling slerp2 function") # Check if function is called

    # Apply Slerp interpolation to the aligned features
    merged_tensor = slerp2(aligned_a, aligned_b, alpha=alpha)

    print("Finished slerp2 function")  # Check if function completes

    return merged_tensor.float()  # Return as float32 for consistency

def cca_alignment(a, b, k=128):  # Renamed features_a and features_b to a and b
    """Aligns the features of two models using CCA.

    Args:
        a (Tensor): Features from model A (2D).
        b (Tensor): Features from model B (2D).
        k (int): Number of dimensions to align.

    Returns:
        Tuple[Tensor, Tensor]: Aligned features for A and B.
    """
    print("Entering cca_alignment function") # Check if function is called

    # Center the data
    centered_a = a - a.mean(dim=0)
    centered_b = b - b.mean(dim=0)
    
    print("Calculating covariance matrices") # Check if covariance calculation is reached
    print(f"Shape of a: {a.shape}") # Print shape of tensor a
    print(f"Shape of b: {b.shape}") # Print shape of tensor b
    print(f"Shape of centered_a: {centered_a.shape}") # Print shape of centered_a
    print(f"Shape of centered_b: {centered_b.shape}") # Print shape of centered_b

    # Calculate covariance matrices
    cov_a = torch.cov(centered_a.T)
    cov_b = torch.cov(centered_b.T)

    # Add regularization AFTER cov_b is calculated 
    cov_b = cov_b + 1e-6 * torch.eye(cov_b.shape[0])  # Add regularization
    
    print(f"Shape of cov_a: {cov_a.shape}") # Print shape of cov_a
    print(f"Shape of cov_b: {cov_b.shape}") # Print shape of cov_b
    
    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = torch.linalg.eig(torch.matmul(cov_a, torch.inverse(cov_b)))
    
    print("Solved the eigenvalue problem") # Check if eigenvalue calculation is completed

    # Sort eigenvalues and select top 'k'
    top_indices = torch.argsort(eigenvalues.real, descending=True)[:k]
    top_eigenvectors = eigenvectors[:, top_indices]

    # Create transformation matrices
    transform_b = top_eigenvectors.real.T

    # Project the features
    aligned_a = centered_a
    aligned_b = torch.matmul(centered_b, transform_b.T)
    
    print("Exiting cca_alignment function") # Check if function completes

    return aligned_a, aligned_b


def slerp2(a: Tensor, b: Tensor, *, alpha: float, **kwargs) -> Tensor:
    a_normalized = a / (a.norm() + EPSILON)
    b_normalized = b / (b.norm() + EPSILON)

    ab_dot = (a_normalized * b_normalized).sum().clip(-1 + EPSILON, 1 - EPSILON)
    
    # if the vectors are almost parallel or anti-parallel, use nlerp
    if 1 - torch.abs(ab_dot) < EPSILON:
        return nlerp(a_normalized, b_normalized, alpha=alpha) * weighted_sum(a.norm(), b.norm(), alpha=alpha)
    
    omega = torch.acos(ab_dot)
    sin_omega = torch.sin(omega) + EPSILON  
    
    a_contrib = a_normalized * torch.sin((1 - alpha) * omega) / sin_omega
    b_contrib = b_normalized * torch.sin(alpha * omega) / sin_omega

    res = (a_contrib + b_contrib) * (
        (a.norm() + EPSILON) * alpha 
        + (b.norm() + EPSILON) * (1 - alpha)
    )
    return res

def nlerp(a: Tensor, b: Tensor, *, alpha: float, **kwargs) -> Tensor:
    return (a * alpha + b * (1 - alpha)).div(torch.norm(a * alpha + b * (1 - alpha)) + 1e-8)


def literal_train_difference(
    a: Tensor, b: Tensor, c: Tensor, alpha: float, beta: float, cache: dict, **kwargs
) -> Tensor:
    try:
        key = kwargs["key"]
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim*i
                t_end = dim*(i+1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                k_c = c[t_start:t_end]
                vs.append(literal_train_difference(k_a, k_b, k_c, alpha=alpha, beta=beta, cache=cache, **k_kwargs))
            return torch.concatenate(vs)
            
        # Reshape tensors to 2D
        original_shape = a.shape
        if not original_shape: # Empty Tensor
            shape_2d = (1, 1)
        elif len(a.shape) == 4: # 4D Tensor (Convolutional)
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif len(a.shape) == 1: # 1D Tensor
            shape_2d = (a.shape[0], 1) 
        elif len(a.shape) == 0:  # Scalar (Logit Scale)
            shape_2d = (1, 1)  # Or handle differently as needed
        else: # 2D Tensor
            shape_2d = (-1, a.shape[-1])

        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)
        c = c.reshape(*shape_2d)

        # Run MINE alignment
        b = mine_alignment(b, a, cache)

        # --- Use statistic_net for dissimilarity calculation ---

        statistic_net = cache.get("statistic_net", None)  
        assert statistic_net is not None, "statistic_net is missing from the cache!"
        statistic_net = statistic_net.to(a.device).half()

        exponent = 2
        dissimilarity = cache.get("dissimilarity", None)
        if dissimilarity is not None:
            dissimilarity = dissimilarity.to(a.device)
        else:
            with torch.no_grad():
                dissimilarity = torch.zeros_like(a, dtype=a.dtype, device=a.device)
                for i in range(a.shape[0]):
                    similarity = statistic_net(a[i].unsqueeze(0), b[i].unsqueeze(0))
                    dissimilarity[i] = torch.exp((1 - torch.sigmoid(similarity).squeeze()) * exponent) - 1 
            cache["dissimilarity"] = dissimilarity.to("cpu") # Move to CPU before caching!

        # --- Merging with Feature-Wise Dissimilarity ---
        merged_tensor = a + (b - c) * alpha * (dissimilarity * beta)
        print(f"Merged Tensor - Mean: {torch.mean(merged_tensor).item():.4f}, "
                                f"Std: {torch.std(merged_tensor).item():.4f}, "
                                f"Min: {torch.min(merged_tensor).item():.4f}, "
                                f"Max: {torch.max(merged_tensor).item():.4f}")
                                    
        # --- Move tensors back to CPU ---
        statistic_net = statistic_net.to("cpu")
        dissimilarity = dissimilarity.to("cpu") 

        # --- Manually trigger garbage collection ---
        #gc.collect()
        torch.cuda.empty_cache()

        return merged_tensor.reshape(original_shape)
        
    except Exception as e:
        # Print the traceback for immediate feedback
        import traceback
        traceback.print_exc() 
        # Re-raise the exception so the training process stops
        raise


class StatisticNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, nhead, nlayers, batch_size):
        super(StatisticNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Positional Encoding (Learned)
        self.pos_encoder = nn.Embedding(batch_size, embedding_dim)  

        # Create encoder layer with pre-Layer Normalization
        encoder_layer = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, norm_first=False)  
        self.transformer_encoder_a = TransformerEncoder(encoder_layer, nlayers)
        self.transformer_encoder_b = TransformerEncoder(encoder_layer, nlayers)

        # Separate embedding layer for cross-attention
        self.cross_attn_embedding = nn.Linear(embedding_dim, embedding_dim) 
        self.cross_attn = nn.MultiheadAttention(embedding_dim, nhead)
        self.fc = nn.Linear(embedding_dim * 2, 1) 

    def forward(self, a, b):
        # Embed the input tensors
        a_embedded = self.embedding(a)
        b_embedded = self.embedding(b)

        # Add positional encoding
        pos_indices_a = torch.arange(a_embedded.size(0), device=a_embedded.device).clamp(0, self.pos_encoder.num_embeddings - 1)
        pos_indices_b = torch.arange(b_embedded.size(0), device=b_embedded.device).clamp(0, self.pos_encoder.num_embeddings - 1)
    
        # Print min and max values of positional indices
#        print(f"Min index in pos_indices_a: {pos_indices_a.min().item()}, Max index in pos_indices_a: {pos_indices_a.max().item()}")
#        print(f"Min index in pos_indices_b: {pos_indices_b.min().item()}, Max index in pos_indices_b: {pos_indices_b.max().item()}")

        a_embedded = a_embedded + self.pos_encoder(pos_indices_a)
        b_embedded = b_embedded + self.pos_encoder(pos_indices_b)

        # Apply self-attention to each tensor
        a_encoded = self.transformer_encoder_a(a_embedded.unsqueeze(0)).squeeze(0)  
        b_encoded = self.transformer_encoder_b(b_embedded.unsqueeze(0)).squeeze(0)  

        # Apply cross-attention, with 'a' as query and 'b' as key and value
        # Apply separate embedding for b in cross-attention
        b_encoded_cross_attn = self.cross_attn_embedding(b_encoded) 
        attn_output, attn_output_weights = self.cross_attn(a_encoded.unsqueeze(0),
                                                         b_encoded_cross_attn.unsqueeze(0), 
                                                         b_encoded_cross_attn.unsqueeze(0))
        attn_output = attn_output.squeeze(0) # [batch_size, embedding_dim]

        # Concatenate the encoded representations
        z = torch.cat((a_encoded, attn_output), dim=1) # [batch_size, embedding_dim * 2]

        # Map to a scalar output
        output = self.fc(z) 
        return output


# BaseNetwork remains similar to StatisticNetwork, but without cross-attention
class BaseNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, nhead, nlayers, batch_size):
        super(BaseNetwork, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Positional Encoding (Learned)
        self.pos_encoder = nn.Embedding(batch_size, embedding_dim)  

        # Create encoder layer with pre-Layer Normalization
        encoder_layer = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, norm_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)

        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, a, b):
        # Embed the input tensors
        a_embedded = self.embedding(a)
        b_embedded = self.embedding(b)

        # Add positional encoding
        pos_indices_a = torch.arange(a_embedded.size(0), device=a_embedded.device).clamp(0, self.pos_encoder.num_embeddings - 1)
        pos_indices_b = torch.arange(b_embedded.size(0), device=b_embedded.device).clamp(0, self.pos_encoder.num_embeddings - 1)
    
        # Print min and max values of positional indices
#        print(f"Min index in pos_indices_a: {pos_indices_a.min().item()}, Max index in pos_indices_a: {pos_indices_a.max().item()}")
#        print(f"Min index in pos_indices_b: {pos_indices_b.min().item()}, Max index in pos_indices_b: {pos_indices_b.max().item()}")

        a_embedded = a_embedded + self.pos_encoder(pos_indices_a)
        b_embedded = b_embedded + self.pos_encoder(pos_indices_b)

        # Apply self-attention to each tensor independently
        a_encoded = self.transformer_encoder(a_embedded.unsqueeze(0)).squeeze(0)
        b_encoded = self.transformer_encoder(b_embedded.unsqueeze(0)).squeeze(0)

        # Combine the encoded representations
        z = torch.cat((a_encoded, b_encoded), dim=1)

        # Map to a scalar output
        output = self.fc(z)
        return output


def mine_alignment(b: Tensor, a: Tensor, cache: dict, num_epochs: int = 5) -> Tensor:
    """Aligns tensor 'b' to the target distribution of 'a' using MINE.

    Args:
        b (Tensor): The tensor to be aligned (2D).
        a (Tensor): The first tensor representing the target distribution (2D).

    Returns:
        Tensor: The aligned tensor 'b'.
    """
    # Check if statistic_net is in the cache (Corrected Check!)
    if cache is not None and "statistic_net" in cache:
        return b  # Return original 'b' since latent spaces are already aligned 
        
        
    def mine_loss(t_joint, t_marginal, similarity_penalty=0.0218):
        similarity_regularization = similarity_penalty * torch.mean(t_joint**2)
        return -(torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))) + similarity_regularization


    # Initialize networks and optimizer
    input_dim = b.shape[1]
    num_epochs = 5
    hidden_dim = 512
    embedding_dim = 512  # Adjust embedding dimension as needed
    nhead = 4          # Number of attention heads
    nlayers = 2         # Number of transformer layers
    batch_size = 64
    accumulation_steps = 4
    lr = 1e-6
    max_lr = 1e-6
    min_lr = 1e-7
    weight_decay = 1e-2
    statistic_net = StatisticNetwork(input_dim, hidden_dim, embedding_dim, nhead, nlayers, batch_size).to('cuda').bfloat16()
    base_net = BaseNetwork(input_dim, hidden_dim, embedding_dim, nhead, nlayers, batch_size).to('cuda').bfloat16()
    optimizer = CAME(
        list(statistic_net.parameters()) + list(base_net.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999, 0.9995),
        eps=(1e-30, 1e-16)
    )


    # Calculate total scheduler steps
    total_steps = num_epochs * ((len(b) + batch_size * accumulation_steps - 1) // (batch_size * accumulation_steps))
    total_steps = max(1, total_steps) 
    scheduler = REXScheduler(optimizer, total_steps, max_lr=max_lr, min_lr=min_lr) 


    # --- Training Loop ---
    step_counter = 0
    optimizer.zero_grad() 


    for epoch in range(num_epochs):
        for batch_start in range(0, len(b), batch_size):
            batch_end = min(batch_start + batch_size, len(b))  # Ensure batch_end is within bounds
            b_batch = b[batch_start:batch_end].bfloat16() 
            a_batch = a[batch_start:batch_end].bfloat16()
            
#            print("b_batch shape:", b_batch.shape)
#            print("a_batch shape:", a_batch.shape)
#            print(f"Batch Start: {batch_start}, Batch End: {batch_end}")
#            print(f"Processing key: {key} in epoch {epoch+1}, batch starting at {batch_start}")

            # --- MINE Calculation ---
            t_joint = statistic_net(b_batch, a_batch)
            t_marginal = base_net(b_batch, a_batch) 


            # MINE Loss
            loss = mine_loss(t_joint, t_marginal)


            # Gradient Accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            
            # Gradient Clipping
#            torch.nn.utils.clip_grad_norm_(statistic_net.parameters(), max_norm=1.0)
#            torch.nn.utils.clip_grad_norm_(base_net.parameters(), max_norm=1.0)


            # --- Optimization and Logging ---
            if (step_counter + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()


            step_counter += 1


            # Print loss every 10 steps
            if step_counter % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step_counter}, Loss: {loss.item():.4f}")

    # Store aligned_b in the cache
    if cache is not None:
        cache["statistic_net"] = statistic_net.to("cpu")

    # --- Print the trained statistic_net ---
#    print(f"Trained statistic_net:\n{statistic_net}")

    # --- Print the shapes of weights and biases ---
#    for name, param in statistic_net.named_parameters():
#        print(f"Parameter: {name}, Shape: {param.shape}")
    
# --- Delete unnecessary variables ---
#    del base_net
#    del optimizer
#    del scheduler

# --- Manually trigger garbage collection ---
#    gc.collect()
    torch.cuda.empty_cache()


    return b  # Return the original 'b'


def merge_with_transformers(
    a: Tensor, b: Tensor, alpha: float, cache: dict, **kwargs
) -> Tensor:
    try:
        key = kwargs["key"]
        if key.endswith(("in_proj_weight", "in_proj_bias")):
            vs = []
            for i, k in enumerate(("to_q", "to_k", "to_v")):
                k_kwargs = kwargs.copy()
                k_kwargs["key"] = key.replace("in_proj_", f"{k}.")
                dim = a.shape[0] // 3
                t_start = dim*i
                t_end = dim*(i+1)
                k_a = a[t_start:t_end]
                k_b = b[t_start:t_end]
                vs.append(merge_with_transformers(k_a, k_b, alpha=alpha, cache=cache, **k_kwargs))
            return torch.concatenate(vs)
            
        # Reshape tensors to 2D
        original_shape = a.shape
        if not original_shape: # Empty Tensor
            shape_2d = (1, 1)
        elif len(a.shape) == 4: # 4D Tensor (Convolutional)
            shape_2d = (-1, functools.reduce(operator.mul, a.shape[1:]))
        elif len(a.shape) == 1: # 1D Tensor
            shape_2d = (a.shape[0], 1)
        elif len(a.shape) == 0: # Scalar (Logit Scale)
            shape_2d = (1, 1) # Or handle differently as needed
        else: # 2D Tensor
            shape_2d = (-1, a.shape[-1])
            
        a = a.reshape(*shape_2d)
        b = b.reshape(*shape_2d)

        # Run MINE alignment
        b = mine_alignment(b, a, cache)

        # --- Use statistic_net for attention-based merging ---

        statistic_net = cache.get("statistic_net", None)  
        assert statistic_net is not None, "statistic_net is missing from the cache!"
        statistic_net = statistic_net.to(a.device).half()
        
        # --- Calculate Similarity ---
        similarity = torch.mean(statistic_net(b, a)) 
        print(f"Similarity: {similarity:.4f}")

        # --- Calculate Dissimilarity (Bounded) ---
        dissimilarity = cache.get("dissimilarity", None)
        if dissimilarity is not None:
            # Move dissimilarity back to GPU *only if it's in cache*
            dissimilarity = dissimilarity.to(a.device).half()
        else:
            with torch.no_grad():
                dissimilarity = 1 - similarity 
            cache["dissimilarity"] = dissimilarity.to("cpu")  # Move to CPU after calculation
            
        print(f"Dissimilarity: {dissimilarity:.4f}")

        # --- Global Merging ---
        merged_tensor = a * alpha + b * (1 - alpha) * dissimilarity  
        print(f"Merged Tensor - Mean: {torch.mean(merged_tensor).item():.4f}, "
                                f"Std: {torch.std(merged_tensor).item():.4f}, "
                                f"Min: {torch.min(merged_tensor).item():.4f}, "
                                f"Max: {torch.max(merged_tensor).item():.4f}")

        # --- Move tensors back to CPU ---
        statistic_net = statistic_net.to("cpu")
        dissimilarity = dissimilarity.to("cpu")  # Move back to CPU after merging
        
        # --- Manually trigger garbage collection ---
        #gc.collect()
        torch.cuda.empty_cache()

        return merged_tensor.reshape(original_shape)

    except Exception as e:
        # Print the traceback for immediate feedback
        import traceback
        traceback.print_exc() 
        # Re-raise the exception so the training process stops
        raise
