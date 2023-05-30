import torch


__all__ = [
    "weighted_sum",
    "weighted_subtraction",
    "tensor_sum",
    "add_difference",
    "sum_twice",
    "triple_sum",
    "euclidian_add_difference",
]


EPSILON = 1e-10  # Define a small constant EPSILON to prevent division by zero


def weighted_sum(a: torch.Tensor, b: torch.Tensor, alpha: float, **kwargs) -> torch.Tensor:
    return (1 - alpha) * a + alpha * b


def weighted_subtraction(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float, **kwargs) -> torch.Tensor:
    # Adjust beta if both alpha and beta are 1.0 to avoid division by zero
    if alpha == 1.0 and beta == 1.0:
        beta -= EPSILON

    return (a - alpha * beta * b) / (1 - alpha * beta)


def tensor_sum(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float, **kwargs) -> torch.Tensor:
    if alpha + beta <= 1:
        tt = a.clone()
        talphas = int(a.shape[0] * beta)
        talphae = int(a.shape[0] * (alpha + beta))
        tt[talphas:talphae] = b[talphas:talphae].clone()
    else:
        talphas = int(a.shape[0] * (alpha + beta - 1))
        talphae = int(a.shape[0] * beta)
        tt = b.clone()
        tt[talphas:talphae] = a[talphas:talphae].clone()
    return tt


def add_difference(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, **kwargs) -> torch.Tensor:
    return a + alpha * (b - c)


def sum_twice(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, beta: float, **kwargs) -> torch.Tensor:
    return (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c


def triple_sum(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, beta: float, **kwargs) -> torch.Tensor:
    return (1 - alpha - beta) * a + alpha * b + beta * c


def euclidian_add_difference(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, **kwargs) -> torch.Tensor:
    distance = (a - c) ** 2 + alpha * (b - c) ** 2
    try:
        distance = torch.sqrt(distance)
    except RuntimeError:
        distance = torch.sqrt(distance.float()).half()
    distance = torch.copysign(distance, a + b - 2 * c)
    norm = (torch.linalg.norm(a - c) + torch.linalg.norm(b - c)) / 2
    return c + distance / torch.linalg.norm(distance) * norm
