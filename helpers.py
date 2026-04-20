import numpy as np
import torch

def _l2_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim)


def estimate_invariant_error_covariance(errors: torch.Tensor) -> torch.Tensor:
    """Compute E[err err^T] for invariant pose errors.

    Args:
        errors: Tensor of shape (N, 3) containing Lie algebra errors.

    Returns:
        (3, 3) covariance matrix.
    """
    if errors.ndim != 2:
        raise ValueError(
            f"Expected errors with shape (N, 3) for covariance estimation, got {errors.shape}"
        )

    outer_products = errors.unsqueeze(-1) @ errors.unsqueeze(-2)
    return outer_products.mean(dim=0)


def estimate_mean_and_covariance(errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate bias (mean) and centered covariance of Lie-algebra errors.

    Args:
        errors: (N, 3) Lie algebra errors.

    Returns:
        bias: (3,) mean of errors.
        covariance: (3, 3) covariance of mean-centered errors.
    """
    if errors.ndim != 2:
        raise ValueError(
            f"Expected errors with shape (N, 3) for mean/covariance estimation, got {errors.shape}"
        )

    bias = errors.mean(dim=0)
    centered = errors - bias
    covariance = estimate_invariant_error_covariance(centered)
    return bias, covariance
