import torch
import math
from typing import Optional, Tuple
from scipy.stats import chi2
from scipy.special import gamma

from torch.distributions.multivariate_normal import _batch_mahalanobis

def covariateShiftCP(scores: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """TODO: is this actually local cp???? or just the easier version where the weights are not data-dependent (i.e. the kernel is pre-fixed) !!!!!!
    Local Conformal Prediction. Inputs: scores/residuals (N), weights (M, N), alpha in (0, 1). Returns (M) qhats.
    TODO: not dealing with ties in residuals properly.
    """
    assert scores.dim() == 1 and weights.dim() == 2 and scores.shape[0] == weights.shape[1]
    assert torch.all(weights >= 0) and 0 < alpha < 1

    _scores = scores.unsqueeze(0).expand(weights.shape[0], -1)  # Expand to (M, N)
    _scores = torch.cat((_scores, torch.full((weights.shape[0], 1), torch.inf)), dim=1)  # Add infinity to residuals (M, N+1) for test point
    _weights = torch.cat((weights, torch.ones(weights.shape[0], 1)), dim=1)  # Add 1.0 to weights (M, N+1) for test point

    _normed_weights = _weights / _weights.sum(dim=1, keepdim=True)  # Normalize weights

    sorted_scores, sorted_indices = torch.sort(_scores, dim=1)  # Sort residuals

    sorted_weights = torch.gather(input=_normed_weights, dim=1, index=sorted_indices)  # Sort weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=1)  # Cumulative sum of weights     (M, N+1)

    quantile_indices = torch.searchsorted(
        cumulative_weights, (1 - alpha) * torch.ones(weights.shape[0], 1), right=False
    )  # Find the fist index where the cumulative sum exceeds 1-alpha

    return torch.gather(input=sorted_scores, dim=1, index=quantile_indices).squeeze(1)

def splitCP(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Split/Inductive Conformal Prediction."""
    return covariateShiftCP(scores=scores, weights=torch.ones((1, len(scores))), alpha=alpha)

def empirical_coverage_l2(mean_pred: torch.Tensor, y_true: torch.Tensor, q_hat: torch.Tensor):
    distances = torch.norm(mean_pred - y_true, dim=-1)
    mask = distances <= q_hat
    return mask, mask.float().mean().item()

def empirical_coverage_mahalanobis(mean_pred: torch.Tensor, cov_pred:torch.Tensor, y_true: torch.Tensor, q_hat: torch.Tensor, alpha: float):
    # difference=mean_pred - y_true
    # failure_rate=alpha
    # md_squared = mahalanobis_distance_squared(
    #     difference=difference,
    #     covariance=cov_pred,
    # )
    
    # D = difference.shape[-1]
    # mask = md_squared.sqrt() <= critical_mahalanobis_distance(failure_rate=failure_rate, D=D)
    
    # mask = md_squared.sqrt() <= q_hat
    # empirical_coverage = mask.float().mean().item()

    difference = mean_pred - y_true
    mask, empirical_coverage = points_in_confidence_hyperellipsoid(
        difference=difference,
        covariance=cov_pred,
        failure_rate=alpha,
    )

    return mask, empirical_coverage

def mahalanobis_distance_squared(
    difference: torch.Tensor,
    *,  # Arguments after this require keyword
    cholesky_factor: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute squared Mahalanobis distance: (x - μ)^T Σ⁻¹ (x - μ)

    Args:
        difference (Tensor): shape (..., D)
        cholesky_factor (Tensor, optional): shape (..., D, D)
        covariance (Tensor, optional): shape (..., D, D)

    Returns:
        Tensor: Mahalanobis distance squared, shape (...,)
    """
    if (cholesky_factor is None) == (covariance is None):
        raise ValueError("Provide exactly one of `cholesky_factor` or `covariance`, not both or neither.")

    L = cholesky_factor if cholesky_factor is not None else torch.linalg.cholesky(covariance)

    assert L.shape[-2] == L.shape[-1], "Cholesky factor must be square"
    assert L.shape[-2] == difference.shape[-1], "Cholesky factor and difference must have same last dimension"

    # Normalize input shapes
    if difference.ndim == 1:
        difference = difference.unsqueeze(0)  # (1, D)
    if L.ndim == 2:
        L = L.unsqueeze(0)  # (1, D, D)

    # If only one of the inputs is batched, expand the other to match
    if difference.shape[0] == 1 and L.shape[0] > 1:
        difference = difference.expand(L.shape[0], -1)
    elif L.shape[0] == 1 and difference.shape[0] > 1:
        L = L.expand(difference.shape[0], -1, -1)

    return _batch_mahalanobis(L, difference) 


# def critical_mahalanobis_distance(failure_rate: float, D: int) -> float:
#     """
#     Compute the critical Mahalanobis distance for a given failure rate and dimensionality.
#     """
#     return math.sqrt(chi2.ppf(1 - failure_rate, D))


def critical_mahalanobis_distance(failure_rate: float, D: int) -> float:
    """
    Nominal Gaussian critical Mahalanobis distance:
        sqrt(chi2.ppf(1 - failure_rate, D))
    """
    if not (0 < failure_rate < 1):
        raise ValueError("`failure_rate` must be in (0, 1).")
    if D <= 0:
        raise ValueError("`D` must be positive.")
    return math.sqrt(chi2.ppf(1 - failure_rate, D))


def scaling_factor_from_qhat(
    q_hat: torch.Tensor,
    *,
    failure_rate: float,
    D: int,
) -> torch.Tensor:
    """
    Convert conformal Mahalanobis threshold q_hat into covariance inflation factor.

    If nominal Gaussian region uses:
        sqrt(chi2_{D, 1-failure_rate})

    and conformal wants radius q_hat, then:
        gamma = q_hat^2 / chi2_{D, 1-failure_rate}
    """
    critical = critical_mahalanobis_distance(failure_rate=failure_rate, D=D)
    return q_hat.pow(2) / (critical ** 2)


def confidence_hyperellipsoid_volume(cholesky_factor: torch.Tensor, failure_rate: float) -> torch.Tensor:
    """
    Computes the volume of the (1 - failure_rate) confidence hyperellipsoid
    defined by the Cholesky factor of the covariance matrix.

    Based on:
    - Theorem 3.10 (p. 100) of David Olive, "Robust Statistics"
      http://parker.ad.siu.edu/Olive/runrob.pdf
    - Theorem 3.9 (p. 99) for the chi-squared quantile (h²)

    Args:
        cholesky_factor (Tensor): Cholesky factor of the covariance matrix,
            shape (..., D, D) or (D, D)
        failure_rate (float): Desired failure rate (e.g. 0.05 for 95% confidence)

    Returns:
        Tensor: (...,) volumes of the confidence hyperellipsoids
    """
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    L = cholesky_factor
    if L.ndim == 2:
        L = L.unsqueeze(0)  # Make it batched: (1, D, D)

    D = L.shape[-1]

    # det(L) = sqrt(det(covariance))
    det_term = torch.linalg.det(L)  # shape (...,)

    h2 = critical_mahalanobis_distance(failure_rate=failure_rate, D=D) ** 2

    numerator = 2 * (torch.pi ** (D / 2)) * (h2 ** (D / 2)) * det_term
    volume = numerator / (D * gamma(D / 2))
    return volume if volume.ndim > 0 else volume.unsqueeze(0)

def ball_volume(radius: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the volume of a ball of radius r in dim dimensions.
    """
    return torch.pi ** (dim / 2) * radius ** dim / gamma(dim / 2 + 1)

def points_in_confidence_hyperellipsoid_from_mahalanobis(
    maha_squared: torch.Tensor,
    *,
    failure_rate: float,
    D: int,
) -> Tuple[torch.Tensor, float]:
    """
    Check if a Mahalanobis-squared value is within the (1 - failure_rate) confidence region.

    Args:
        maha_squared (Tensor): shape (...,), precomputed Mahalanobis distance squared
        failure_rate (float): e.g. 0.05 for 95% confidence
        D (int, optional): Dimensionality of the data. Required if chi-squared threshold can't be inferred.

    Returns:
        Tensor: Boolean tensor of shape (...) indicating whether each point is inside the ellipsoid
    """
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    mask = maha_squared.sqrt() <= critical_mahalanobis_distance(failure_rate=failure_rate, D=D)
    fraction = mask.float().mean().item()
    return mask, fraction


def points_in_confidence_hyperellipsoid(
    difference: torch.Tensor,
    *,
    failure_rate: float,
    cholesky_factor: Optional[torch.Tensor] = None,
    covariance: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Check if difference vectors are inside the (1 - failure_rate) confidence hyperellipsoid.

    Args:
        difference (Tensor): shape (..., D), difference from the mean (x - μ)
        failure_rate (float): e.g. 0.05 for 95% confidence
        cholesky_factor (Tensor, optional): shape (..., D, D), lower-triangular matrix L such that Σ = L @ L.T
        covariance (Tensor, optional): shape (..., D, D)

    Returns:
        Tensor: Boolean tensor of shape (...) indicating whether each point is inside the ellipsoid
    """
    if (cholesky_factor is None) == (covariance is None):
        raise ValueError("Provide exactly one of `cholesky_factor` or `covariance`, not both or neither.")
    if not 0 < failure_rate < 1:
        raise ValueError("failure_rate must be between 0 and 1")

    D = difference.shape[-1]

    # Compute Mahalanobis distance squared
    md_squared = mahalanobis_distance_squared(
        difference,
        cholesky_factor=cholesky_factor,
        covariance=covariance,
    )

    return points_in_confidence_hyperellipsoid_from_mahalanobis(
        maha_squared=md_squared,
        failure_rate=failure_rate,
        D=D,
    )