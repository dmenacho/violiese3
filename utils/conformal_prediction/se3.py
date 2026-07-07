"""Shared SE(3) conformal calibration and task-space projection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ConformalSE3Model:
    """Calibrated tangent-space uncertainty model.

    ``covariance`` is either one global empirical covariance with shape (6, 6)
    or per-pose VIO covariance with shape (N, 6, 6). The covariance and errors
    must use the same body-frame perturbation convention.
    """

    bias: np.ndarray
    covariance: np.ndarray
    quantile: float
    alpha: float
    covariance_source: str


def skew(vectors: np.ndarray) -> np.ndarray:
    matrices = np.zeros((len(vectors), 3, 3), dtype=np.float64)
    matrices[:, 0, 1] = -vectors[:, 2]
    matrices[:, 0, 2] = vectors[:, 1]
    matrices[:, 1, 0] = vectors[:, 2]
    matrices[:, 1, 2] = -vectors[:, 0]
    matrices[:, 2, 0] = -vectors[:, 1]
    matrices[:, 2, 1] = vectors[:, 0]
    return matrices


def se3_exp(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation and translation for Exp(xi), xi=[rho, phi]."""
    xi = np.atleast_2d(np.asarray(xi, dtype=np.float64))
    rho = xi[:, :3]
    phi = xi[:, 3:]
    rotation = Rotation.from_rotvec(phi).as_matrix()
    theta = np.linalg.norm(phi, axis=1)
    phi_hat = skew(phi)
    phi_hat_sq = phi_hat @ phi_hat
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    small = theta < 1e-7
    theta_sq = theta[small] ** 2
    a[small] = 0.5 - theta_sq / 24.0 + theta_sq**2 / 720.0
    b[small] = 1.0 / 6.0 - theta_sq / 120.0 + theta_sq**2 / 5040.0
    a[~small] = (1.0 - np.cos(theta[~small])) / theta[~small] ** 2
    b[~small] = (
        theta[~small] - np.sin(theta[~small])
    ) / theta[~small] ** 3
    identity = np.broadcast_to(np.eye(3), phi_hat.shape)
    v_matrix = identity + a[:, None, None] * phi_hat + b[:, None, None] * phi_hat_sq
    translation = np.einsum("nij,nj->ni", v_matrix, rho)
    return rotation, translation


def se3_body_error(
    estimated_pose: np.ndarray,
    ground_truth_pose: np.ndarray,
) -> np.ndarray:
    """Return xi = Log(T_est^{-1} T_gt), with xi=[rho, phi]."""
    estimated_pose = np.asarray(estimated_pose, dtype=np.float64)
    ground_truth_pose = np.asarray(ground_truth_pose, dtype=np.float64)
    estimated_rotation = Rotation.from_quat(estimated_pose[:, 3:])
    ground_truth_rotation = Rotation.from_quat(ground_truth_pose[:, 3:])
    relative_rotation = estimated_rotation.inv() * ground_truth_rotation
    phi = relative_rotation.as_rotvec()
    relative_translation = estimated_rotation.inv().apply(
        ground_truth_pose[:, :3] - estimated_pose[:, :3]
    )

    theta = np.linalg.norm(phi, axis=1)
    phi_hat = skew(phi)
    phi_hat_sq = phi_hat @ phi_hat
    coefficient = np.empty_like(theta)
    small = theta < 1e-7
    coefficient[small] = (
        1.0 / 12.0
        + theta[small] ** 2 / 720.0
        + theta[small] ** 4 / 30240.0
    )
    half_theta = 0.5 * theta[~small]
    coefficient[~small] = (
        1.0 - half_theta / np.tan(half_theta)
    ) / theta[~small] ** 2
    identity = np.broadcast_to(np.eye(3), phi_hat.shape)
    v_inverse = identity - 0.5 * phi_hat + coefficient[:, None, None] * phi_hat_sq
    rho = np.einsum("nij,nj->ni", v_inverse, relative_translation)
    return np.concatenate([rho, phi], axis=1)


def regularize_covariance(covariance: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = eigenvalues[..., 0]
    scale = np.maximum(
        np.trace(covariance, axis1=-2, axis2=-1) / covariance.shape[-1],
        epsilon,
    )
    shift = np.maximum(epsilon * scale - minimum, 0.0)
    return covariance + shift[..., None, None] * np.eye(covariance.shape[-1])


def fit_empirical_covariance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 6 or len(errors) < 8:
        raise ValueError("Empirical SE(3) covariance requires at least eight 6D errors")
    bias = errors.mean(axis=0)
    centered = errors - bias
    covariance = centered.T @ centered / len(centered)
    diagonal = np.diag(np.diag(covariance))
    shrinkage = min(0.10, 6.0 / len(errors))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    return bias, regularize_covariance(covariance)


def mahalanobis_scores(
    errors: np.ndarray,
    covariance: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    if bias is not None:
        errors = errors - np.asarray(bias, dtype=np.float64)
    covariance = regularize_covariance(covariance)
    if covariance.ndim == 2:
        solved = np.linalg.solve(covariance, errors.T).T
    elif covariance.ndim == 3:
        if len(covariance) != len(errors):
            raise ValueError("Per-pose covariance count must match the error count")
        solved = np.linalg.solve(covariance, errors[..., None])[..., 0]
    else:
        raise ValueError("Covariance must have shape (6,6) or (N,6,6)")
    return np.sqrt(np.maximum(np.einsum("ni,ni->n", errors, solved), 0.0))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if not 0.0 < alpha < 1.0 or len(scores) == 0:
        raise ValueError("Conformal calibration requires scores and alpha in (0,1)")
    rank = min(math.ceil((len(scores) + 1) * (1.0 - alpha)), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


def calibrate_empirical_model(
    fit_errors: np.ndarray,
    calibration_errors: np.ndarray,
    alpha: float,
) -> ConformalSE3Model:
    bias, covariance = fit_empirical_covariance(fit_errors)
    quantile = conformal_quantile(
        mahalanobis_scores(calibration_errors, covariance, bias), alpha
    )
    return ConformalSE3Model(
        bias=bias,
        covariance=covariance,
        quantile=quantile,
        alpha=alpha,
        covariance_source="empirical",
    )


def calibrate_vio_covariance_model(
    calibration_errors: np.ndarray,
    calibration_covariance: np.ndarray,
    alpha: float,
) -> ConformalSE3Model:
    covariance = regularize_covariance(calibration_covariance)
    quantile = conformal_quantile(
        mahalanobis_scores(calibration_errors, covariance), alpha
    )
    return ConformalSE3Model(
        bias=np.zeros(6),
        covariance=covariance,
        quantile=quantile,
        alpha=alpha,
        covariance_source="vio",
    )


def covariance_at(model: ConformalSE3Model, index: int | None = None) -> np.ndarray:
    if model.covariance.ndim == 2:
        return model.covariance
    if index is None:
        raise ValueError("A pose index is required for per-pose VIO covariance")
    return model.covariance[index]


def sample_tangent_ellipsoid(
    model: ConformalSE3Model,
    index: int | None = None,
    sample_count: int = 4000,
    seed: int = 0,
    interior: bool = True,
) -> np.ndarray:
    """Sample the calibrated uncertainty set in the SE(3) tangent space."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(sample_count, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    if interior:
        directions *= rng.random(sample_count)[:, None] ** (1.0 / 6.0)
    cholesky = np.linalg.cholesky(regularize_covariance(covariance_at(model, index)))
    return model.bias + model.quantile * (directions @ cholesky.T)


def project_pose_set_to_task_space(
    estimated_pose: np.ndarray,
    tangent_samples: np.ndarray,
    body_points: np.ndarray,
) -> np.ndarray:
    """Map SE(3) samples to world positions of points attached to the body.

    Returns shape (sample_count, point_count, 3). A nonzero lever arm exposes
    the curved translation-rotation geometry. Using only [0,0,0] projects the
    uncertain camera origin and usually looks ellipsoidal rather than banana-like.
    """
    estimated_pose = np.asarray(estimated_pose, dtype=np.float64).reshape(7)
    body_points = np.atleast_2d(np.asarray(body_points, dtype=np.float64))
    perturbation_rotation, perturbation_translation = se3_exp(tangent_samples)
    perturbed_points = (
        np.einsum("nij,pj->npi", perturbation_rotation, body_points)
        + perturbation_translation[:, None, :]
    )
    estimated_rotation = Rotation.from_quat(estimated_pose[3:]).as_matrix()
    return (
        np.einsum("ij,npj->npi", estimated_rotation, perturbed_points)
        + estimated_pose[:3]
    )
