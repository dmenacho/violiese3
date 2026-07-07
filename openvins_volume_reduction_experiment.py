"""Ablations for reducing adaptive OpenVINS conformal task-space volume.

This runner reuses the trajectory folds, ACI/AgACI implementations, SE(3)
errors, and task-volume metric from ``adaptive_openvins_experiment.py``.  It
tests only estimator-derived OpenVINS covariance; empirical covariance is used
only as a fit-set shrinkage target in explicitly named blend ablations.

The experiment separates three possible improvements:

1. covariance shape correction learned on complete fit trajectories;
2. a heteroscedastic score-scale model learned on fit trajectories; and
3. recent labelled-score feedback in the online quantile.

Calibration trajectories remain disjoint from fit and test trajectories.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_openvins_experiment import (
    DEFAULT_GAMMAS,
    Fold,
    OnlineResult,
    _adaptive_radius,
    _pinball_loss,
    build_cross_environment_folds,
    build_pooled_same_robot_folds,
    load_sequences,
    longest_false_run,
    parse_point,
    rolling_coverage_error,
    run_aci,
    run_agaci,
    task_log_volumes,
)
from utils.dataset_io import PoseSequence
from utils.conformal_prediction.se3 import (
    fit_empirical_covariance,
    mahalanobis_scores,
    regularize_covariance,
)


DEFAULT_ALPHA = 0.10
DEFAULT_ACI_GAMMA = 0.01
RECENT_WINDOWS = (100, 300)


@dataclass(frozen=True)
class BalancedData:
    errors: np.ndarray
    covariance: np.ndarray
    elapsed_seconds: np.ndarray


@dataclass(frozen=True)
class CovarianceStrategy:
    name: str
    bias: np.ndarray
    fit_covariance: np.ndarray
    calibration_covariance: np.ndarray
    test_covariance: np.ndarray


@dataclass(frozen=True)
class LogScaleModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    minimum_log_scale: float
    maximum_log_scale: float
    include_time: bool


def balanced_data(
    sequences: dict[str, PoseSequence], names: tuple[str, ...]
) -> BalancedData:
    """Pool complete trajectories with equal sample contribution."""
    count = min(len(sequences[name].errors) for name in names)
    errors: list[np.ndarray] = []
    covariance: list[np.ndarray] = []
    elapsed: list[np.ndarray] = []
    for name in names:
        sequence = sequences[name]
        indices = np.linspace(0, len(sequence.errors) - 1, count, dtype=int)
        errors.append(sequence.errors[indices])
        covariance.append(sequence.covariance[indices])
        timestamps = sequence.timestamps[indices]
        elapsed.append(timestamps - timestamps[0])
    return BalancedData(
        errors=np.concatenate(errors),
        covariance=regularize_covariance(np.concatenate(covariance)),
        elapsed_seconds=np.concatenate(elapsed),
    )


def sequence_data(sequence: PoseSequence) -> BalancedData:
    return BalancedData(
        errors=sequence.errors,
        covariance=regularize_covariance(sequence.covariance),
        elapsed_seconds=sequence.timestamps - sequence.timestamps[0],
    )


def _sample_covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    diagonal = np.diag(np.diag(covariance))
    shrinkage = min(0.25, values.shape[1] / max(len(values), 1))
    return regularize_covariance(
        (1.0 - shrinkage) * covariance + shrinkage * diagonal
    )


def _diagonal_scale_covariance(
    covariance: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    standard_deviation = np.sqrt(
        np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-15)
    )
    corrected = np.einsum(
        "ni,ij,nj->nij", standard_deviation, correction, standard_deviation
    )
    return regularize_covariance(corrected)


def _diagonal_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.zeros_like(covariance)
    coordinates = np.arange(covariance.shape[-1])
    diagonal[:, coordinates, coordinates] = np.diagonal(
        covariance, axis1=1, axis2=2
    )
    return regularize_covariance(diagonal)


def _correlation_shrinkage(
    covariance: np.ndarray, diagonal_weight: float
) -> np.ndarray:
    return regularize_covariance(
        (1.0 - diagonal_weight) * covariance
        + diagonal_weight * _diagonal_covariance(covariance)
    )


def _whitened_covariance(
    covariance: np.ndarray, correction: np.ndarray
) -> np.ndarray:
    factors = np.linalg.cholesky(regularize_covariance(covariance))
    corrected = np.einsum("nij,jk,nlk->nil", factors, correction, factors)
    return regularize_covariance(corrected)


def _normalized_blend(
    covariance: np.ndarray,
    empirical_covariance: np.ndarray,
    raw_fit_trace: float,
    weight: float,
) -> np.ndarray:
    empirical_trace = float(np.trace(empirical_covariance))
    scaled_openvins = covariance * (empirical_trace / max(raw_fit_trace, 1e-15))
    blended = (
        weight * scaled_openvins
        + (1.0 - weight) * empirical_covariance[None, :, :]
    )
    return regularize_covariance(blended)


def build_covariance_strategies(
    fit: BalancedData,
    calibration: BalancedData,
    test: BalancedData,
) -> list[CovarianceStrategy]:
    fit_bias, empirical_covariance = fit_empirical_covariance(fit.errors)
    raw_fit_trace = float(np.median(np.trace(fit.covariance, axis1=1, axis2=2)))

    standard_deviation = np.sqrt(
        np.maximum(np.diagonal(fit.covariance, axis1=1, axis2=2), 1e-15)
    )
    standardized = (fit.errors - fit_bias) / standard_deviation
    standardized_correction = _sample_covariance(standardized)

    factors = np.linalg.cholesky(fit.covariance)
    whitened = np.linalg.solve(factors, (fit.errors - fit_bias)[..., None])[..., 0]
    whitened_correction = _sample_covariance(whitened)

    diagonal_fit = _diagonal_covariance(fit.covariance)
    diagonal_calibration = _diagonal_covariance(calibration.covariance)
    diagonal_test = _diagonal_covariance(test.covariance)

    strategies = [
        CovarianceStrategy(
            "raw_openvins", np.zeros(6), fit.covariance,
            calibration.covariance, test.covariance,
        ),
        CovarianceStrategy(
            "raw_openvins_fit_bias", fit_bias, fit.covariance,
            calibration.covariance, test.covariance,
        ),
        CovarianceStrategy(
            "diagonal_openvins", np.zeros(6), diagonal_fit,
            diagonal_calibration, diagonal_test,
        ),
        CovarianceStrategy(
            "diagonal_openvins_fit_bias", fit_bias,
            diagonal_fit, diagonal_calibration, diagonal_test,
        ),
        CovarianceStrategy(
            "standardized_correlation", fit_bias,
            _diagonal_scale_covariance(fit.covariance, standardized_correction),
            _diagonal_scale_covariance(
                calibration.covariance, standardized_correction
            ),
            _diagonal_scale_covariance(test.covariance, standardized_correction),
        ),
        CovarianceStrategy(
            "whitened_correlation", fit_bias,
            _whitened_covariance(fit.covariance, whitened_correction),
            _whitened_covariance(calibration.covariance, whitened_correction),
            _whitened_covariance(test.covariance, whitened_correction),
        ),
    ]
    for diagonal_weight in (0.25, 0.50, 0.75):
        for name_suffix, bias in (("", np.zeros(6)), ("_fit_bias", fit_bias)):
            strategies.append(
                CovarianceStrategy(
                    f"correlation_shrink_{diagonal_weight:.2f}{name_suffix}",
                    bias,
                    _correlation_shrinkage(fit.covariance, diagonal_weight),
                    _correlation_shrinkage(
                        calibration.covariance, diagonal_weight
                    ),
                    _correlation_shrinkage(test.covariance, diagonal_weight),
                )
            )
    for weight in (0.25, 0.50, 0.75):
        strategies.append(
            CovarianceStrategy(
                f"normalized_empirical_blend_{weight:.2f}",
                fit_bias,
                _normalized_blend(
                    fit.covariance, empirical_covariance, raw_fit_trace, weight
                ),
                _normalized_blend(
                    calibration.covariance,
                    empirical_covariance,
                    raw_fit_trace,
                    weight,
                ),
                _normalized_blend(
                    test.covariance, empirical_covariance, raw_fit_trace, weight
                ),
            )
        )
    return strategies


def covariance_features(
    covariance: np.ndarray,
    elapsed_seconds: np.ndarray,
    include_time: bool,
) -> np.ndarray:
    covariance = regularize_covariance(covariance)
    diagonal = np.log(
        np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-15)
    )
    translation_eigenvalues = np.linalg.eigvalsh(covariance[:, :3, :3])
    rotation_eigenvalues = np.linalg.eigvalsh(covariance[:, 3:, 3:])
    features = [
        diagonal,
        np.log(np.maximum(translation_eigenvalues, 1e-15)),
        np.log(np.maximum(rotation_eigenvalues, 1e-15)),
    ]
    if include_time:
        features.append(np.log1p(np.maximum(elapsed_seconds, 0.0))[:, None])
    return np.concatenate(features, axis=1)


def fit_log_scale_model(
    covariance: np.ndarray,
    elapsed_seconds: np.ndarray,
    scores: np.ndarray,
    include_time: bool,
    ridge: float = 1.0,
) -> LogScaleModel:
    features = covariance_features(covariance, elapsed_seconds, include_time)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(features)), standardized])
    target = np.log(np.maximum(scores, 1e-12))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    fitted = design @ coefficients
    return LogScaleModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        minimum_log_scale=float(np.quantile(fitted, 0.01)),
        maximum_log_scale=float(np.quantile(fitted, 0.99)),
        include_time=include_time,
    )


def predict_log_scale(
    model: LogScaleModel,
    covariance: np.ndarray,
    elapsed_seconds: np.ndarray,
) -> np.ndarray:
    features = covariance_features(
        covariance, elapsed_seconds, model.include_time
    )
    standardized = (features - model.feature_mean) / model.feature_scale
    design = np.column_stack([np.ones(len(features)), standardized])
    prediction = design @ model.coefficients
    return np.exp(
        np.clip(
            prediction, model.minimum_log_scale, model.maximum_log_scale
        )
    )


def _recent_pool(
    calibration_scores: np.ndarray,
    history: list[float],
    window: int,
) -> np.ndarray:
    minimum_history = min(50, window)
    if len(history) < minimum_history:
        return calibration_scores
    return np.asarray(history[-window:], dtype=np.float64)


def run_recent_aci(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    gamma: float,
    window: int,
) -> OnlineResult:
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    effective_alpha = alpha
    history: list[float] = []
    radii = np.empty(len(test_scores))
    covered = np.empty(len(test_scores), dtype=bool)
    effective_alphas = np.empty(len(test_scores))
    for index, score in enumerate(test_scores):
        pool = np.sort(_recent_pool(calibration_scores, history, window))
        radius = _adaptive_radius(pool, effective_alpha)
        radii[index] = radius
        covered[index] = score <= radius
        effective_alphas[index] = effective_alpha
        effective_alpha = float(
            np.clip(
                effective_alpha
                + gamma * (alpha - float(not covered[index])),
                0.0,
                1.0,
            )
        )
        history.append(float(score))
    return OnlineResult(radii, covered, effective_alphas)


def run_recent_agaci(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    gammas: tuple[float, ...],
    window: int,
) -> OnlineResult:
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    gammas_array = np.asarray(gammas, dtype=np.float64)
    expert_alphas = np.full(len(gammas_array), alpha, dtype=np.float64)
    log_weights = np.zeros(len(gammas_array), dtype=np.float64)
    quantile_level = 1.0 - alpha
    history: list[float] = []
    radii = np.empty(len(test_scores))
    covered = np.empty(len(test_scores), dtype=bool)
    effective_alphas = np.empty(len(test_scores))
    for index, score in enumerate(test_scores):
        pool = np.sort(_recent_pool(calibration_scores, history, window))
        expert_radii = np.asarray(
            [_adaptive_radius(pool, value) for value in expert_alphas]
        )
        shifted = log_weights - np.max(log_weights)
        weights = np.exp(shifted)
        weights /= weights.sum()
        radius = float(weights @ expert_radii)
        radii[index] = radius
        covered[index] = score <= radius
        effective_alphas[index] = float(weights @ expert_alphas)

        expert_covered = score <= expert_radii
        expert_alphas = np.clip(
            expert_alphas
            + gammas_array * (alpha - (~expert_covered).astype(float)),
            0.0,
            1.0,
        )
        loss_scale = max(float(np.median(np.abs(pool))), 1e-9)
        losses = _pinball_loss(score, expert_radii, quantile_level) / loss_scale
        log_weights -= 0.5 * np.clip(losses, 0.0, 50.0)
        history.append(float(score))
    return OnlineResult(radii, covered, effective_alphas)


def _online_variants(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    aci_gamma: float,
    agaci_gammas: tuple[float, ...],
    use_recent_feedback: bool,
) -> dict[tuple[str, str], OnlineResult]:
    variants = {
        ("ACI", "calibration_only"): run_aci(
            calibration_scores, test_scores, alpha, aci_gamma
        ),
        ("AgACI-EWA", "calibration_only"): run_agaci(
            calibration_scores, test_scores, alpha, agaci_gammas
        ),
    }
    if use_recent_feedback:
        for window in RECENT_WINDOWS:
            variants[("ACI", f"recent_{window}")] = run_recent_aci(
                calibration_scores, test_scores, alpha, aci_gamma, window
            )
            variants[("AgACI-EWA", f"recent_{window}")] = run_recent_agaci(
                calibration_scores, test_scores, alpha, agaci_gammas, window
            )
    return variants


def evaluate(
    sequences: dict[str, PoseSequence],
    folds: list[Fold],
    alpha: float,
    aci_gamma: float,
    agaci_gammas: tuple[float, ...],
    body_point: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds, start=1):
        print(f"[{fold_index:02d}/{len(folds):02d}] {fold.fold_id}", flush=True)
        fit = balanced_data(sequences, fold.fit_names)
        calibration = balanced_data(sequences, fold.calibration_names)
        test = sequence_data(sequences[fold.test_name])
        for strategy in build_covariance_strategies(fit, calibration, test):
            fit_scores = mahalanobis_scores(
                fit.errors, strategy.fit_covariance, strategy.bias
            )
            calibration_scores = mahalanobis_scores(
                calibration.errors,
                strategy.calibration_covariance,
                strategy.bias,
            )
            test_scores = mahalanobis_scores(
                test.errors, strategy.test_covariance, strategy.bias
            )
            scale_variants: list[tuple[str, np.ndarray, np.ndarray]] = [
                (
                    "constant",
                    np.ones(len(calibration_scores)),
                    np.ones(len(test_scores)),
                )
            ]
            for name, include_time in (
                ("loglinear_covariance", False),
                ("loglinear_covariance_time", True),
            ):
                model = fit_log_scale_model(
                    strategy.fit_covariance,
                    fit.elapsed_seconds,
                    fit_scores,
                    include_time,
                )
                scale_variants.append(
                    (
                        name,
                        predict_log_scale(
                            model,
                            strategy.calibration_covariance,
                            calibration.elapsed_seconds,
                        ),
                        predict_log_scale(
                            model,
                            strategy.test_covariance,
                            test.elapsed_seconds,
                        ),
                    )
                )

            for scale_name, calibration_scale, test_scale in scale_variants:
                normalized_calibration = calibration_scores / calibration_scale
                normalized_test = test_scores / test_scale
                use_recent = strategy.name in {
                    "raw_openvins",
                    "standardized_correlation",
                    "normalized_empirical_blend_0.50",
                }
                for (method, feedback), result in _online_variants(
                    normalized_calibration,
                    normalized_test,
                    alpha,
                    aci_gamma,
                    agaci_gammas,
                    use_recent,
                ).items():
                    radii = result.radii * test_scale
                    task_logs = task_log_volumes(
                        strategy.test_covariance, radii, body_point
                    )
                    coverage = float(result.covered.mean())
                    rows.append(
                        {
                            "protocol": fold.protocol,
                            "fold_id": fold.fold_id,
                            "test_environment": fold.test_environment,
                            "test_trajectory": fold.test_name,
                            "covariance_strategy": strategy.name,
                            "score_scale": scale_name,
                            "method": method,
                            "feedback": feedback,
                            "alpha": alpha,
                            "target_coverage": 1.0 - alpha,
                            "n_test": len(test.errors),
                            "coverage": coverage,
                            "absolute_coverage_gap": abs(coverage - (1.0 - alpha)),
                            "median_log_task_volume_m3": float(np.median(task_logs)),
                            "p90_log_task_volume_m3": float(np.quantile(task_logs, 0.90)),
                            "mean_log_task_volume_m3": float(np.mean(task_logs)),
                            "mean_radius": float(np.mean(radii)),
                            "rolling_coverage_mae_100": rolling_coverage_error(
                                result.covered, 1.0 - alpha
                            ),
                            "longest_miss_streak": longest_false_run(result.covered),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "protocol",
        "covariance_strategy",
        "score_scale",
        "method",
        "feedback",
        "alpha",
        "target_coverage",
    ]
    summary = (
        frame.groupby(groups, as_index=False)
        .agg(
            trajectory_mean_coverage=("coverage", "mean"),
            trajectory_std_coverage=("coverage", "std"),
            minimum_trajectory_coverage=("coverage", "min"),
            geometric_mean_median_volume_m3=(
                "median_log_task_volume_m3",
                lambda values: float(np.exp(np.mean(values))),
            ),
            geometric_mean_p90_volume_m3=(
                "p90_log_task_volume_m3",
                lambda values: float(np.exp(np.mean(values))),
            ),
            rolling_coverage_mae_100=("rolling_coverage_mae_100", "mean"),
            longest_miss_streak=("longest_miss_streak", "mean"),
            folds=("test_trajectory", "count"),
        )
    )
    pooled_rows = []
    for key, group in frame.groupby(groups):
        pooled_rows.append(
            (*key, float(np.average(group.coverage, weights=group.n_test)))
        )
    pooled = pd.DataFrame(
        pooled_rows, columns=[*groups, "sample_pooled_coverage"]
    )
    summary = summary.merge(pooled, on=groups, how="left")
    summary["coverage_shortfall"] = np.maximum(
        summary.target_coverage - summary.sample_pooled_coverage, 0.0
    )
    return summary.sort_values(
        [
            "protocol",
            "coverage_shortfall",
            "geometric_mean_median_volume_m3",
        ]
    )


def add_cross_protocol_screen(summary: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "covariance_strategy", "score_scale", "method", "feedback", "alpha"
    ]
    rows = []
    for key, group in summary.groupby(keys):
        if set(group.protocol) != {"pooled_same_robot", "cross_environment"}:
            continue
        by_protocol = group.set_index("protocol")
        pooled = by_protocol.loc["pooled_same_robot"]
        cross = by_protocol.loc["cross_environment"]
        rows.append(
            {
                **dict(zip(keys, key)),
                "pooled_coverage": pooled.sample_pooled_coverage,
                "cross_environment_coverage": cross.sample_pooled_coverage,
                "worst_coverage": min(
                    pooled.sample_pooled_coverage,
                    cross.sample_pooled_coverage,
                ),
                "pooled_minimum_trajectory_coverage": (
                    pooled.minimum_trajectory_coverage
                ),
                "cross_minimum_trajectory_coverage": (
                    cross.minimum_trajectory_coverage
                ),
                "worst_trajectory_coverage": min(
                    pooled.minimum_trajectory_coverage,
                    cross.minimum_trajectory_coverage,
                ),
                "pooled_volume_m3": pooled.geometric_mean_median_volume_m3,
                "cross_environment_volume_m3": cross.geometric_mean_median_volume_m3,
                "worst_volume_m3": max(
                    pooled.geometric_mean_median_volume_m3,
                    cross.geometric_mean_median_volume_m3,
                ),
                "pooled_p90_volume_m3": pooled.geometric_mean_p90_volume_m3,
                "cross_environment_p90_volume_m3": cross.geometric_mean_p90_volume_m3,
                "worst_rolling_mae": max(
                    pooled.rolling_coverage_mae_100,
                    cross.rolling_coverage_mae_100,
                ),
            }
        )
    screen = pd.DataFrame(rows)
    screen["coverage_valid_1pct"] = screen.worst_coverage >= 0.89
    return screen.sort_values(
        ["coverage_valid_1pct", "worst_volume_m3"],
        ascending=[False, True],
    )


def plot_screen(screen: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    for method, group in screen.groupby("method"):
        valid = group.coverage_valid_1pct
        axis.scatter(
            group.loc[valid, "worst_coverage"],
            group.loc[valid, "worst_volume_m3"],
            s=42,
            alpha=0.75,
            label=f"{method}, coverage-valid",
        )
        axis.scatter(
            group.loc[~valid, "worst_coverage"],
            group.loc[~valid, "worst_volume_m3"],
            marker="x",
            s=34,
            alpha=0.5,
            label=f"{method}, undercoverage",
        )
    axis.axvline(0.90, color="black", linestyle=":", linewidth=1)
    axis.axvline(0.89, color="tab:red", linestyle="--", linewidth=1)
    strict = screen[
        (screen.pooled_coverage >= 0.90)
        & (screen.cross_environment_coverage >= 0.90)
    ].sort_values("worst_volume_m3")
    recommended = strict.iloc[0]
    baseline = screen[
        (screen.covariance_strategy == "raw_openvins")
        & (screen.score_scale == "constant")
        & (screen.method == "AgACI-EWA")
        & (screen.feedback == "calibration_only")
    ].iloc[0]
    for row, label, offset in (
        (baseline, "Raw OpenVINS AgACI", (8, 8)),
        (recommended, "Recommended", (8, -18)),
    ):
        axis.scatter(
            row.worst_coverage,
            row.worst_volume_m3,
            marker="*",
            s=180,
            facecolor="gold",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        axis.annotate(
            label,
            (row.worst_coverage, row.worst_volume_m3),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Worst pooled/cross-environment sample coverage")
    axis.set_ylabel("Worst pooled/cross-environment geometric mean volume [m³]")
    axis.set_title("OpenVINS 90% coverage-volume ablation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def write_report(screen: pd.DataFrame, output_path: Path) -> None:
    valid = screen[screen.coverage_valid_1pct].copy()
    strict = screen[
        (screen.pooled_coverage >= 0.90)
        & (screen.cross_environment_coverage >= 0.90)
    ].sort_values("worst_volume_m3")
    recommended = strict.iloc[0]
    baseline = screen[
        (screen.covariance_strategy == "raw_openvins")
        & (screen.score_scale == "constant")
        & (screen.method == "AgACI-EWA")
        & (screen.feedback == "calibration_only")
    ].iloc[0]
    pooled_reduction = 1.0 - (
        recommended.pooled_volume_m3 / baseline.pooled_volume_m3
    )
    cross_reduction = 1.0 - (
        recommended.cross_environment_volume_m3
        / baseline.cross_environment_volume_m3
    )
    columns = [
        "covariance_strategy",
        "score_scale",
        "method",
        "feedback",
        "pooled_coverage",
        "cross_environment_coverage",
        "worst_trajectory_coverage",
        "pooled_volume_m3",
        "cross_environment_volume_m3",
        "worst_rolling_mae",
    ]
    lines = [
        "# OpenVINS quantile-volume study",
        "",
        "Target coverage is 90%. A configuration is screened as coverage-valid only if",
        "both pooled and cross-environment sample coverage are at least 89%.",
        "",
        "## Recommended checkpoint",
        "",
        f"`{recommended.covariance_strategy}` + `{recommended.score_scale}` + "
        f"`{recommended.method}` with `{recommended.feedback}` feedback is the smallest",
        "configuration that reaches at least 90% sample-pooled coverage in both protocols.",
        "",
        f"- Pooled: coverage {recommended.pooled_coverage:.4f}, median volume "
        f"{recommended.pooled_volume_m3:.4f} m^3.",
        f"- Cross-environment: coverage {recommended.cross_environment_coverage:.4f}, "
        f"median volume {recommended.cross_environment_volume_m3:.4f} m^3.",
        f"- Relative to raw OpenVINS AgACI, median volume falls by "
        f"{pooled_reduction:.1%} pooled and {cross_reduction:.1%} cross-environment.",
        f"- Worst individual trajectory coverage is "
        f"{recommended.worst_trajectory_coverage:.4f}; aggregate validity does not imply "
        "uniform trajectory-level validity.",
        "",
        "The method retains each OpenVINS marginal variance, shrinks its reported",
        "off-diagonal correlations 75% toward zero, and predicts a pose-specific score",
        "scale from covariance eigenvalue/diagonal features using fit trajectories.",
        "AgACI then calibrates the normalized score on disjoint trajectories.",
        "",
        "## Coverage-valid configurations",
        "",
        valid[columns].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "Scalar covariance inflation is intentionally absent: conformal calibration",
        "cancels any global scalar and cannot change the final calibrated set.",
        "",
        "Recent-score variants require delayed ground-truth/error feedback and are not",
        "deployable on an unlabeled robot without an external localization signal. In",
        "this experiment, 100/300-sample rolling replacements undercovered and are rejected.",
        "Empirical covariance blends reached very small volumes but failed cross-environment",
        "coverage, so they are also rejected.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(","))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "OPEN_VINS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/openvins_volume_reduction"),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--aci-gamma", type=float, default=DEFAULT_ACI_GAMMA)
    parser.add_argument(
        "--agaci-gammas", type=parse_float_tuple, default=DEFAULT_GAMMAS
    )
    parser.add_argument(
        "--body-point", type=parse_point, default=np.array([3.0, 0.0, 0.0])
    )
    parser.add_argument(
        "--covariance-frame",
        choices=(
            "body",
            "world",
            "position_world_rotation_body",
            "position_body_rotation_world",
        ),
        default="position_world_rotation_body",
    )
    parser.add_argument("--max-time-difference", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("OpenVINS adaptive quantile-volume reduction study")
    print("- Only OpenVINS estimator covariance is evaluated as the online input.")
    print("- Shape/scale models use fit trajectories; CP uses separate calibration trajectories.")
    print("- Test state resets at each complete trajectory.")
    print("- Scalar covariance inflation is omitted because CP cancels it exactly.")
    sequences = load_sequences(
        args.dataset_dir, args.covariance_frame, args.max_time_difference
    )
    folds = build_pooled_same_robot_folds(sequences) + build_cross_environment_folds(
        sequences
    )
    frame = evaluate(
        sequences,
        folds,
        args.alpha,
        args.aci_gamma,
        args.agaci_gammas,
        args.body_point,
    )
    summary = summarize(frame)
    screen = add_cross_protocol_screen(summary)
    frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    screen.to_csv(args.output_dir / "cross_protocol_screen.csv", index=False)
    plot_screen(screen, args.output_dir / "coverage_volume_screen.png")
    write_report(screen, args.output_dir / "report.md")
    print("\nTop coverage-valid configurations")
    print(
        screen[screen.coverage_valid_1pct]
        .head(20)
        .to_string(index=False)
    )
    print(f"\nSaved results under {args.output_dir}")


if __name__ == "__main__":
    main()
