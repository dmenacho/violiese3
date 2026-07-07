"""Trajectory-level ACI/AgACI experiments for aligned OpenVINS sequences.

The experimental unit is a complete trajectory.  The runner evaluates
same-environment, pooled same-robot, and leave-one-environment-out protocols.
Complete trajectories are assigned to fit, calibration, or test, and source
trajectories contribute equal sample counts so long flights cannot dominate.
Test poses are processed in timestamp order and adaptive state is reset
between trajectories.

Every updated trajectory exports both translational covariance Pt and
rotational covariance Pr. The OpenVINS branch therefore uses the complete
dynamic block-diagonal 6D covariance supplied by the estimator.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from utils.dataset_io import PoseSequence, load_aligned_openvins_pair
from utils.conformal_prediction.se3 import (
    ConformalSE3Model,
    fit_empirical_covariance,
    mahalanobis_scores,
    project_pose_set_to_task_space,
    regularize_covariance,
    sample_tangent_ellipsoid,
)


DEFAULT_ALPHAS = (0.01, 0.05, 0.10, 0.20, 0.40, 0.50)
DEFAULT_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)


@dataclass(frozen=True)
class Fold:
    protocol: str
    fold_id: str
    test_environment: str
    fit_names: tuple[str, ...]
    calibration_names: tuple[str, ...]
    test_name: str


@dataclass(frozen=True)
class CovarianceInput:
    name: str
    bias: np.ndarray
    calibration_covariance: np.ndarray
    test_covariance: np.ndarray


@dataclass(frozen=True)
class OnlineResult:
    radii: np.ndarray
    covered: np.ndarray
    effective_alphas: np.ndarray


def environment_from_name(name: str) -> str:
    stripped = name.removeprefix("OV_")
    return stripped.split("_", maxsplit=1)[0]


def load_sequences(
    dataset_dir: Path,
    covariance_frame: str,
    max_time_difference: float,
) -> dict[str, PoseSequence]:
    sequences: dict[str, PoseSequence] = {}
    for directory in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        estimate_files = sorted(directory.glob("OV_*.csv"))
        ground_truth_files = sorted(directory.glob("GT_*.csv"))
        if len(estimate_files) != 1 or len(ground_truth_files) != 1:
            continue
        sequence = load_aligned_openvins_pair(
            estimate_files[0],
            ground_truth_files[0],
            max_time_difference=max_time_difference,
            covariance_frame=covariance_frame,
        )
        sequences[directory.name] = PoseSequence(
            name=directory.name,
            timestamps=sequence.timestamps,
            estimated_pose=sequence.estimated_pose,
            ground_truth_pose=sequence.ground_truth_pose,
            errors=sequence.errors,
            covariance=sequence.covariance,
        )
    if not sequences:
        raise ValueError(f"No aligned OpenVINS sequences found under {dataset_dir}")
    return sequences


def build_same_environment_folds(sequences: dict[str, PoseSequence]) -> list[Fold]:
    by_environment: dict[str, list[str]] = {}
    for name in sequences:
        by_environment.setdefault(environment_from_name(name), []).append(name)

    folds: list[Fold] = []
    for environment, unsorted_names in sorted(by_environment.items()):
        names = sorted(unsorted_names)
        if len(names) < 3:
            continue
        for test_index, test_name in enumerate(names):
            calibration_name = names[(test_index + 1) % len(names)]
            fit_names = tuple(
                name for name in names if name not in (test_name, calibration_name)
            )
            folds.append(
                Fold(
                    "same_environment",
                    f"same_environment:{test_name}",
                    environment,
                    fit_names,
                    (calibration_name,),
                    test_name,
                )
            )
    if not folds:
        raise ValueError("At least three complete trajectories per environment are required")
    return folds


def _stratified_source_partition(
    names: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split complete trajectories into fit/calibration within each environment."""
    by_environment: dict[str, list[str]] = {}
    for name in names:
        by_environment.setdefault(environment_from_name(name), []).append(name)
    fit_names: list[str] = []
    calibration_names: list[str] = []
    for environment_names in by_environment.values():
        ordered = sorted(environment_names)
        fit_count = max(1, len(ordered) // 2)
        fit_names.extend(ordered[:fit_count])
        calibration_names.extend(ordered[fit_count:])
    if not fit_names or not calibration_names:
        raise ValueError("Each protocol requires complete fit and calibration trajectories")
    return tuple(sorted(fit_names)), tuple(sorted(calibration_names))


def build_pooled_same_robot_folds(
    sequences: dict[str, PoseSequence],
) -> list[Fold]:
    folds: list[Fold] = []
    all_names = sorted(sequences)
    for test_name in all_names:
        source_names = [name for name in all_names if name != test_name]
        fit_names, calibration_names = _stratified_source_partition(source_names)
        folds.append(
            Fold(
                "pooled_same_robot",
                f"pooled_same_robot:{test_name}",
                environment_from_name(test_name),
                fit_names,
                calibration_names,
                test_name,
            )
        )
    return folds


def build_cross_environment_folds(
    sequences: dict[str, PoseSequence],
) -> list[Fold]:
    folds: list[Fold] = []
    environments = sorted({environment_from_name(name) for name in sequences})
    for target_environment in environments:
        test_names = sorted(
            name
            for name in sequences
            if environment_from_name(name) == target_environment
        )
        source_names = sorted(
            name
            for name in sequences
            if environment_from_name(name) != target_environment
        )
        fit_names, calibration_names = _stratified_source_partition(source_names)
        for test_name in test_names:
            folds.append(
                Fold(
                    "cross_environment",
                    f"cross_environment:{target_environment}:{test_name}",
                    target_environment,
                    fit_names,
                    calibration_names,
                    test_name,
                )
            )
    return folds


def concatenate_errors(
    sequences: dict[str, PoseSequence], names: tuple[str, ...]
) -> np.ndarray:
    count = min(len(sequences[name].errors) for name in names)
    arrays = []
    for name in names:
        errors = sequences[name].errors
        indices = np.linspace(0, len(errors) - 1, count, dtype=int)
        arrays.append(errors[indices])
    return np.concatenate(arrays, axis=0)


def concatenate_vio_covariance(
    sequences: dict[str, PoseSequence],
    names: tuple[str, ...],
) -> np.ndarray:
    count = min(len(sequences[name].errors) for name in names)
    arrays = []
    for name in names:
        sequence = sequences[name]
        indices = np.linspace(0, len(sequence.errors) - 1, count, dtype=int)
        arrays.append(regularize_covariance(sequence.covariance[indices]))
    return np.concatenate(arrays, axis=0)


def _constant_covariance(covariance: np.ndarray, count: int) -> np.ndarray:
    return np.broadcast_to(covariance, (count, 6, 6)).copy()


def build_covariance_inputs(
    sequences: dict[str, PoseSequence], fold: Fold
) -> tuple[CovarianceInput, CovarianceInput]:
    fit_errors = concatenate_errors(sequences, fold.fit_names)
    empirical_bias, empirical_covariance = fit_empirical_covariance(fit_errors)
    test = sequences[fold.test_name]
    calibration_count = len(concatenate_errors(sequences, fold.calibration_names))

    openvins = CovarianceInput(
        name="openvins_full_6d",
        bias=np.zeros(6),
        calibration_covariance=concatenate_vio_covariance(
            sequences, fold.calibration_names
        ),
        test_covariance=regularize_covariance(test.covariance),
    )
    empirical = CovarianceInput(
        name="empirical_full_6d",
        bias=empirical_bias,
        calibration_covariance=_constant_covariance(
            empirical_covariance, calibration_count
        ),
        test_covariance=_constant_covariance(empirical_covariance, len(test.errors)),
    )
    return openvins, empirical


def _adaptive_radius(calibration_scores: np.ndarray, effective_alpha: float) -> float:
    # Finite calibration data cannot represent an infinite ACI set.  Clipping
    # to the attainable rank range makes this approximation explicit.
    minimum_alpha = 1.0 / (len(calibration_scores) + 1.0)
    bounded_alpha = float(np.clip(effective_alpha, minimum_alpha, 1.0 - minimum_alpha))
    rank = min(
        math.ceil((len(calibration_scores) + 1) * (1.0 - bounded_alpha)),
        len(calibration_scores),
    )
    return float(calibration_scores[rank - 1])


def run_aci(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    gamma: float,
) -> OnlineResult:
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    effective_alpha = alpha
    radii = np.empty(len(test_scores))
    covered = np.empty(len(test_scores), dtype=bool)
    effective_alphas = np.empty(len(test_scores))
    for index, score in enumerate(test_scores):
        radius = _adaptive_radius(calibration_scores, effective_alpha)
        radii[index] = radius
        effective_alphas[index] = effective_alpha
        covered[index] = score <= radius
        effective_alpha = float(
            np.clip(
                effective_alpha + gamma * (alpha - float(not covered[index])),
                0.0,
                1.0,
            )
        )
    return OnlineResult(radii, covered, effective_alphas)


def _pinball_loss(observation: float, prediction: np.ndarray, quantile: float) -> np.ndarray:
    residual = observation - prediction
    return np.where(residual >= 0.0, quantile * residual, (quantile - 1.0) * residual)


def run_agaci(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
    gammas: tuple[float, ...],
) -> OnlineResult:
    """Aggregate ACI radius experts with online exponential weights.

    This is the radial-set analogue of AgACI: experts differ only by gamma and
    are combined using quantile (pinball) loss.  Since SE(3) sets are nested by
    radius, aggregating radii is the direct counterpart of aggregating interval
    bounds.  The original paper uses BOA; exponential weights are used here to
    avoid an R dependency and are identified in output metadata.
    """
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    gammas_array = np.asarray(gammas, dtype=np.float64)
    expert_alphas = np.full(len(gammas_array), alpha, dtype=np.float64)
    log_weights = np.zeros(len(gammas_array), dtype=np.float64)
    radii = np.empty(len(test_scores))
    covered = np.empty(len(test_scores), dtype=bool)
    effective_alphas = np.empty(len(test_scores))
    quantile_level = 1.0 - alpha
    loss_scale = max(float(np.median(np.abs(calibration_scores))), 1e-9)

    for index, score in enumerate(test_scores):
        expert_radii = np.asarray(
            [_adaptive_radius(calibration_scores, value) for value in expert_alphas]
        )
        shifted = log_weights - np.max(log_weights)
        weights = np.exp(shifted)
        weights /= weights.sum()
        radius = float(weights @ expert_radii)
        radii[index] = radius
        effective_alphas[index] = float(weights @ expert_alphas)
        covered[index] = score <= radius

        expert_covered = score <= expert_radii
        expert_alphas = np.clip(
            expert_alphas
            + gammas_array * (alpha - (~expert_covered).astype(np.float64)),
            0.0,
            1.0,
        )
        losses = _pinball_loss(score, expert_radii, quantile_level) / loss_scale
        log_weights -= 0.5 * np.clip(losses, 0.0, 50.0)
    return OnlineResult(radii, covered, effective_alphas)


def _unit_ball_log_volume(dimension: int) -> float:
    return 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)


def log_volumes(covariance: np.ndarray, radii: np.ndarray) -> np.ndarray:
    covariance = regularize_covariance(covariance)
    _, log_determinant = np.linalg.slogdet(covariance)
    return _unit_ball_log_volume(6) + 6.0 * np.log(np.maximum(radii, 1e-12)) + 0.5 * log_determinant


def task_log_volumes(
    covariance: np.ndarray, radii: np.ndarray, body_point: np.ndarray
) -> np.ndarray:
    """Linearized 3D volume for a body-fixed task point.

    The nonlinear projection is used for visualization.  This Jacobian metric
    provides a stable per-timestamp size measure without replacing a curved set
    by its potentially much larger convex hull.
    """
    x, y, z = np.asarray(body_point, dtype=np.float64)
    point_skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    jacobian = np.concatenate([np.eye(3), -point_skew], axis=1)
    task_covariance = np.einsum(
        "ij,njk,lk->nil", jacobian, regularize_covariance(covariance), jacobian
    )
    task_covariance = regularize_covariance(task_covariance)
    _, log_determinant = np.linalg.slogdet(task_covariance)
    return _unit_ball_log_volume(3) + 3.0 * np.log(np.maximum(radii, 1e-12)) + 0.5 * log_determinant


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def rolling_coverage_error(covered: np.ndarray, target: float, window: int = 100) -> float:
    if len(covered) < window:
        return float(abs(covered.mean() - target))
    kernel = np.ones(window) / window
    rolling = np.convolve(covered.astype(float), kernel, mode="valid")
    return float(np.mean(np.abs(rolling - target)))


def evaluate(
    sequences: dict[str, PoseSequence],
    folds: list[Fold],
    alphas: tuple[float, ...],
    aci_gamma: float,
    agaci_gammas: tuple[float, ...],
    body_point: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, float, str], OnlineResult]]:
    rows: list[dict[str, object]] = []
    online_results: dict[tuple[str, str, str, float, str], OnlineResult] = {}
    for fold in folds:
        calibration_errors = concatenate_errors(sequences, fold.calibration_names)
        test = sequences[fold.test_name]
        for covariance_input in build_covariance_inputs(sequences, fold):
            calibration_scores = mahalanobis_scores(
                calibration_errors,
                covariance_input.calibration_covariance,
                covariance_input.bias,
            )
            test_scores = mahalanobis_scores(
                test.errors,
                covariance_input.test_covariance,
                covariance_input.bias,
            )
            for alpha in alphas:
                methods = {
                    "ACI": run_aci(calibration_scores, test_scores, alpha, aci_gamma),
                    "AgACI-EWA": run_agaci(
                        calibration_scores, test_scores, alpha, agaci_gammas
                    ),
                }
                for method, result in methods.items():
                    key = (
                        fold.protocol,
                        fold.test_name,
                        covariance_input.name,
                        alpha,
                        method,
                    )
                    online_results[key] = result
                    logs = log_volumes(covariance_input.test_covariance, result.radii)
                    task_logs = task_log_volumes(
                        covariance_input.test_covariance, result.radii, body_point
                    )
                    target = 1.0 - alpha
                    coverage = float(result.covered.mean())
                    median_log_volume = float(np.median(logs))
                    rows.append(
                        {
                            "protocol": fold.protocol,
                            "fold_id": fold.fold_id,
                            "test_environment": fold.test_environment,
                            "fit_trajectories": "|".join(fold.fit_names),
                            "calibration_trajectories": "|".join(fold.calibration_names),
                            "test_trajectory": fold.test_name,
                            "covariance_input": covariance_input.name,
                            "method": method,
                            "agaci_aggregation": "exponential_weights" if method.startswith("AgACI") else "none",
                            "alpha": alpha,
                            "target_coverage": target,
                            "n_calibration": len(calibration_errors),
                            "n_test": len(test.errors),
                            "coverage": coverage,
                            "absolute_coverage_gap": abs(coverage - target),
                            "median_log_6d_volume": median_log_volume,
                            "mean_log_6d_volume": float(np.mean(logs)),
                            "median_log_task_volume_m3": float(np.median(task_logs)),
                            "mean_log_task_volume_m3": float(np.mean(task_logs)),
                            "coverage_per_median_volume": float(
                                coverage * math.exp(-np.clip(median_log_volume, -700, 700))
                            ),
                            "mean_radius": float(np.mean(result.radii)),
                            "mean_effective_alpha": float(np.mean(result.effective_alphas)),
                            "rolling_coverage_mae_100": rolling_coverage_error(
                                result.covered, target
                            ),
                            "longest_miss_streak": longest_false_run(result.covered),
                        }
                    )
    return pd.DataFrame(rows), online_results


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "protocol", "covariance_input", "method", "alpha", "target_coverage"
    ]
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            absolute_coverage_gap_mean=("absolute_coverage_gap", "mean"),
            median_log_6d_volume_mean=("median_log_6d_volume", "mean"),
            median_log_task_volume_m3_mean=("median_log_task_volume_m3", "mean"),
            rolling_coverage_mae_100_mean=("rolling_coverage_mae_100", "mean"),
            longest_miss_streak_mean=("longest_miss_streak", "mean"),
            folds=("test_trajectory", "count"),
        )
        .sort_values(group_columns)
    )
    pooled = []
    for key, group in frame.groupby(group_columns):
        pooled.append((*key, float(np.average(group["coverage"], weights=group["n_test"]))))
    pooled_frame = pd.DataFrame(pooled, columns=[*group_columns, "coverage_pooled"])
    return summary.merge(pooled_frame, on=group_columns, how="left")


def plot_tradeoffs(summary: pd.DataFrame, output_path: Path) -> None:
    protocols = list(summary["protocol"].drop_duplicates())
    figure, axes = plt.subplots(len(protocols), 2, figsize=(13, 4.5 * len(protocols)))
    if len(protocols) == 1:
        axes = np.asarray([axes])
    styles = {
        ("empirical_full_6d", "ACI"): ("tab:blue", "o", "-"),
        ("empirical_full_6d", "AgACI-EWA"): ("tab:cyan", "s", "--"),
        ("openvins_full_6d", "ACI"): ("tab:orange", "o", "-"),
        ("openvins_full_6d", "AgACI-EWA"): ("tab:red", "s", "--"),
    }
    for row, protocol in enumerate(protocols):
        protocol_summary = summary[summary["protocol"] == protocol]
        for (covariance_input, method), group in protocol_summary.groupby(["covariance_input", "method"]):
            group = group.sort_values("target_coverage")
            color, marker, linestyle = styles[(covariance_input, method)]
            label = f"{covariance_input.replace('_', ' ')} + {method}"
            axes[row, 0].plot(
                group["target_coverage"], group["coverage_mean"],
                color=color, marker=marker, linestyle=linestyle, label=label,
            )
            axes[row, 1].plot(
                group["coverage_mean"], group["median_log_task_volume_m3_mean"],
                color=color, marker=marker, linestyle=linestyle, label=label,
            )
        axes[row, 0].plot([0.45, 1.0], [0.45, 1.0], color="black", linewidth=1, linestyle=":")
        axes[row, 0].set(
            xlabel="Target coverage",
            ylabel="Observed trajectory-mean coverage",
            title=f"{protocol.replace('_', ' ')}: calibration",
        )
        axes[row, 1].set(
            xlabel="Observed coverage",
            ylabel="Mean median log task volume [m^3]",
            title=f"{protocol.replace('_', ' ')}: efficiency",
        )
        axes[row, 0].legend(fontsize=7)
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _task_trajectory(poses: np.ndarray, body_point: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(poses[:, 3:]).apply(
        np.broadcast_to(body_point, (len(poses), 3))
    ) + poses[:, :3]


def _add_surface(axis, boundary: np.ndarray, color) -> None:
    hull = ConvexHull(boundary)
    axis.add_collection3d(
        Poly3DCollection(
            boundary[hull.simplices],
            facecolor=color,
            edgecolor=color,
            linewidth=0.15,
            alpha=0.16,
        )
    )


def plot_regions(
    sequences: dict[str, PoseSequence],
    folds: list[Fold],
    online_results: dict[tuple[str, str, str, float, str], OnlineResult],
    alpha: float,
    body_point: np.ndarray,
    output_path: Path,
) -> None:
    fold = next(
        item
        for item in folds
        if item.protocol == "same_environment" and item.test_environment == "V1"
    )
    test = sequences[fold.test_name]
    covariance_inputs = {item.name: item for item in build_covariance_inputs(sequences, fold)}
    estimated_task = _task_trajectory(test.estimated_pose, body_point)
    ground_truth_task = _task_trajectory(test.ground_truth_pose, body_point)
    selected = np.linspace(0, len(test.errors) - 1, 4, dtype=int)
    figure = plt.figure(figsize=(15, 12))
    combinations = [
        ("openvins_full_6d", "ACI"),
        ("openvins_full_6d", "AgACI-EWA"),
        ("empirical_full_6d", "ACI"),
        ("empirical_full_6d", "AgACI-EWA"),
    ]
    colors = plt.get_cmap("tab10")
    for panel, (covariance_name, method) in enumerate(combinations, start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        axis.plot(*ground_truth_task.T, color="black", linewidth=1.4, label="GT task trajectory")
        axis.plot(*estimated_task.T, color="tab:orange", linewidth=1.1, label="VIO task trajectory")
        covariance_input = covariance_inputs[covariance_name]
        result = online_results[
            (fold.protocol, fold.test_name, covariance_name, alpha, method)
        ]
        plotted = [ground_truth_task, estimated_task]
        for region_number, pose_index in enumerate(selected):
            model = ConformalSE3Model(
                bias=covariance_input.bias,
                covariance=covariance_input.test_covariance[pose_index],
                quantile=result.radii[pose_index],
                alpha=alpha,
                covariance_source=covariance_name,
            )
            tangent = sample_tangent_ellipsoid(
                model, sample_count=1400, seed=1000 * panel + pose_index, interior=False
            )
            boundary = project_pose_set_to_task_space(
                test.estimated_pose[pose_index], tangent, body_point[None]
            )[:, 0]
            color = colors(region_number)
            _add_surface(axis, boundary, color)
            gt_point = ground_truth_task[pose_index]
            estimate_point = estimated_task[pose_index]
            covered = result.covered[pose_index]
            axis.scatter(*estimate_point, color=color, s=25)
            axis.scatter(
                *gt_point,
                color="green" if covered else "red",
                marker="*", s=90, edgecolor="black", linewidth=0.4,
            )
            plotted.append(boundary)
        points = np.concatenate(plotted)
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        radius = max(0.5 * np.max(maximum - minimum), 1e-3)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(f"{covariance_name.replace('_', ' ')}\n{method}, coverage={result.covered.mean():.3f}", fontsize=10)
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        if panel == 1:
            axis.legend(fontsize=7)
    figure.suptitle(
        f"Complete test trajectory {fold.test_name}, alpha={alpha:g}, body point={body_point.tolist()}\n"
        "Translucent surfaces are convex envelopes of nonlinear SE(3) task-point projections"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def plot_region_closeups(
    sequences: dict[str, PoseSequence],
    folds: list[Fold],
    online_results: dict[tuple[str, str, str, float, str], OnlineResult],
    alpha: float,
    body_point: np.ndarray,
    output_path: Path,
) -> None:
    """Show nonlinear projected-set shape in its dominant local 2D plane."""
    fold = next(
        item
        for item in folds
        if item.protocol == "same_environment" and item.test_environment == "V1"
    )
    test = sequences[fold.test_name]
    pose_index = int(0.65 * (len(test.errors) - 1))
    covariance_inputs = {item.name: item for item in build_covariance_inputs(sequences, fold)}
    combinations = [
        ("openvins_full_6d", "ACI"),
        ("openvins_full_6d", "AgACI-EWA"),
        ("empirical_full_6d", "ACI"),
        ("empirical_full_6d", "AgACI-EWA"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for panel, (axis, (covariance_name, method)) in enumerate(
        zip(axes.flat, combinations)
    ):
        covariance_input = covariance_inputs[covariance_name]
        result = online_results[
            (fold.protocol, fold.test_name, covariance_name, alpha, method)
        ]
        model = ConformalSE3Model(
            bias=covariance_input.bias,
            covariance=covariance_input.test_covariance[pose_index],
            quantile=result.radii[pose_index],
            alpha=alpha,
            covariance_source=covariance_name,
        )
        tangent = sample_tangent_ellipsoid(
            model, sample_count=30000, seed=4000 + panel, interior=True
        )
        points = project_pose_set_to_task_space(
            test.estimated_pose[pose_index], tangent, body_point[None]
        )[:, 0]
        center = points.mean(axis=0)
        _, _, right = np.linalg.svd(points - center, full_matrices=False)
        local = (points - center) @ right[:2].T
        histogram, x_edges, y_edges = np.histogram2d(
            local[:, 0], local[:, 1], bins=130
        )
        density = gaussian_filter(histogram.T, sigma=1.2)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        positive = density[density > 0]
        lower = max(float(np.quantile(positive, 0.18)), 0.015 * float(density.max()))
        levels = np.linspace(lower, float(density.max()), 8)
        axis.contourf(x_centers, y_centers, density, levels=levels, cmap="Blues", alpha=0.9)
        axis.contour(x_centers, y_centers, density, levels=[lower], colors="tab:blue", linewidths=1.5)

        estimated_task = _task_trajectory(
            test.estimated_pose[pose_index : pose_index + 1], body_point
        )[0]
        ground_truth_task = _task_trajectory(
            test.ground_truth_pose[pose_index : pose_index + 1], body_point
        )[0]
        estimated_local = (estimated_task - center) @ right[:2].T
        ground_truth_local = (ground_truth_task - center) @ right[:2].T
        axis.scatter(*estimated_local, color="tab:orange", marker="o", s=55, label="VIO task point")
        axis.scatter(
            *ground_truth_local,
            color="green" if result.covered[pose_index] else "red",
            marker="*", s=120, edgecolor="black", linewidth=0.5,
            label="GT task point",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Local principal axis 1 [m]")
        axis.set_ylabel("Local principal axis 2 [m]")
        axis.set_title(f"{covariance_name.replace('_', ' ')}\n{method}", fontsize=10)
        axis.grid(alpha=0.2)
        if panel == 0:
            axis.legend(fontsize=8)
    figure.suptitle(
        f"Nonlinear task-space region close-up, {fold.test_name}, pose {pose_index}, alpha={alpha:g}\n"
        "Filled contours approximate support from interior SE(3) samples; they are not Gaussian confidence contours"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=210)
    plt.close(figure)


def plot_openvins_alpha_surfaces(
    sequences: dict[str, PoseSequence],
    folds: list[Fold],
    online_results: dict[tuple[str, str, str, float, str], OnlineResult],
    alphas: tuple[float, ...],
    body_point: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    """Project one fixed OpenVINS pose without convexifying the point cloud."""
    fold = next(
        item
        for item in folds
        if item.protocol == "pooled_same_robot" and item.test_name == "OV_V1_02_medium"
    )
    test = sequences[fold.test_name]
    covariance_input = {
        item.name: item for item in build_covariance_inputs(sequences, fold)
    }["openvins_full_6d"]
    methods = ("ACI", "AgACI-EWA")
    ordered_alphas = tuple(sorted(alphas))
    reference_alpha = min(ordered_alphas, key=lambda value: abs(value - 0.10))
    reference_result = online_results[
        (
            fold.protocol,
            fold.test_name,
            "openvins_full_6d",
            reference_alpha,
            "AgACI-EWA",
        )
    ]
    rotation_eigenvalues = np.linalg.eigvalsh(
        covariance_input.test_covariance[:, 3:, 3:]
    )
    angular_extent = reference_result.radii * np.sqrt(
        np.maximum(rotation_eigenvalues[:, -1], 0.0)
    )
    anisotropy = rotation_eigenvalues[:, -1] / np.maximum(
        rotation_eigenvalues[:, 0], 1e-15
    )
    moderate_extent = (angular_extent >= 0.12) & (angular_extent <= 0.80)
    selection_score = np.where(
        moderate_extent,
        np.log1p(anisotropy) - ((angular_extent - 0.35) / 0.30) ** 2,
        -np.inf,
    )
    pose_index = int(np.argmax(selection_score))
    figure, axes = plt.subplots(
        len(methods),
        len(ordered_alphas),
        figsize=(4.2 * len(ordered_alphas), 8.2),
        squeeze=False,
    )
    rows = []
    covariance = covariance_input.test_covariance[pose_index]
    estimated_task = _task_trajectory(
        test.estimated_pose[pose_index : pose_index + 1], body_point
    )[0]
    ground_truth_task = _task_trajectory(
        test.ground_truth_pose[pose_index : pose_index + 1], body_point
    )[0]

    for row_index, method in enumerate(methods):
        for column_index, alpha in enumerate(ordered_alphas):
            result = online_results[
                (fold.protocol, fold.test_name, "openvins_full_6d", alpha, method)
            ]
            radius = float(result.radii[pose_index])
            model = ConformalSE3Model(
                bias=covariance_input.bias,
                covariance=covariance,
                quantile=radius,
                alpha=alpha,
                covariance_source="openvins_full_6d",
            )
            tangent = sample_tangent_ellipsoid(
                model,
                sample_count=12000,
                seed=10000 + row_index * 100 + column_index,
                interior=True,
            )
            points = project_pose_set_to_task_space(
                test.estimated_pose[pose_index], tangent, body_point[None]
            )[:, 0]
            center = points.mean(axis=0)
            _, _, principal_axes = np.linalg.svd(
                points - center, full_matrices=False
            )
            local = (points - center) @ principal_axes[:2].T
            estimate_local = (estimated_task - center) @ principal_axes[:2].T
            ground_truth_local = (ground_truth_task - center) @ principal_axes[:2].T
            task_volume = float(
                np.exp(
                    task_log_volumes(
                        covariance[None], np.array([radius]), body_point
                    )[0]
                )
            )
            covered = bool(result.covered[pose_index])
            axis = axes[row_index, column_index]
            axis.scatter(
                local[:, 0],
                local[:, 1],
                s=1.0,
                alpha=0.055,
                color="tab:blue",
                rasterized=True,
            )
            axis.scatter(
                *estimate_local,
                color="tab:orange",
                marker="o",
                s=35,
                label="VIO" if column_index == 0 else None,
            )
            axis.scatter(
                *ground_truth_local,
                color="green" if covered else "red",
                marker="*",
                s=75,
                edgecolor="black",
                linewidth=0.4,
                label="GT" if column_index == 0 else None,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_title(
                f"target={1.0-alpha:.0%}\n"
                f"r={radius:.2f}, V={task_volume:.2f} m³, "
                f"{'covered' if covered else 'missed'}",
                fontsize=9,
            )
            if row_index == len(methods) - 1:
                axis.set_xlabel("Local principal axis 1 [m]")
            if column_index == 0:
                axis.set_ylabel(f"{method}\nLocal principal axis 2 [m]")
                axis.legend(fontsize=7)
            rows.append(
                {
                    "protocol": fold.protocol,
                    "test_trajectory": fold.test_name,
                    "pose_index": pose_index,
                    "method": method,
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "adaptive_radius": radius,
                    "task_volume_m3": task_volume,
                    "covered": covered,
                    "body_point": body_point.tolist(),
                }
            )

    figure.suptitle(
        f"OpenVINS nonlinear task-space samples at one fixed pose\n"
        f"{fold.test_name}, pose={pose_index}, body point={body_point.tolist()}\n"
        "Point clouds are direct SE(3) projections; no convex hull is applied"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    return pd.DataFrame(rows)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(","))


def parse_point(value: str) -> np.ndarray:
    point = np.asarray(parse_float_tuple(value), dtype=np.float64)
    if point.shape != (3,):
        raise argparse.ArgumentTypeError("Body point must have form x,y,z")
    return point


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=project_root / "datasets" / "OPEN_VINS")
    parser.add_argument("--output-dir", type=Path, default=Path("results/adaptive_openvins_full_covariance"))
    parser.add_argument("--alphas", type=parse_float_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--aci-gamma", type=float, default=0.01)
    parser.add_argument("--agaci-gammas", type=parse_float_tuple, default=DEFAULT_GAMMAS)
    parser.add_argument("--covariance-frame", choices=("body", "world", "position_world_rotation_body", "position_body_rotation_world"), default="position_world_rotation_body")
    parser.add_argument("--max-time-difference", type=float, default=0.05)
    parser.add_argument("--body-point", type=parse_point, default=np.array([3.0, 0.0, 0.0]))
    parser.add_argument("--region-alpha", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Adaptive OpenVINS experiment")
    print("- Complete trajectories define fit/calibration/test folds; no trajectory fragments.")
    print("- Fit/calibration trajectories are sample-balanced before pooling environments.")
    print("- ACI and AgACI state reset for every test trajectory.")
    print("- OpenVINS covariance uses dynamic Pt and Pr from every trajectory.")
    print("- Empirical covariance is fitted only from fit trajectories.")
    sequences = load_sequences(
        args.dataset_dir, args.covariance_frame, args.max_time_difference
    )
    same_environment_folds = build_same_environment_folds(sequences)
    pooled_folds = build_pooled_same_robot_folds(sequences)
    cross_environment_folds = build_cross_environment_folds(sequences)
    folds = same_environment_folds + pooled_folds + cross_environment_folds
    print(
        f"Loaded {len(sequences)} trajectories and constructed {len(folds)} folds: "
        f"{len(same_environment_folds)} same-environment, "
        f"{len(pooled_folds)} pooled, {len(cross_environment_folds)} cross-environment."
    )
    frame, online_results = evaluate(
        sequences, folds, args.alphas, args.aci_gamma, args.agaci_gammas,
        args.body_point,
    )
    summary = summarize(frame)
    frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    openvins_summary = summary[
        summary["covariance_input"] == "openvins_full_6d"
    ].copy()
    openvins_summary["task_volume_m3"] = np.exp(
        openvins_summary["median_log_task_volume_m3_mean"]
    )
    openvins_summary.to_csv(
        args.output_dir / "openvins_aci_agaci_alpha_summary.csv", index=False
    )
    plot_tradeoffs(summary, args.output_dir / "coverage_volume_tradeoff.png")
    if args.region_alpha in args.alphas:
        plot_regions(
            sequences, folds, online_results, args.region_alpha,
            args.body_point, args.output_dir / "task_space_regions.png",
        )
        plot_region_closeups(
            sequences, folds, online_results, args.region_alpha,
            args.body_point, args.output_dir / "task_space_region_closeups.png",
        )
    sample_metrics = plot_openvins_alpha_surfaces(
        sequences,
        folds,
        online_results,
        args.alphas,
        args.body_point,
        args.output_dir / "openvins_fixed_pose_alpha_surfaces.png",
    )
    sample_metrics.to_csv(
        args.output_dir / "openvins_fixed_pose_alpha_surfaces.csv", index=False
    )
    print(summary.to_string(index=False))
    print(f"Saved results under {args.output_dir}")


if __name__ == "__main__":
    main()
