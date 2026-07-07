"""Leakage-safe conformal experiments for the aligned OpenVINS trajectories.

The uploaded comparison CSVs contain pose estimates and ground truth, but no
per-pose VIO covariance. This runner therefore evaluates the pose-only form of
the method: fit a full empirical covariance in se(3), calibrate its radius with
split conformal prediction, and test on disjoint temporal blocks or sequences.

Use ``openvins_covariance_experiment.py`` for aligned OpenVINS files that
contain per-pose VIO covariance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.conformal_prediction.se3 import (
    conformal_quantile,
    mahalanobis_scores,
    regularize_covariance,
    se3_body_error,
)

POSE_COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ENVIRONMENTS = ("MH", "V1", "V2")


@dataclass(frozen=True)
class Trajectory:
    name: str
    environment: str
    timestamps: np.ndarray
    estimated_pose: np.ndarray
    ground_truth_pose: np.ndarray
    errors: np.ndarray

    def take(self, indices: np.ndarray) -> "Trajectory":
        return Trajectory(
            name=self.name,
            environment=self.environment,
            timestamps=self.timestamps[indices],
            estimated_pose=self.estimated_pose[indices],
            ground_truth_pose=self.ground_truth_pose[indices],
            errors=self.errors[indices],
        )


def print_methodology(alpha: float, sample_period: float, gap_seconds: float) -> None:
    target = 1.0 - alpha
    print("=" * 76)
    print("OPENVINS SE(3) CONFORMAL EVALUATION")
    print("=" * 76)
    print("Methodological updates and reasons")
    print("1. Body-frame SE(3) residual: xi = Log(T_est^{-1} T_gt).")
    print("   Reason: it matches a right perturbation and is invariant to world-frame changes.")
    print("2. No additional trajectory alignment is applied.")
    print("   Reason: the uploaded prediction and ground-truth CSVs are already aligned.")
    print("3. Fit, conformal-calibration, and test data are disjoint.")
    print("   Reason: covariance fitting and quantile calibration on the same samples is optimistic.")
    print("4. Whole temporal blocks or whole sequences remain in one split.")
    print("   Reason: adjacent VIO poses are strongly correlated; random frame splitting leaks data.")
    print(f"5. Poses are sampled every {sample_period:.2f} s and temporal gaps are {gap_seconds:.1f} s.")
    print("   Reason: this reduces near-duplicate samples and dependence at split boundaries.")
    print("6. Full 6x6 covariance includes translation-rotation cross-correlation.")
    print("   Reason: block-diagonal covariance cannot represent coupled SE(3) uncertainty.")
    print(f"7. Finite-sample split-CP targets {target:.1%} coverage.")
    print("   Reason: CP calibrates the radius without assuming Gaussian error distributions.")
    print("8. Euclidean position CP is retained as a sphere baseline.")
    print("   Reason: coverage and volume must be compared against the simpler representation.")
    print()
    print("Dataset limitation")
    print("The uploaded OpenVINS CSVs do not contain VIO covariance. Results below use an")
    print("empirical SE(3) covariance learned from fit data, not dynamic per-pose VIO covariance.")
    print("=" * 76)


def environment_from_name(name: str) -> str:
    for environment in ENVIRONMENTS:
        if name.startswith(environment + "_"):
            return environment
    raise ValueError(f"Cannot infer environment from sequence name {name!r}")


def _read_pose_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"timestamp", *POSE_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[["timestamp", *POSE_COLUMNS]].dropna().sort_values("timestamp")
    values = frame[POSE_COLUMNS].to_numpy(dtype=np.float64)
    quaternion_norm = np.linalg.norm(values[:, 3:], axis=1)
    valid = quaternion_norm > 1e-12
    if not np.all(valid):
        frame = frame.loc[valid].copy()
        values = values[valid]
        quaternion_norm = quaternion_norm[valid]
    frame.loc[:, ["qx", "qy", "qz", "qw"]] = (
        values[:, 3:] / quaternion_norm[:, None]
    )
    return frame.reset_index(drop=True)


def associate_poses(
    estimate: pd.DataFrame,
    ground_truth: pd.DataFrame,
    max_time_difference: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate_times = estimate["timestamp"].to_numpy(dtype=np.float64)
    gt_times = ground_truth["timestamp"].to_numpy(dtype=np.float64)
    insertion = np.searchsorted(gt_times, estimate_times)
    right = np.clip(insertion, 0, len(gt_times) - 1)
    left = np.clip(insertion - 1, 0, len(gt_times) - 1)
    choose_left = np.abs(estimate_times - gt_times[left]) <= np.abs(
        estimate_times - gt_times[right]
    )
    matched = np.where(choose_left, left, right)
    time_difference = np.abs(estimate_times - gt_times[matched])
    keep = time_difference <= max_time_difference
    if not np.any(keep):
        raise ValueError(
            f"No pose associations within {max_time_difference:.3f} seconds"
        )
    return (
        estimate_times[keep],
        estimate.loc[keep, POSE_COLUMNS].to_numpy(dtype=np.float64),
        ground_truth.iloc[matched[keep]][POSE_COLUMNS].to_numpy(dtype=np.float64),
    )


def time_subsample_indices(timestamps: np.ndarray, period: float) -> np.ndarray:
    if period <= 0:
        return np.arange(len(timestamps))
    selected = [0]
    next_time = timestamps[0] + period
    for index in range(1, len(timestamps)):
        if timestamps[index] >= next_time:
            selected.append(index)
            next_time = timestamps[index] + period
    return np.asarray(selected, dtype=np.int64)


def load_openvins_dataset(
    dataset_dir: Path,
    max_time_difference: float,
    sample_period: float,
) -> dict[str, Trajectory]:
    trajectories: dict[str, Trajectory] = {}
    for sequence_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        ground_truth_files = sorted(sequence_dir.glob("GT_*.csv"))
        estimate_files = sorted(
            path for path in sequence_dir.glob("*.csv") if not path.name.startswith("GT_")
        )
        if len(ground_truth_files) != 1 or len(estimate_files) != 1:
            raise ValueError(
                f"{sequence_dir} must contain one estimate CSV and one GT CSV"
            )
        estimate = _read_pose_csv(estimate_files[0])
        ground_truth = _read_pose_csv(ground_truth_files[0])
        timestamps, estimated_pose, ground_truth_pose = associate_poses(
            estimate, ground_truth, max_time_difference
        )
        indices = time_subsample_indices(timestamps, sample_period)
        estimated_pose = estimated_pose[indices]
        ground_truth_pose = ground_truth_pose[indices]
        trajectory = Trajectory(
            name=sequence_dir.name,
            environment=environment_from_name(sequence_dir.name),
            timestamps=timestamps[indices],
            estimated_pose=estimated_pose,
            ground_truth_pose=ground_truth_pose,
            errors=se3_body_error(estimated_pose, ground_truth_pose),
        )
        trajectories[trajectory.name] = trajectory
        print(
            f"Loaded {trajectory.name:18s}: {len(trajectory.errors):5d} associated samples"
        )
    return trajectories


def temporal_split(
    trajectory: Trajectory,
    gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(trajectory.errors)
    fit_end = max(1, int(0.40 * n))
    calibration_start = np.searchsorted(
        trajectory.timestamps, trajectory.timestamps[fit_end - 1] + gap_seconds
    )
    calibration_end = max(calibration_start + 1, int(0.70 * n))
    calibration_end = min(calibration_end, n - 1)
    test_start = np.searchsorted(
        trajectory.timestamps, trajectory.timestamps[calibration_end - 1] + gap_seconds
    )
    if test_start >= n:
        raise ValueError(f"{trajectory.name} is too short for the requested temporal gaps")
    return (
        np.arange(0, fit_end),
        np.arange(calibration_start, calibration_end),
        np.arange(test_start, n),
    )


def regularized_covariance(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(errors) < errors.shape[1] + 2:
        raise ValueError("Not enough fit samples for a full SE(3) covariance")
    bias = errors.mean(axis=0)
    centered = errors - bias
    covariance = centered.T @ centered / len(centered)
    diagonal_target = np.diag(np.diag(covariance))
    shrinkage = min(0.10, errors.shape[1] / max(len(errors), 1))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal_target
    scale = max(float(np.trace(covariance)) / errors.shape[1], 1e-12)
    covariance += np.eye(errors.shape[1]) * scale * 1e-8
    return bias, regularize_covariance(covariance)


def mahalanobis(errors: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return mahalanobis_scores(errors, covariance)


def ellipsoid_volume(covariance: np.ndarray, radius: float) -> float:
    dimension = covariance.shape[0]
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("nan")
    log_unit_ball = (
        0.5 * dimension * math.log(math.pi)
        - math.lgamma(0.5 * dimension + 1.0)
    )
    return float(math.exp(log_unit_ball + dimension * math.log(radius) + 0.5 * log_det))


def concatenate_errors(
    trajectories: dict[str, Trajectory],
    names: Iterable[str],
) -> np.ndarray:
    arrays = [trajectories[name].errors for name in names]
    if not arrays:
        raise ValueError("A split contains no trajectories")
    return np.concatenate(arrays, axis=0)


def evaluate_fold(
    protocol: str,
    fold: str,
    fit_errors: np.ndarray,
    calibration_errors: np.ndarray,
    test_errors: np.ndarray,
    alpha: float,
    fit_sequences: str,
    calibration_sequences: str,
    test_sequences: str,
) -> dict[str, float | int | str]:
    bias, covariance = regularized_covariance(fit_errors)
    calibration_centered = calibration_errors - bias
    test_centered = test_errors - bias

    q_joint = conformal_quantile(
        mahalanobis(calibration_centered, covariance), alpha
    )
    joint_scores = mahalanobis(test_centered, covariance)
    block_covariance = covariance.copy()
    block_covariance[:3, 3:] = 0.0
    block_covariance[3:, :3] = 0.0
    q_block = conformal_quantile(
        mahalanobis(calibration_centered, block_covariance), alpha
    )
    block_scores = mahalanobis(test_centered, block_covariance)

    translation_covariance = covariance[:3, :3]
    rotation_covariance = covariance[3:, 3:]
    q_translation = conformal_quantile(
        mahalanobis(calibration_centered[:, :3], translation_covariance), alpha
    )
    q_rotation = conformal_quantile(
        mahalanobis(calibration_centered[:, 3:], rotation_covariance), alpha
    )
    translation_scores = mahalanobis(test_centered[:, :3], translation_covariance)
    rotation_scores = mahalanobis(test_centered[:, 3:], rotation_covariance)

    position_calibration_scores = np.linalg.norm(calibration_errors[:, :3], axis=1)
    q_position = conformal_quantile(position_calibration_scores, alpha)
    position_test_scores = np.linalg.norm(test_errors[:, :3], axis=1)

    joint_volume = ellipsoid_volume(covariance, q_joint)
    block_volume = ellipsoid_volume(block_covariance, q_block)
    translation_projection_volume = ellipsoid_volume(
        translation_covariance, q_joint
    )
    translation_marginal_volume = ellipsoid_volume(
        translation_covariance, q_translation
    )
    rotation_marginal_volume = ellipsoid_volume(rotation_covariance, q_rotation)
    sphere_volume = 4.0 * math.pi * q_position**3 / 3.0

    joint_coverage = float(np.mean(joint_scores <= q_joint))
    block_coverage = float(np.mean(block_scores <= q_block))
    translation_projection_coverage = float(np.mean(translation_scores <= q_joint))
    translation_marginal_coverage = float(
        np.mean(translation_scores <= q_translation)
    )
    rotation_marginal_coverage = float(np.mean(rotation_scores <= q_rotation))
    position_sphere_coverage = float(np.mean(position_test_scores <= q_position))

    return {
        "protocol": protocol,
        "fold": fold,
        "target_coverage": 1.0 - alpha,
        "fit_sequences": fit_sequences,
        "calibration_sequences": calibration_sequences,
        "test_sequences": test_sequences,
        "n_fit": len(fit_errors),
        "n_calibration": len(calibration_errors),
        "n_test": len(test_errors),
        "joint_6d_coverage": joint_coverage,
        "joint_6d_coverage_gap": joint_coverage - (1.0 - alpha),
        "block_diagonal_6d_coverage": block_coverage,
        "block_diagonal_6d_coverage_gap": block_coverage - (1.0 - alpha),
        "translation_projection_coverage": translation_projection_coverage,
        "translation_marginal_coverage": translation_marginal_coverage,
        "rotation_marginal_coverage": rotation_marginal_coverage,
        "position_sphere_coverage": position_sphere_coverage,
        "q_joint": q_joint,
        "q_block_diagonal": q_block,
        "q_translation": q_translation,
        "q_rotation": q_rotation,
        "q_position_sphere_m": q_position,
        "joint_6d_volume": joint_volume,
        "joint_6d_log10_volume": math.log10(max(joint_volume, 1e-300)),
        "block_diagonal_6d_volume": block_volume,
        "block_diagonal_6d_log10_volume": math.log10(max(block_volume, 1e-300)),
        "translation_projection_volume_m3": translation_projection_volume,
        "translation_marginal_volume_m3": translation_marginal_volume,
        "rotation_marginal_volume_rad3": rotation_marginal_volume,
        "position_sphere_volume_m3": sphere_volume,
        "joint_coverage_per_volume": joint_coverage / max(joint_volume, 1e-300),
        "block_diagonal_coverage_per_volume": (
            block_coverage / max(block_volume, 1e-300)
        ),
        "translation_coverage_per_volume": (
            translation_projection_coverage
            / max(translation_projection_volume, 1e-300)
        ),
        "position_sphere_coverage_per_volume": (
            position_sphere_coverage / max(sphere_volume, 1e-300)
        ),
    }


def run_in_distribution(
    trajectories: dict[str, Trajectory],
    alpha: float,
    gap_seconds: float,
) -> list[dict]:
    rows = []
    for name, trajectory in sorted(trajectories.items()):
        fit_indices, calibration_indices, test_indices = temporal_split(
            trajectory, gap_seconds
        )
        rows.append(
            evaluate_fold(
                protocol="in_distribution",
                fold=name,
                fit_errors=trajectory.errors[fit_indices],
                calibration_errors=trajectory.errors[calibration_indices],
                test_errors=trajectory.errors[test_indices],
                alpha=alpha,
                fit_sequences=f"{name}:early",
                calibration_sequences=f"{name}:middle",
                test_sequences=f"{name}:late",
            )
        )
    return rows


def run_cross_sequence(
    trajectories: dict[str, Trajectory],
    alpha: float,
) -> list[dict]:
    rows = []
    for environment in ENVIRONMENTS:
        names = sorted(
            name
            for name, trajectory in trajectories.items()
            if trajectory.environment == environment
        )
        for held_out_index, test_name in enumerate(names):
            source_names = [name for name in names if name != test_name]
            calibration_name = source_names[held_out_index % len(source_names)]
            fit_names = [name for name in source_names if name != calibration_name]
            rows.append(
                evaluate_fold(
                    protocol="cross_sequence",
                    fold=test_name,
                    fit_errors=concatenate_errors(trajectories, fit_names),
                    calibration_errors=trajectories[calibration_name].errors,
                    test_errors=trajectories[test_name].errors,
                    alpha=alpha,
                    fit_sequences="|".join(fit_names),
                    calibration_sequences=calibration_name,
                    test_sequences=test_name,
                )
            )
    return rows


def run_cross_environment(
    trajectories: dict[str, Trajectory],
    alpha: float,
) -> list[dict]:
    rows = []
    for test_environment in ENVIRONMENTS:
        test_names = sorted(
            name
            for name, trajectory in trajectories.items()
            if trajectory.environment == test_environment
        )
        source_names = sorted(set(trajectories).difference(test_names))
        fit_names: list[str] = []
        calibration_names: list[str] = []
        for source_environment in ENVIRONMENTS:
            environment_names = [
                name
                for name in source_names
                if trajectories[name].environment == source_environment
            ]
            if not environment_names:
                continue
            calibration_names.append(environment_names[-1])
            fit_names.extend(environment_names[:-1])
        rows.append(
            evaluate_fold(
                protocol="cross_environment",
                fold=f"test_{test_environment}",
                fit_errors=concatenate_errors(trajectories, fit_names),
                calibration_errors=concatenate_errors(
                    trajectories, calibration_names
                ),
                test_errors=concatenate_errors(trajectories, test_names),
                alpha=alpha,
                fit_sequences="|".join(fit_names),
                calibration_sequences="|".join(calibration_names),
                test_sequences="|".join(test_names),
            )
        )
    return rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "joint_6d_coverage",
        "joint_6d_coverage_gap",
        "block_diagonal_6d_coverage",
        "block_diagonal_6d_coverage_gap",
        "translation_projection_coverage",
        "translation_marginal_coverage",
        "rotation_marginal_coverage",
        "position_sphere_coverage",
        "joint_6d_log10_volume",
        "block_diagonal_6d_log10_volume",
        "translation_projection_volume_m3",
        "position_sphere_volume_m3",
        "translation_coverage_per_volume",
        "position_sphere_coverage_per_volume",
    ]
    return (
        results.groupby("protocol", sort=False)[metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


def print_results(results: pd.DataFrame) -> None:
    columns = [
        "protocol",
        "fold",
        "joint_6d_coverage",
        "block_diagonal_6d_coverage",
        "translation_marginal_coverage",
        "rotation_marginal_coverage",
        "position_sphere_coverage",
        "translation_projection_volume_m3",
        "position_sphere_volume_m3",
    ]
    printable = results[columns].copy()
    coverage_columns = [column for column in columns if "coverage" in column]
    printable[coverage_columns] = printable[coverage_columns].map(
        lambda value: f"{value:.3f}"
    )
    for column in ("translation_projection_volume_m3", "position_sphere_volume_m3"):
        printable[column] = printable[column].map(lambda value: f"{value:.3e}")
    print()
    print("Per-fold results")
    print(printable.to_string(index=False))
    print()
    print("Protocol means")
    means = results.groupby("protocol", sort=False)[
        [
            "joint_6d_coverage",
            "block_diagonal_6d_coverage",
            "translation_marginal_coverage",
            "rotation_marginal_coverage",
            "position_sphere_coverage",
            "translation_projection_volume_m3",
            "position_sphere_volume_m3",
        ]
    ].mean()
    print(means.to_string(float_format=lambda value: f"{value:.4g}"))
    print()
    print("Pooled coverage weighted by test samples")
    pooled_rows = []
    for protocol, group in results.groupby("protocol", sort=False):
        weights = group["n_test"].to_numpy(dtype=np.float64)
        pooled_rows.append(
            {
                "protocol": protocol,
                "joint_6d": np.average(group["joint_6d_coverage"], weights=weights),
                "block_diagonal_6d": np.average(
                    group["block_diagonal_6d_coverage"], weights=weights
                ),
                "translation_marginal": np.average(
                    group["translation_marginal_coverage"], weights=weights
                ),
                "rotation_marginal": np.average(
                    group["rotation_marginal_coverage"], weights=weights
                ),
                "position_sphere": np.average(
                    group["position_sphere_coverage"], weights=weights
                ),
            }
        )
    print(
        pd.DataFrame(pooled_rows)
        .set_index("protocol")
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run leakage-safe SE(3) conformal experiments on OpenVINS"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "OPENVINS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "openvins_cp",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max-time-difference", type=float, default=0.05)
    parser.add_argument("--sample-period", type=float, default=0.20)
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie in (0, 1)")
    print_methodology(args.alpha, args.sample_period, args.gap_seconds)
    trajectories = load_openvins_dataset(
        args.dataset_dir,
        args.max_time_difference,
        args.sample_period,
    )
    protocols = [
        ("in-distribution", run_in_distribution(trajectories, args.alpha, args.gap_seconds)),
        ("cross-sequence", run_cross_sequence(trajectories, args.alpha)),
        ("cross-environment", run_cross_environment(trajectories, args.alpha)),
    ]
    for name, rows in protocols:
        print(f"Completed {name}: {len(rows)} folds")
    results = pd.DataFrame([row for _, rows in protocols for row in rows])
    summary = summarize_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "fold_results.csv"
    summary_path = args.output_dir / "protocol_summary.csv"
    config_path = args.output_dir / "run_config.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "max_time_difference": args.max_time_difference,
                "sample_period": args.sample_period,
                "gap_seconds": args.gap_seconds,
                "residual": "Log(T_est^{-1} T_gt)",
                "alignment": "none; input CSVs are already aligned",
                "covariance": "full empirical 6x6 fit on disjoint fit data",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print_results(results)
    print()
    print(f"Saved fold results:     {results_path}")
    print(f"Saved protocol summary: {summary_path}")
    print(f"Saved run configuration:{config_path}")


if __name__ == "__main__":
    main()
