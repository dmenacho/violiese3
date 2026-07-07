"""Four covariance models on aligned OpenVINS poses with SE(3) conformal sets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from utils.dataset_io import PoseSequence, load_aligned_openvins_pair
from utils.conformal_prediction.se3 import (
    ConformalSE3Model,
    conformal_quantile,
    fit_empirical_covariance,
    mahalanobis_scores,
    project_pose_set_to_task_space,
    regularize_covariance,
    sample_tangent_ellipsoid,
)


@dataclass(frozen=True)
class MethodResult:
    name: str
    model: ConformalSE3Model
    test_covariance: np.ndarray
    coverage: float
    test_scores: np.ndarray


def log_ellipsoid_volume(covariance: np.ndarray, radius: float) -> np.ndarray:
    dimension = covariance.shape[-1]
    sign, log_determinant = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise ValueError("Covariance must be positive definite for volume")
    log_unit_ball = (
        0.5 * dimension * math.log(math.pi)
        - math.lgamma(0.5 * dimension + 1.0)
    )
    return log_unit_ball + dimension * math.log(radius) + 0.5 * log_determinant


def split_indices(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_end = int(0.30 * length)
    calibration_end = int(0.60 * length)
    return (
        np.arange(0, fit_end),
        np.arange(fit_end, calibration_end),
        np.arange(calibration_end, length),
    )


def _constant_covariance(covariance: np.ndarray, count: int) -> np.ndarray:
    return np.broadcast_to(covariance, (count, 6, 6)).copy()


def _calibrated_model(
    name: str,
    covariance_source: str,
    bias: np.ndarray,
    calibration_errors: np.ndarray,
    calibration_covariance: np.ndarray,
    test_errors: np.ndarray,
    test_covariance: np.ndarray,
    alpha: float,
) -> MethodResult:
    calibration_scores = mahalanobis_scores(
        calibration_errors,
        calibration_covariance,
        bias,
    )
    quantile = conformal_quantile(calibration_scores, alpha)
    test_scores = mahalanobis_scores(test_errors, test_covariance, bias)
    model = ConformalSE3Model(
        bias=bias,
        covariance=calibration_covariance,
        quantile=quantile,
        alpha=alpha,
        covariance_source=covariance_source,
    )
    return MethodResult(
        name=name,
        model=model,
        test_covariance=test_covariance,
        coverage=float(np.mean(test_scores <= quantile)),
        test_scores=test_scores,
    )


def build_methods(
    sequence: PoseSequence,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> list[MethodResult]:
    fit_errors = sequence.errors[fit_indices]
    calibration_errors = sequence.errors[calibration_indices]
    test_errors = sequence.errors[test_indices]
    vio_covariance = regularize_covariance(sequence.covariance)
    if np.isnan(vio_covariance).any():
        raise ValueError(
            "This sequence lacks Pr rotation covariance. Use OV_V1_01 for the "
            "full four-method 6D comparison."
        )

    empirical_bias, empirical_full = fit_empirical_covariance(fit_errors)
    empirical_centered = fit_errors - empirical_bias
    translation_covariance = (
        empirical_centered[:, :3].T @ empirical_centered[:, :3] / len(fit_errors)
    )
    rotation_covariance = (
        empirical_centered[:, 3:].T @ empirical_centered[:, 3:] / len(fit_errors)
    )
    empirical_blocks = np.zeros((6, 6))
    empirical_blocks[:3, :3] = translation_covariance
    empirical_blocks[3:, 3:] = rotation_covariance
    empirical_blocks = regularize_covariance(empirical_blocks)
    empirical_diagonal = regularize_covariance(
        np.diag(np.maximum(np.var(empirical_centered, axis=0), 1e-12))
    )

    method_specs = [
        (
            "OpenVINS covariance",
            "openvins",
            np.zeros(6),
            vio_covariance[calibration_indices],
            vio_covariance[test_indices],
        ),
        (
            "Empirical full 6D",
            "empirical_full",
            empirical_bias,
            _constant_covariance(empirical_full, len(calibration_indices)),
            _constant_covariance(empirical_full, len(test_indices)),
        ),
        (
            "Independent t/r blocks",
            "empirical_blocks",
            empirical_bias,
            _constant_covariance(empirical_blocks, len(calibration_indices)),
            _constant_covariance(empirical_blocks, len(test_indices)),
        ),
        (
            "Independent axis scales",
            "empirical_diagonal",
            empirical_bias,
            _constant_covariance(empirical_diagonal, len(calibration_indices)),
            _constant_covariance(empirical_diagonal, len(test_indices)),
        ),
    ]
    return [
        _calibrated_model(
            name,
            source,
            bias,
            calibration_errors,
            calibration_covariance,
            test_errors,
            test_covariance,
            alpha,
        )
        for name, source, bias, calibration_covariance, test_covariance in method_specs
    ]


def _task_trajectory(poses: np.ndarray, body_point: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(poses[:, 3:]).apply(
            np.broadcast_to(body_point, (len(poses), 3))
        )
        + poses[:, :3]
    )


def _equal_axes(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(0.5 * np.max(maximum - minimum), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _surface(axis, points: np.ndarray, color) -> None:
    hull = ConvexHull(points)
    axis.add_collection3d(
        Poly3DCollection(
            points[hull.simplices],
            facecolor=color,
            edgecolor=color,
            linewidth=0.12,
            alpha=0.10,
        )
    )


def plot_methods(
    sequence: PoseSequence,
    methods: list[MethodResult],
    test_indices: np.ndarray,
    body_point: np.ndarray,
    region_count: int,
    region_samples: int,
    covariance_frame: str,
    output_path: Path,
) -> None:
    selected_offsets = np.linspace(
        0,
        len(test_indices) - 1,
        min(region_count, len(test_indices)),
        dtype=int,
    )
    estimated_task = _task_trajectory(sequence.estimated_pose[test_indices], body_point)
    ground_truth_task = _task_trajectory(
        sequence.ground_truth_pose[test_indices], body_point
    )
    figure = plt.figure(figsize=(18, 15))
    colors = plt.get_cmap("tab10")

    for method_index, method in enumerate(methods):
        axis = figure.add_subplot(2, 2, method_index + 1, projection="3d")
        axis.plot(*ground_truth_task.T, color="black", linewidth=1.8, label="GT task point")
        axis.plot(
            *estimated_task.T,
            color="tab:orange",
            linewidth=1.4,
            label="Estimated task point",
        )
        plotted = [estimated_task, ground_truth_task]
        selected_covered = 0
        for region_number, test_offset in enumerate(selected_offsets):
            pose_index = int(test_indices[test_offset])
            covariance = method.test_covariance[test_offset]
            dynamic_model = ConformalSE3Model(
                bias=method.model.bias,
                covariance=covariance,
                quantile=method.model.quantile,
                alpha=method.model.alpha,
                covariance_source=method.model.covariance_source,
            )
            tangent_boundary = sample_tangent_ellipsoid(
                dynamic_model,
                sample_count=region_samples,
                seed=pose_index,
                interior=False,
            )
            boundary = project_pose_set_to_task_space(
                sequence.estimated_pose[pose_index],
                tangent_boundary,
                body_point[None, :],
            )[:, 0, :]
            color = colors(region_number % 10)
            _surface(axis, boundary, color)
            is_covered = bool(
                method.test_scores[test_offset] <= method.model.quantile
            )
            selected_covered += int(is_covered)
            gt_point = ground_truth_task[test_offset]
            estimate_point = estimated_task[test_offset]
            axis.scatter(*estimate_point, color=color, marker="o", s=25)
            axis.scatter(
                *gt_point,
                color="green" if is_covered else "red",
                edgecolor="black",
                linewidth=0.4,
                marker="*",
                s=90,
            )
            axis.text(
                *gt_point,
                f" {region_number + 1}",
                fontsize=7,
                color="green" if is_covered else "red",
            )
            plotted.append(boundary)
        axis.set_title(
            f"{method.name}\ncoverage={method.coverage:.3f}, "
            f"q={method.model.quantile:.2f}, "
            f"selected={selected_covered}/{len(selected_offsets)}"
        )
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.plot([], [], [], color="green", marker="*", linestyle="", label="Covered")
        axis.plot([], [], [], color="red", marker="*", linestyle="", label="Missed")
        axis.legend(fontsize=7, loc="upper left")
        _equal_axes(axis, np.concatenate(plotted))

    figure.suptitle(
        f"{sequence.name}: aligned OpenVINS covariance comparison\n"
        f"body point={body_point.tolist()}, no additional alignment, "
        f"covariance frame={covariance_frame}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def parse_point(value: str) -> np.ndarray:
    point = np.asarray([float(part) for part in value.split(",")])
    if point.shape != (3,):
        raise argparse.ArgumentTypeError("Body point must have form x,y,z")
    return point


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help="Run one sequence instead of every sequence under --dataset-dir.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "OPEN_VINS",
    )
    parser.add_argument(
        "--plot-sequence",
        default="OV_V1_01_easy",
        help="Sequence plotted during a full-dataset run.",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--covariance-frame",
        choices=(
            "world",
            "body",
            "position_world_rotation_body",
            "position_body_rotation_world",
        ),
        default="position_world_rotation_body",
    )
    parser.add_argument("--body-point", type=parse_point, default=np.array([3.0, 0.0, 0.0]))
    parser.add_argument("--region-count", type=int, default=3)
    parser.add_argument("--region-samples", type=int, default=900)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/openvins_covariance/four_methods.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Aligned OpenVINS experiment: no trajectory alignment applied.")
    print("Split: 30% covariance fit, 30% CP calibration, 40% test.")
    sequence_dirs = (
        [args.sequence_dir]
        if args.sequence_dir is not None
        else sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
    )
    rows = []
    plotted = False
    for sequence_dir in sequence_dirs:
        estimate_files = sorted(sequence_dir.glob("OV_*.csv"))
        ground_truth_files = sorted(sequence_dir.glob("GT_*.csv"))
        if len(estimate_files) != 1 or len(ground_truth_files) != 1:
            continue
        sequence = load_aligned_openvins_pair(
            estimate_files[0],
            ground_truth_files[0],
            covariance_frame=args.covariance_frame,
        )
        fit_indices, calibration_indices, test_indices = split_indices(
            len(sequence.errors)
        )
        methods = build_methods(
            sequence,
            fit_indices,
            calibration_indices,
            test_indices,
            args.alpha,
        )
        for method in methods:
            log_volumes = log_ellipsoid_volume(
                method.test_covariance,
                method.model.quantile,
            )
            rows.append(
                {
                    "sequence": sequence_dir.name,
                    "method": method.name,
                    "coverage": method.coverage,
                    "coverage_error": method.coverage - (1.0 - args.alpha),
                    "quantile": method.model.quantile,
                    "median_log_6d_volume": float(np.median(log_volumes)),
                    "mean_log_6d_volume": float(np.mean(log_volumes)),
                    "target_coverage": 1.0 - args.alpha,
                    "n_fit": len(fit_indices),
                    "n_calibration": len(calibration_indices),
                    "n_test": len(test_indices),
                    "covariance_frame": args.covariance_frame,
                    "body_point": args.body_point.tolist(),
                }
            )
            print(
                f"{sequence_dir.name:22s} {method.name:28s} "
                f"coverage={method.coverage:.3f} q={method.model.quantile:.3f} "
                f"median_log_volume={np.median(log_volumes):.2f}"
            )
        should_plot = args.sequence_dir is not None or sequence_dir.name == args.plot_sequence
        if should_plot and not plotted:
            plot_methods(
                sequence,
                methods,
                test_indices,
                args.body_point,
                args.region_count,
                args.region_samples,
                args.covariance_frame,
                args.output,
            )
            plotted = True

    metrics_path = args.output.with_suffix(".csv")
    results = pd.DataFrame(rows)
    results.to_csv(metrics_path, index=False)
    summary = results.groupby("method")[
        ["coverage", "coverage_error", "quantile", "median_log_6d_volume"]
    ].agg(["mean", "median", "min", "max"])
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary.to_csv(summary_path)
    primary_summary = summary.loc[["OpenVINS covariance", "Empirical full 6D"]]
    primary_summary_path = args.output.with_name(
        f"{args.output.stem}_primary_summary.csv"
    )
    primary_summary.to_csv(primary_summary_path)
    print()
    print(
        summary.to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Saved four-method plot: {args.output}")
    print(f"Saved four-method metrics: {metrics_path}")
    print(f"Saved aggregate summary: {summary_path}")
    print(f"Saved VIO/empirical summary: {primary_summary_path}")


if __name__ == "__main__":
    main()
