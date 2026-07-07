"""Compare causal initial-block and benchmark full-trajectory alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from utils.dataset_io import load_merged_vio_csv
from utils.conformal_prediction.se3 import (
    ConformalSE3Model,
    calibrate_vio_covariance_model,
    mahalanobis_scores,
    project_pose_set_to_task_space,
    sample_tangent_ellipsoid,
)


ALIGNMENT_MODES = ("initial_block", "full_trajectory")


def split_indices(length: int) -> tuple[np.ndarray, np.ndarray]:
    calibration_start = int(0.40 * length)
    calibration_end = int(0.70 * length)
    return (
        np.arange(calibration_start, calibration_end),
        np.arange(calibration_end, length),
    )


def evaluate_file(
    csv_path: Path,
    alignment_mode: str,
    alignment_fraction: float,
    covariance_frame: str,
    alpha: float,
) -> dict[str, float | int | str]:
    sequence = load_merged_vio_csv(
        csv_path,
        alignment_mode=alignment_mode,
        alignment_fraction=alignment_fraction,
        covariance_frame=covariance_frame,
    )
    calibration_indices, test_indices = split_indices(len(sequence.errors))
    model = calibrate_vio_covariance_model(
        sequence.errors[calibration_indices],
        sequence.covariance[calibration_indices],
        alpha,
    )
    test_scores = mahalanobis_scores(
        sequence.errors[test_indices],
        sequence.covariance[test_indices],
    )
    position_error = np.linalg.norm(
        sequence.ground_truth_pose[:, :3] - sequence.estimated_pose[:, :3],
        axis=1,
    )
    rotation_error = np.linalg.norm(sequence.errors[:, 3:], axis=1)
    test_position_max = float(np.max(position_error[test_indices]))
    return {
        "sequence": sequence.name,
        "alignment_mode": alignment_mode,
        "alignment_fraction": alignment_fraction,
        "n_calibration": len(calibration_indices),
        "n_test": len(test_indices),
        "quantile": model.quantile,
        "test_6d_coverage": float(np.mean(test_scores <= model.quantile)),
        "all_position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "test_position_rmse_m": float(
            np.sqrt(np.mean(position_error[test_indices] ** 2))
        ),
        "test_position_max_m": test_position_max,
        "test_rotation_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(rotation_error[test_indices] ** 2)))
        ),
        "trajectory_status": "diverged" if test_position_max > 100.0 else "nominal",
    }


def _equal_3d_axes(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(0.5 * np.max(maximum - minimum), 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _task_point_trajectory(poses: np.ndarray, body_point: np.ndarray) -> np.ndarray:
    return (
        Rotation.from_quat(poses[:, 3:]).apply(
            np.broadcast_to(body_point, (len(poses), 3))
        )
        + poses[:, :3]
    )


def _draw_region_surface(
    axis,
    points: np.ndarray,
    color,
) -> None:
    try:
        hull = ConvexHull(points)
    except QhullError:
        axis.scatter(*points.T, s=2, alpha=0.12, color=color)
        return
    triangles = points[hull.simplices]
    surface = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor=color,
        linewidth=0.15,
        alpha=0.10,
    )
    axis.add_collection3d(surface)


def plot_test_trajectory_regions(
    csv_path: Path,
    alignment_mode: str,
    alignment_fraction: float,
    covariance_frame: str,
    alpha: float,
    body_point: np.ndarray,
    region_count: int,
    region_samples: int,
    output_path: Path,
) -> None:
    sequence = load_merged_vio_csv(
        csv_path,
        alignment_mode=alignment_mode,
        alignment_fraction=alignment_fraction,
        covariance_frame=covariance_frame,
    )
    calibration_indices, test_indices = split_indices(len(sequence.errors))
    model = calibrate_vio_covariance_model(
        sequence.errors[calibration_indices],
        sequence.covariance[calibration_indices],
        alpha,
    )
    test_scores = mahalanobis_scores(
        sequence.errors[test_indices],
        sequence.covariance[test_indices],
    )
    covered = test_scores <= model.quantile
    selected_offsets = np.linspace(
        0,
        len(test_indices) - 1,
        min(region_count, len(test_indices)),
        dtype=int,
    )
    selected_indices = test_indices[selected_offsets]

    estimated_origin_test = sequence.estimated_pose[test_indices, :3]
    ground_truth_origin_test = sequence.ground_truth_pose[test_indices, :3]
    estimated_task_test = _task_point_trajectory(
        sequence.estimated_pose[test_indices], body_point
    )
    ground_truth_task_test = _task_point_trajectory(
        sequence.ground_truth_pose[test_indices], body_point
    )
    figure = plt.figure(figsize=(18, 8))
    origin_axis = figure.add_subplot(121, projection="3d")
    task_axis = figure.add_subplot(122, projection="3d")
    origin_axis.plot(
        *ground_truth_origin_test.T,
        color="black",
        linewidth=2.0,
        label="Ground-truth camera origin",
    )
    origin_axis.plot(
        *estimated_origin_test.T,
        color="tab:orange",
        linewidth=1.6,
        label="Aligned VIO camera origin",
    )
    task_axis.plot(
        *ground_truth_task_test.T,
        color="black",
        linewidth=2.0,
        label="Ground-truth task point",
    )
    task_axis.plot(
        *estimated_task_test.T,
        color="tab:orange",
        linewidth=1.6,
        label="Estimated task point",
    )

    origin_plot_points = [estimated_origin_test, ground_truth_origin_test]
    task_plot_points = [estimated_task_test, ground_truth_task_test]
    region_colors = plt.get_cmap("tab10")
    for region_number, (pose_index, test_offset) in enumerate(
        zip(selected_indices, selected_offsets)
    ):
        dynamic_model = ConformalSE3Model(
            bias=model.bias,
            covariance=sequence.covariance[pose_index : pose_index + 1],
            quantile=model.quantile,
            alpha=model.alpha,
            covariance_source=model.covariance_source,
        )
        tangent_boundary = sample_tangent_ellipsoid(
            dynamic_model,
            index=0,
            sample_count=region_samples,
            seed=int(pose_index),
            interior=False,
        )
        projected_boundary = project_pose_set_to_task_space(
            sequence.estimated_pose[pose_index],
            tangent_boundary,
            body_point[None, :],
        )[:, 0, :]
        estimated_task_point = (
            Rotation.from_quat(sequence.estimated_pose[pose_index, 3:]).apply(body_point)
            + sequence.estimated_pose[pose_index, :3]
        )
        ground_truth_task_point = (
            Rotation.from_quat(sequence.ground_truth_pose[pose_index, 3:]).apply(
                body_point
            )
            + sequence.ground_truth_pose[pose_index, :3]
        )
        is_covered = bool(covered[test_offset])
        region_color = region_colors(region_number % 10)
        _draw_region_surface(task_axis, projected_boundary, region_color)
        origin_axis.scatter(
            *sequence.estimated_pose[pose_index, :3],
            marker="o",
            s=32,
            color=region_color,
        )
        origin_axis.scatter(
            *sequence.ground_truth_pose[pose_index, :3],
            marker="*",
            s=100,
            color="green" if is_covered else "red",
            edgecolor="black",
            linewidth=0.4,
        )
        task_axis.scatter(
            *estimated_task_point,
            marker="o",
            s=32,
            color=region_color,
        )
        task_axis.scatter(
            *ground_truth_task_point,
            marker="*",
            s=100,
            color="green" if is_covered else "red",
            edgecolor="black",
            linewidth=0.4,
        )
        task_axis.plot(
            [estimated_task_point[0], ground_truth_task_point[0]],
            [estimated_task_point[1], ground_truth_task_point[1]],
            [estimated_task_point[2], ground_truth_task_point[2]],
            color="green" if is_covered else "red",
            linewidth=0.8,
            alpha=0.8,
        )
        origin_axis.text(
            *sequence.ground_truth_pose[pose_index, :3],
            f" {region_number + 1}",
            fontsize=7,
            color="green" if is_covered else "red",
        )
        task_axis.text(
            *ground_truth_task_point,
            f" {region_number + 1}",
            fontsize=7,
            color="green" if is_covered else "red",
        )
        task_plot_points.append(projected_boundary)

    total_coverage = float(np.mean(covered))
    selected_coverage = int(np.sum(covered[selected_offsets]))
    figure.suptitle(
        f"{sequence.name} | {alignment_mode.replace('_', ' ')} alignment | "
        f"test coverage={total_coverage:.3f}, q={model.quantile:.2f}, "
        f"selected covered={selected_coverage}/{len(selected_indices)}"
    )
    origin_axis.set_title("Pose-origin trajectories")
    task_axis.set_title(
        f"Projected CP regions for body point {body_point.tolist()}"
    )
    for axis in (origin_axis, task_axis):
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.plot(
            [],
            [],
            [],
            color="green",
            marker="*",
            linestyle="",
            label="Joint SE(3) covered",
        )
        axis.plot(
            [],
            [],
            [],
            color="red",
            marker="*",
            linestyle="",
            label="Joint SE(3) missed",
        )
    task_axis.plot(
        [],
        [],
        [],
        color="tab:blue",
        linewidth=5,
        alpha=0.2,
        label="CP task-space surfaces",
    )
    origin_axis.legend(loc="upper left", fontsize=8)
    task_axis.legend(loc="upper left", fontsize=8)
    _equal_3d_axes(origin_axis, np.concatenate(origin_plot_points, axis=0))
    _equal_3d_axes(task_axis, np.concatenate(task_plot_points, axis=0))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def parse_body_point(value: str) -> np.ndarray:
    coordinates = np.asarray([float(part) for part in value.split(",")])
    if coordinates.shape != (3,):
        raise argparse.ArgumentTypeError("Body point must have form x,y,z")
    return coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/VIO"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/alignment_comparison/results.csv"),
    )
    parser.add_argument("--alignment-fraction", type=float, default=0.20)
    parser.add_argument("--covariance-frame", choices=("world", "body"), default="world")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--visualize-sequence",
        default="V2_02_medium_merged",
        help="Merged CSV stem to visualize; use 'none' to disable.",
    )
    parser.add_argument("--region-count", type=int, default=7)
    parser.add_argument("--region-samples", type=int, default=600)
    parser.add_argument("--body-point", type=parse_body_point, default=np.array([1.0, 0.0, 0.0]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = sorted(args.data_dir.glob("*_merged.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No merged CSV files found under {args.data_dir}")
    print("Alignment comparison")
    print("initial_block: Horn alignment on the first trajectory block, then frozen.")
    print("full_trajectory: Horn alignment on all poses, used only as benchmark alignment.")
    print("Both modes use identical CP calibration and test indices.")
    rows = []
    for csv_path in csv_files:
        for alignment_mode in ALIGNMENT_MODES:
            row = evaluate_file(
                csv_path,
                alignment_mode,
                args.alignment_fraction,
                args.covariance_frame,
                args.alpha,
            )
            rows.append(row)
            print(
                f"{row['sequence']:28s} {alignment_mode:15s} "
                f"coverage={row['test_6d_coverage']:.3f} "
                f"q={row['quantile']:.2f} "
                f"test_rmse={row['test_position_rmse_m']:.3f} m "
                f"status={row['trajectory_status']}"
            )
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = (
        results.groupby("alignment_mode")[
            [
                "test_6d_coverage",
                "quantile",
                "all_position_rmse_m",
                "test_position_rmse_m",
                "test_position_max_m",
                "test_rotation_rmse_deg",
            ]
        ]
        .agg(["mean", "median", "min", "max"])
    )
    summary_path = args.output.with_name("summary.csv")
    summary.to_csv(summary_path)
    nominal_results = results[results["trajectory_status"] == "nominal"]
    nominal_summary = (
        nominal_results.groupby("alignment_mode")[
            [
                "test_6d_coverage",
                "quantile",
                "all_position_rmse_m",
                "test_position_rmse_m",
                "test_position_max_m",
                "test_rotation_rmse_deg",
            ]
        ]
        .agg(["mean", "median", "min", "max"])
    )
    nominal_summary_path = args.output.with_name("summary_nominal.csv")
    nominal_summary.to_csv(nominal_summary_path)
    if args.visualize_sequence.lower() != "none":
        visualization_csv = args.data_dir / f"{args.visualize_sequence}.csv"
        if not visualization_csv.exists():
            raise FileNotFoundError(
                f"Visualization sequence not found: {visualization_csv}"
            )
        for alignment_mode in ALIGNMENT_MODES:
            visualization_path = args.output.parent / (
                f"{args.visualize_sequence}_{alignment_mode}_trajectory_regions.png"
            )
            plot_test_trajectory_regions(
                visualization_csv,
                alignment_mode,
                args.alignment_fraction,
                args.covariance_frame,
                args.alpha,
                args.body_point,
                args.region_count,
                args.region_samples,
                visualization_path,
            )
            print(f"Saved trajectory visualization: {visualization_path}")
    print()
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Saved per-sequence results: {args.output}")
    print(f"Saved alignment summary:    {summary_path}")
    print(f"Saved nominal-only summary: {nominal_summary_path}")


if __name__ == "__main__":
    main()
