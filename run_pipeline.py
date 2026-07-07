"""Unified dataset -> SE(3) conformal calibration -> 3D region pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_io import PoseSequence, load_merged_vio_csv, load_pose_directory
from utils.conformal_prediction.se3 import (
    calibrate_empirical_model,
    calibrate_vio_covariance_model,
    covariance_at,
    mahalanobis_scores,
    project_pose_set_to_task_space,
    sample_tangent_ellipsoid,
)


def chronological_indices(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = int(0.4 * length)
    second = int(0.7 * length)
    return np.arange(first), np.arange(first, second), np.arange(second, length)


def plot_task_region(
    sequence: PoseSequence,
    model,
    pose_index: int,
    covariance_index: int | None,
    body_points: np.ndarray,
    output_path: Path,
) -> None:
    tangent_samples = sample_tangent_ellipsoid(
        model, covariance_index, sample_count=6000, seed=pose_index
    )
    projected = project_pose_set_to_task_space(
        sequence.estimated_pose[pose_index], tangent_samples, body_points
    )
    estimated_rotation = sequence.estimated_pose[pose_index, 3:]
    from scipy.spatial.transform import Rotation

    gt_task_points = (
        Rotation.from_quat(sequence.ground_truth_pose[pose_index, 3:]).apply(body_points)
        + sequence.ground_truth_pose[pose_index, :3]
    )
    estimated_task_points = (
        Rotation.from_quat(estimated_rotation).apply(body_points)
        + sequence.estimated_pose[pose_index, :3]
    )

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    for point_index in range(len(body_points)):
        points = projected[:, point_index]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=1,
            alpha=0.08,
            color=colors[point_index % len(colors)],
            label=f"Projected point {body_points[point_index].tolist()}",
        )
        axis.scatter(*gt_task_points[point_index], marker="*", s=90, color="red")
        axis.scatter(*estimated_task_points[point_index], marker="o", s=35, color="black")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.set_title(
        f"{sequence.name}: conformal SE(3) set projected to task space\n"
        f"covariance={model.covariance_source}, q={model.quantile:.3f}"
    )
    axis.legend(fontsize=7)
    axis.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        tangent_samples=tangent_samples,
        projected_points=projected,
        body_points=body_points,
        covariance=covariance_at(model, covariance_index),
        quantile=model.quantile,
    )


def run_merged(args: argparse.Namespace) -> None:
    sequence = load_merged_vio_csv(
        args.merged_csv,
        alignment_mode=args.alignment_mode,
        alignment_fraction=args.alignment_fraction,
        covariance_frame=args.covariance_frame,
    )
    _, calibration_indices, test_indices = chronological_indices(len(sequence.errors))
    model = calibrate_vio_covariance_model(
        sequence.errors[calibration_indices],
        sequence.covariance[calibration_indices],
        args.alpha,
    )
    test_scores = mahalanobis_scores(
        sequence.errors[test_indices],
        sequence.covariance[test_indices],
    )
    coverage = np.mean(test_scores <= model.quantile)
    pose_index = int(test_indices[min(args.pose_offset, len(test_indices) - 1)])
    # The model stores calibration covariance; use the matching test covariance
    # for the selected dynamic region.
    dynamic_model = type(model)(
        bias=model.bias,
        covariance=sequence.covariance[pose_index : pose_index + 1],
        quantile=model.quantile,
        alpha=model.alpha,
        covariance_source=model.covariance_source,
    )
    print("Pipeline mode: per-pose VIO covariance")
    print(
        f"Alignment: {args.alignment_mode}; "
        f"input covariance frame: {args.covariance_frame}"
    )
    print("Score: sqrt(xi^T Sigma_i^{-1} xi), xi=Log(T_est^{-1}T_gt)")
    if np.allclose(sequence.covariance[:, :3, 3:], 0.0):
        print(
            "Warning: covariance has zero translation-rotation cross terms; "
            "coupled SE(3) shape information is unavailable."
        )
    if model.quantile > 10.0:
        print(
            "Warning: the conformal radius is very large, indicating severe "
            "raw-covariance underestimation or distribution shift."
        )
    print(f"Test coverage: {coverage:.4f}; target: {1.0 - args.alpha:.4f}")
    plot_task_region(
        sequence,
        dynamic_model,
        pose_index,
        0,
        args.body_points,
        args.output,
    )


def run_pose_directory(args: argparse.Namespace) -> None:
    sequences = load_pose_directory(args.dataset_dir)
    names = sorted(sequences)
    if len(names) < 3:
        raise ValueError("Pose-only mode requires at least three sequences")
    fit_names, calibration_name, test_name = names[:-2], names[-2], names[-1]
    fit_errors = np.concatenate([sequences[name].errors for name in fit_names])
    calibration = sequences[calibration_name]
    test = sequences[test_name]
    model = calibrate_empirical_model(fit_errors, calibration.errors, args.alpha)
    test_scores = mahalanobis_scores(test.errors, model.covariance, model.bias)
    coverage = np.mean(test_scores <= model.quantile)
    pose_index = min(args.pose_offset, len(test.errors) - 1)
    print("Pipeline mode: pose-only empirical covariance fallback")
    print(f"Fit sequences: {', '.join(fit_names)}")
    print(f"Calibration sequence: {calibration_name}; test sequence: {test_name}")
    print("The shape is shared across poses because this dataset has no VIO covariance.")
    print(f"Test coverage: {coverage:.4f}; target: {1.0 - args.alpha:.4f}")
    plot_task_region(
        test,
        model,
        pose_index,
        None,
        args.body_points,
        args.output,
    )


def parse_body_points(values: list[str]) -> np.ndarray:
    points = []
    for value in values:
        coordinates = [float(part) for part in value.split(",")]
        if len(coordinates) != 3:
            raise argparse.ArgumentTypeError("Body points must have form x,y,z")
        points.append(coordinates)
    return np.asarray(points, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path)
    source.add_argument("--merged-csv", type=Path)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--pose-offset", type=int, default=50)
    parser.add_argument(
        "--covariance-frame",
        choices=("world", "body"),
        default="world",
        help="Frame used by covariance columns in merged VIO CSV files.",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("none", "first_pose", "initial_block", "full_trajectory"),
        default="initial_block",
        help="One fixed rigid alignment used for merged legacy trajectories.",
    )
    parser.add_argument(
        "--alignment-fraction",
        type=float,
        default=0.20,
        help="Initial trajectory fraction used by initial_block alignment.",
    )
    parser.add_argument(
        "--body-point",
        action="append",
        default=None,
        help="Body-attached point x,y,z. Repeat for multiple points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unified_pipeline/task_space_region.png"),
    )
    args = parser.parse_args()
    args.body_points = parse_body_points(args.body_point or ["0,0,0", "1,0,0"])
    return args


def main() -> None:
    args = parse_args()
    print("Unified pipeline")
    print("1. Load and synchronize poses without adding an alignment transform.")
    print("2. Compute the body-frame SE(3) residual once.")
    print("3. Calibrate a Mahalanobis radius using split conformal prediction.")
    print("4. Map tangent-space samples through Exp and the estimated pose.")
    print("5. Project body-attached points into 3D; nonzero points expose curvature.")
    if args.merged_csv:
        run_merged(args)
    else:
        run_pose_directory(args)
    print(f"Saved 3D region: {args.output}")
    print(f"Saved numerical samples: {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
