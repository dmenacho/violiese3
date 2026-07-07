"""Dataset adapters for pose-only CSV directories and merged VIO CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from utils.conformal_prediction.se3 import se3_body_error


POSE_COLUMNS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


@dataclass(frozen=True)
class PoseSequence:
    name: str
    timestamps: np.ndarray
    estimated_pose: np.ndarray
    ground_truth_pose: np.ndarray
    errors: np.ndarray
    covariance: np.ndarray | None = None
    alignment_rotation: np.ndarray | None = None
    alignment_translation: np.ndarray | None = None


def _normalize_pose_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["timestamp", *POSE_COLUMNS]].dropna().sort_values("timestamp").copy()
    quaternion = frame[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=1)
    valid = norm > 1e-12
    frame = frame.loc[valid].copy()
    frame.loc[:, ["qx", "qy", "qz", "qw"]] = quaternion[valid] / norm[valid, None]
    return frame.reset_index(drop=True)


def _associate(
    estimate: pd.DataFrame,
    ground_truth: pd.DataFrame,
    max_time_difference: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate_time = estimate["timestamp"].to_numpy(dtype=np.float64)
    gt_time = ground_truth["timestamp"].to_numpy(dtype=np.float64)
    insertion = np.searchsorted(gt_time, estimate_time)
    right = np.clip(insertion, 0, len(gt_time) - 1)
    left = np.clip(insertion - 1, 0, len(gt_time) - 1)
    matched = np.where(
        np.abs(estimate_time - gt_time[left]) <= np.abs(estimate_time - gt_time[right]),
        left,
        right,
    )
    keep = np.abs(estimate_time - gt_time[matched]) <= max_time_difference
    return (
        estimate_time[keep],
        estimate.loc[keep, POSE_COLUMNS].to_numpy(dtype=np.float64),
        ground_truth.iloc[matched[keep]][POSE_COLUMNS].to_numpy(dtype=np.float64),
    )


def load_pose_pair(
    estimate_csv: Path,
    ground_truth_csv: Path,
    max_time_difference: float = 0.05,
) -> PoseSequence:
    estimate = _normalize_pose_frame(pd.read_csv(estimate_csv))
    ground_truth = _normalize_pose_frame(pd.read_csv(ground_truth_csv))
    timestamps, estimated_pose, ground_truth_pose = _associate(
        estimate, ground_truth, max_time_difference
    )
    return PoseSequence(
        name=estimate_csv.stem,
        timestamps=timestamps,
        estimated_pose=estimated_pose,
        ground_truth_pose=ground_truth_pose,
        errors=se3_body_error(estimated_pose, ground_truth_pose),
    )


def load_pose_directory(dataset_dir: Path) -> dict[str, PoseSequence]:
    sequences = {}
    for directory in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        gt_files = sorted(directory.glob("GT_*.csv"))
        estimate_files = sorted(
            path for path in directory.glob("*.csv") if not path.name.startswith("GT_")
        )
        if len(gt_files) != 1 or len(estimate_files) != 1:
            continue
        sequence = load_pose_pair(estimate_files[0], gt_files[0])
        sequences[directory.name] = PoseSequence(
            name=directory.name,
            timestamps=sequence.timestamps,
            estimated_pose=sequence.estimated_pose,
            ground_truth_pose=sequence.ground_truth_pose,
            errors=sequence.errors,
        )
    if not sequences:
        raise ValueError(f"No estimate/GT CSV pairs found under {dataset_dir}")
    return sequences


def load_aligned_openvins_pair(
    estimate_csv: Path,
    ground_truth_csv: Path,
    max_time_difference: float = 0.05,
    covariance_frame: str = "position_world_rotation_body",
) -> PoseSequence:
    """Load the aligned pose and covariance exported in datasets/OPEN_VINS."""
    frame = pd.read_csv(estimate_csv).sort_values("timestamp").reset_index(drop=True)
    if all(
        column in frame.columns
        for column in (
            "p_x_evo",
            "p_y_evo",
            "p_z_evo",
            "q_x_evo",
            "q_y_evo",
            "q_z_evo",
            "q_w_evo",
        )
    ):
        aligned_columns = [
            "p_x_evo",
            "p_y_evo",
            "p_z_evo",
            "q_x_evo",
            "q_y_evo",
            "q_z_evo",
            "q_w_evo",
        ]
        raw_columns = ["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]
    elif all(
        column in frame.columns
        for column in (
            "p_x_x",
            "p_y_x",
            "p_z_x",
            "q_x_x",
            "q_y_x",
            "q_z_x",
            "q_w_x",
            "p_x_y",
            "p_y_y",
            "p_z_y",
            "q_x_y",
            "q_y_y",
            "q_z_y",
            "q_w_y",
        )
    ):
        aligned_columns = [
            "p_x_x",
            "p_y_x",
            "p_z_x",
            "q_x_x",
            "q_y_x",
            "q_z_x",
            "q_w_x",
        ]
        raw_columns = [
            "p_x_y",
            "p_y_y",
            "p_z_y",
            "q_x_y",
            "q_y_y",
            "q_z_y",
            "q_w_y",
        ]
    else:
        raise ValueError(
            f"{estimate_csv} has neither the _evo nor legacy _x/_y aligned schema"
        )
    estimated_pose = frame[aligned_columns].to_numpy(dtype=np.float64)
    raw_pose = frame[raw_columns].to_numpy(dtype=np.float64)
    estimate = pd.DataFrame(
        np.column_stack([frame["timestamp"].to_numpy(), estimated_pose]),
        columns=["timestamp", *POSE_COLUMNS],
    )
    ground_truth = _normalize_pose_frame(pd.read_csv(ground_truth_csv))
    timestamps, associated_pose, ground_truth_pose = _associate(
        estimate,
        ground_truth,
        max_time_difference,
    )
    keep = np.isin(frame["timestamp"].to_numpy(dtype=np.float64), timestamps)
    covariance = np.zeros((len(frame), 6, 6), dtype=np.float64)
    covariance[:, :3, :3] = _symmetric_3x3(frame, "Pt")
    has_rotation_covariance = all(
        column in frame.columns
        for column in ("Pr11", "Pr12", "Pr13", "Pr22", "Pr23", "Pr33")
    )
    if has_rotation_covariance:
        covariance[:, 3:, 3:] = _symmetric_3x3(frame, "Pr")
    else:
        covariance[:, 3:, 3:] = np.nan
    covariance = _aligned_covariance_to_body_tangent(
        covariance[keep],
        raw_pose[keep],
        associated_pose,
        covariance_frame,
    )
    return PoseSequence(
        name=estimate_csv.stem,
        timestamps=timestamps,
        estimated_pose=associated_pose,
        ground_truth_pose=ground_truth_pose,
        errors=se3_body_error(associated_pose, ground_truth_pose),
        covariance=covariance,
    )


def _symmetric_3x3(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    matrix = np.zeros((len(frame), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = frame[f"{prefix}11"]
    matrix[:, 0, 1] = matrix[:, 1, 0] = frame[f"{prefix}12"]
    matrix[:, 0, 2] = matrix[:, 2, 0] = frame[f"{prefix}13"]
    matrix[:, 1, 1] = frame[f"{prefix}22"]
    matrix[:, 1, 2] = matrix[:, 2, 1] = frame[f"{prefix}23"]
    matrix[:, 2, 2] = frame[f"{prefix}33"]
    return matrix


def estimate_horn_alignment(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the metric rigid transform target ~= R @ source + t."""
    source_positions = np.asarray(source_positions, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    if source_positions.shape != target_positions.shape:
        raise ValueError("Source and target positions must have matching shapes")
    if source_positions.ndim != 2 or source_positions.shape[1] != 3:
        raise ValueError("Alignment positions must have shape (N, 3)")
    if len(source_positions) < 3:
        raise ValueError("Horn alignment requires at least three pose pairs")

    source_mean = source_positions.mean(axis=0)
    target_mean = target_positions.mean(axis=0)
    source_centered = source_positions - source_mean
    target_centered = target_positions - target_mean
    cross_covariance = source_centered.T @ target_centered
    left, _, right_transpose = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(right_transpose.T @ left.T)
    rotation = right_transpose.T @ correction @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def _alignment_transform(
    estimated_pose: np.ndarray,
    ground_truth_pose: np.ndarray,
    alignment_mode: str,
    alignment_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if alignment_mode == "none":
        return np.eye(3), np.zeros(3)
    if alignment_mode == "first_pose":
        estimated_rotation = Rotation.from_quat(estimated_pose[0, 3:]).as_matrix()
        ground_truth_rotation = Rotation.from_quat(ground_truth_pose[0, 3:]).as_matrix()
        rotation = ground_truth_rotation @ estimated_rotation.T
        translation = ground_truth_pose[0, :3] - rotation @ estimated_pose[0, :3]
        return rotation, translation
    if alignment_mode == "initial_block":
        if not 0.0 < alignment_fraction < 1.0:
            raise ValueError("alignment_fraction must lie in (0, 1)")
        alignment_count = max(3, int(len(estimated_pose) * alignment_fraction))
        indices = slice(0, alignment_count)
    elif alignment_mode == "full_trajectory":
        indices = slice(None)
    else:
        raise ValueError(
            "alignment_mode must be none, first_pose, initial_block, or full_trajectory"
        )
    return estimate_horn_alignment(
        estimated_pose[indices, :3],
        ground_truth_pose[indices, :3],
    )


def _align_estimate_and_covariance(
    estimated_pose: np.ndarray,
    ground_truth_pose: np.ndarray,
    covariance: np.ndarray,
    covariance_frame: str,
    alignment_mode: str,
    alignment_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    estimated_rotation = Rotation.from_quat(estimated_pose[:, 3:])
    rotation_matrix, alignment_translation = _alignment_transform(
        estimated_pose,
        ground_truth_pose,
        alignment_mode,
        alignment_fraction,
    )
    alignment_rotation = Rotation.from_matrix(rotation_matrix)
    aligned_pose = estimated_pose.copy()
    aligned_pose[:, :3] = (
        alignment_rotation.apply(estimated_pose[:, :3]) + alignment_translation
    )
    aligned_rotation = alignment_rotation * estimated_rotation
    aligned_pose[:, 3:] = aligned_rotation.as_quat()

    if covariance_frame == "world":
        adjoint_rotation = np.zeros((6, 6), dtype=np.float64)
        adjoint_rotation[:3, :3] = rotation_matrix
        adjoint_rotation[3:, 3:] = rotation_matrix
        covariance = (
            adjoint_rotation[None, :, :]
            @ covariance
            @ adjoint_rotation.T[None, :, :]
        )
    elif covariance_frame != "body":
        raise ValueError("covariance_frame must be 'world' or 'body'")
    return aligned_pose, covariance, rotation_matrix, alignment_translation


def _world_covariance_to_body(
    covariance: np.ndarray,
    estimated_pose: np.ndarray,
) -> np.ndarray:
    rotation = Rotation.from_quat(estimated_pose[:, 3:]).as_matrix()
    transform = np.zeros((len(estimated_pose), 6, 6), dtype=np.float64)
    transform[:, :3, :3] = np.swapaxes(rotation, 1, 2)
    transform[:, 3:, 3:] = np.swapaxes(rotation, 1, 2)
    return transform @ covariance @ np.swapaxes(transform, 1, 2)


def _covariance_to_body_tangent(
    covariance: np.ndarray,
    estimated_pose: np.ndarray,
    covariance_frame: str,
) -> np.ndarray:
    if covariance_frame == "body":
        return covariance
    rotation = Rotation.from_quat(estimated_pose[:, 3:]).as_matrix()
    transform = np.zeros((len(estimated_pose), 6, 6), dtype=np.float64)
    if covariance_frame == "world":
        transform[:, :3, :3] = np.swapaxes(rotation, 1, 2)
        transform[:, 3:, 3:] = np.swapaxes(rotation, 1, 2)
    elif covariance_frame == "position_world_rotation_body":
        transform[:, :3, :3] = np.swapaxes(rotation, 1, 2)
        transform[:, 3:, 3:] = np.eye(3)
    elif covariance_frame == "position_body_rotation_world":
        transform[:, :3, :3] = np.eye(3)
        transform[:, 3:, 3:] = np.swapaxes(rotation, 1, 2)
    else:
        raise ValueError(
            "covariance_frame must be 'body', 'world', "
            "'position_world_rotation_body', or 'position_body_rotation_world'"
        )
    return transform @ covariance @ np.swapaxes(transform, 1, 2)


def _aligned_covariance_to_body_tangent(
    covariance: np.ndarray,
    raw_pose: np.ndarray,
    aligned_pose: np.ndarray,
    covariance_frame: str,
) -> np.ndarray:
    """Express raw OpenVINS covariance in the aligned pose's body tangent."""
    if covariance_frame == "body":
        return covariance

    alignment_rotation, _ = estimate_horn_alignment(
        raw_pose[:, :3],
        aligned_pose[:, :3],
    )
    aligned_rotation = Rotation.from_quat(aligned_pose[:, 3:]).as_matrix()
    world_to_body = np.swapaxes(aligned_rotation, 1, 2)
    raw_world_to_aligned_body = np.einsum(
        "nij,jk->nik", world_to_body, alignment_rotation
    )
    transform = np.zeros((len(covariance), 6, 6), dtype=np.float64)
    if covariance_frame == "world":
        transform[:, :3, :3] = raw_world_to_aligned_body
        transform[:, 3:, 3:] = raw_world_to_aligned_body
    elif covariance_frame == "position_world_rotation_body":
        transform[:, :3, :3] = raw_world_to_aligned_body
        transform[:, 3:, 3:] = np.eye(3)
    elif covariance_frame == "position_body_rotation_world":
        transform[:, :3, :3] = np.eye(3)
        transform[:, 3:, 3:] = raw_world_to_aligned_body
    else:
        raise ValueError(
            "covariance_frame must be 'body', 'world', "
            "'position_world_rotation_body', or 'position_body_rotation_world'"
        )
    return transform @ covariance @ np.swapaxes(transform, 1, 2)


def load_merged_vio_csv(
    csv_path: Path,
    alignment_mode: str = "initial_block",
    alignment_fraction: float = 0.20,
    covariance_frame: str = "world",
) -> PoseSequence:
    frame = pd.read_csv(csv_path)
    estimated_pose = frame[
        ["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]
    ].to_numpy(dtype=np.float64)
    ground_truth_pose = frame[
        ["gt_p_x", "gt_p_y", "gt_p_z", "gt_q_x", "gt_q_y", "gt_q_z", "gt_q_w"]
    ].to_numpy(dtype=np.float64)
    covariance = np.zeros((len(frame), 6, 6), dtype=np.float64)
    covariance[:, :3, :3] = _symmetric_3x3(frame, "Pt")
    covariance[:, 3:, 3:] = _symmetric_3x3(frame, "Pr")
    alignment_rotation = np.eye(3)
    alignment_translation = np.zeros(3)
    if alignment_mode != "none":
        (
            estimated_pose,
            covariance,
            alignment_rotation,
            alignment_translation,
        ) = _align_estimate_and_covariance(
            estimated_pose,
            ground_truth_pose,
            covariance,
            covariance_frame,
            alignment_mode,
            alignment_fraction,
        )
    if covariance_frame == "world":
        covariance = _world_covariance_to_body(covariance, estimated_pose)
    return PoseSequence(
        name=csv_path.stem,
        timestamps=frame["timestamp"].to_numpy(dtype=np.float64),
        estimated_pose=estimated_pose,
        ground_truth_pose=ground_truth_pose,
        errors=se3_body_error(estimated_pose, ground_truth_pose),
        covariance=covariance,
        alignment_rotation=alignment_rotation,
        alignment_translation=alignment_translation,
    )
