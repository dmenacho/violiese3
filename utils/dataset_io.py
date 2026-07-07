"""Dataset adapters for aligned OpenVINS pose and covariance CSV files."""

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
