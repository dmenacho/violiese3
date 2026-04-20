import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

def _pose_to_mat(pose_np: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose_np[3:]).as_matrix()
    T[:3, 3]  = pose_np[:3]
    return T


def _mat_to_pose(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])


def _upper_tri_to_mat(t11, t12, t13, t22, t23, t33) -> torch.Tensor:
    n = len(t11)
    M = torch.zeros(n, 3, 3, dtype=torch.float64)
    M[:, 0, 0] = torch.tensor(t11, dtype=torch.float64)
    M[:, 0, 1] = M[:, 1, 0] = torch.tensor(t12, dtype=torch.float64)
    M[:, 0, 2] = M[:, 2, 0] = torch.tensor(t13, dtype=torch.float64)
    M[:, 1, 1] = torch.tensor(t22, dtype=torch.float64)
    M[:, 1, 2] = M[:, 2, 1] = torch.tensor(t23, dtype=torch.float64)
    M[:, 2, 2] = torch.tensor(t33, dtype=torch.float64)
    return M


def _build_se3_cov(Pt: torch.Tensor, Pr: torch.Tensor) -> torch.Tensor:
    n = Pt.shape[0]
    cov = torch.zeros(n, 6, 6, dtype=torch.float64)
    cov[:, 0:3, 0:3] = Pt
    cov[:, 3:6, 3:6] = Pr
    return cov


def _align_vio_to_gt(vio_np: np.ndarray, gt_np: np.ndarray) -> np.ndarray:
    T_align = _pose_to_mat(gt_np[0]) @ np.linalg.inv(_pose_to_mat(vio_np[0]))
    return np.stack([_mat_to_pose(T_align @ _pose_to_mat(p)) for p in vio_np])


def _load_trajectory(csv_path: str, align: bool = True):
    df = pd.read_csv(csv_path)

    vio_np = df[["p_x", "p_y", "p_z", "q_x", "q_y", "q_z", "q_w"]].values
    gt_np  = df[["gt_p_x", "gt_p_y", "gt_p_z","gt_q_x", "gt_q_y", "gt_q_z", "gt_q_w"]].values

    if align:
        vio_np = _align_vio_to_gt(vio_np, gt_np)

    Pr = _upper_tri_to_mat(df["Pr11"], df["Pr12"], df["Pr13"], df["Pr22"], df["Pr23"], df["Pr33"])
    Pt = _upper_tri_to_mat(df["Pt11"], df["Pt12"], df["Pt13"], df["Pt22"], df["Pt23"], df["Pt33"])

    return (torch.tensor(vio_np, dtype=torch.float64), torch.tensor(gt_np,  dtype=torch.float64),_build_se3_cov(Pt, Pr), vio_np, gt_np)

def _save_chunks( vio_pose: torch.Tensor, gt_pose: torch.Tensor, vio_cov: torch.Tensor, out_dir: Path, traj_name: str, chunk_size: int) -> int:
    n = vio_pose.shape[0]
    n_chunks = n // chunk_size
    for i in range(n_chunks):
        s, e = i * chunk_size, (i + 1) * chunk_size
        torch.save( {"vio_pose": vio_pose[s:e], "gt_pose":  gt_pose[s:e], "vio_cov":  vio_cov[s:e]}, out_dir / f"{traj_name}_chunk_{i:03d}.pt")
    return n_chunks

def process_multiple_csvs( cal_csvs: list, val_csvs: list, chunk_size: int  = 60, output_dir: str  = "data/vio", align: bool = True,) -> None:

    cal_dir = Path(output_dir) / "calibration"
    val_dir = Path(output_dir) / "validation"
    cal_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_cal = total_val = 0
    for csv_path in cal_csvs:
        name = Path(csv_path).stem
        vio_pose, gt_pose, vio_cov, _, _ = _load_trajectory(csv_path, align)
        n_chunks = _save_chunks(vio_pose, gt_pose, vio_cov, cal_dir, name, chunk_size)
        err = (vio_pose[:, :3] - gt_pose[:, :3]).norm(dim=-1)
        print(f"CAL pos err  mean={err.mean():.3f} m  max={err.max():.3f} m")
        total_cal += n_chunks
    for csv_path in val_csvs:
        name = Path(csv_path).stem
        vio_pose, gt_pose, vio_cov, _, _ = _load_trajectory(csv_path, align)
        n_chunks = _save_chunks(vio_pose, gt_pose, vio_cov, val_dir, name, chunk_size)
        err = (vio_pose[:, :3] - gt_pose[:, :3]).norm(dim=-1)
        print(f"VAL pos err  mean={err.mean():.3f} m  max={err.max():.3f} m")
        total_val += n_chunks


def plot_trajectory(csv_path: str, align: bool = True) -> None:
    _, _, _, vio_np, gt_np = _load_trajectory(csv_path, align)

    n = len(vio_np)
    t = np.arange(n)
    pos_err  = vio_np[:, :3] - gt_np[:, :3]
    pos_norm = np.linalg.norm(pos_err, axis=1)

    rot_err_deg = np.degrees([Rotation.from_matrix(Rotation.from_quat(gt_np[i, 3:]).as_matrix().T @  Rotation.from_quat(vio_np[i, 3:]).as_matrix()).magnitude() for i in range(n)])
    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(*gt_np[:, :3].T,  color="tab:blue",   lw=1.5, label="Ground truth")
    ax3d.plot(*vio_np[:, :3].T, color="tab:orange", lw=1.5, label="VIO (aligned)")
    ax3d.scatter(*gt_np[0, :3],  color="tab:blue",   s=40, zorder=5)
    ax3d.scatter(*vio_np[0, :3], color="tab:orange", s=40, zorder=5)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D Trajectory"); ax3d.legend(fontsize=8)

    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xy.plot(gt_np[:, 0],  gt_np[:, 1],  color="tab:blue",   lw=1.5, label="GT")
    ax_xy.plot(vio_np[:, 0], vio_np[:, 1], color="tab:orange", lw=1.5, label="VIO")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-down (XY)")
    ax_xy.legend(fontsize=8); ax_xy.set_aspect("equal", "box"); ax_xy.grid(True)

    ax_pn = fig.add_subplot(2, 3, 3)
    ax_pn.plot(t, pos_norm, color="tab:red", lw=1)
    ax_pn.axhline(pos_norm.mean(), color="tab:red", ls="--", lw=1,
                  label=f"mean={pos_norm.mean():.3f} m")
    ax_pn.set_xlabel("Timestep"); ax_pn.set_ylabel("‖error‖ (m)")
    ax_pn.set_title("Position Error Norm")
    ax_pn.legend(fontsize=8); ax_pn.grid(True)

    ax_pa = fig.add_subplot(2, 3, 4)
    for i, (label, color) in enumerate([("X","tab:red"),("Y","tab:green"),("Z","tab:purple")]):
        ax_pa.plot(t, pos_err[:, i], color=color, lw=1, label=label)
    ax_pa.axhline(0, color="black", lw=0.5)
    ax_pa.set_xlabel("Timestep"); ax_pa.set_ylabel("Error (m)")
    ax_pa.set_title("Per-axis Position Error")
    ax_pa.legend(fontsize=8); ax_pa.grid(True)

    ax_r = fig.add_subplot(2, 3, 5)
    ax_r.plot(t, rot_err_deg, color="tab:brown", lw=1)
    ax_r.axhline(rot_err_deg.mean(), color="tab:brown", ls="--", lw=1,
                 label=f"mean={rot_err_deg.mean():.2f}°")
    ax_r.set_xlabel("Timestep"); ax_r.set_ylabel("Rotation error (°)")
    ax_r.set_title("Rotation Error"); ax_r.legend(fontsize=8); ax_r.grid(True)

    ax_h = fig.add_subplot(2, 3, 6)
    ax_h.hist(pos_norm, bins=30, color="tab:red", edgecolor="white", alpha=0.8)
    ax_h.axvline(pos_norm.mean(), color="black", ls="--", lw=1.5,
                 label=f"mean={pos_norm.mean():.3f} m")
    ax_h.set_xlabel("‖error‖ (m)"); ax_h.set_ylabel("Count")
    ax_h.set_title("Position Error Histogram")
    ax_h.legend(fontsize=8); ax_h.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CAL_CSVS = [
        "data/VIO/V2_02_medium_merged.csv",
        "data/VIO/MH_05_difficult_merged.csv",
        "data/VIO/V1_03_difficult_merged.csv",
    ]
    VAL_CSVS = [
        "data/VIO/V1_01_easy_merged.csv",
    ]

    parser = argparse.ArgumentParser(description="Preprocess VIO CSVs for CLAPS-VIO")
    parser.add_argument("--chunk_size", type=int,  default=60)
    parser.add_argument("--output_dir", default="data/vio")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--plot", default=None, metavar="CSV")
    args = parser.parse_args()

    if args.plot:
        plot_trajectory(args.plot, align=not args.no_align)
    else:
        process_multiple_csvs(cal_csvs=CAL_CSVS, val_csvs=VAL_CSVS, chunk_size=args.chunk_size, output_dir=args.output_dir, align=not args.no_align)
