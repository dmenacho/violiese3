from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import torch
import yaml

from data_creator import _load_trajectory
from helpers import _l2_norm
from pymatlie.se3 import SE3
from minimal_run import (
    SE3_DOF,
    compute_se3_error,
    ensure_dir_structure,
    get_experiment_dir,
    methods,
    per_sample_mahalanobis_sq,
    regularize_covariance,
)
from utils.algorithms import NonconformityScore
from utils.conformal_prediction.base import critical_mahalanobis_distance

VAL_CSV = "data/VIO/V1_01_easy_merged.csv"
ROBOT_NAME = "EuRoC_Drone"
CONF_LEVEL = "default"
METHODS_TO_PLOT  = ["BASELINE4", "BASELINE2", "BASELINE6", "BASELINE8"]

params = yaml.safe_load(open("systems.yaml"))[ROBOT_NAME]
CHI2_3D = critical_mahalanobis_distance(params["failure_rate"], D=3) ** 2
CHI2_6D = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF) ** 2

def _load_artifact(method_name: str, method_cfg: dict):
    if not method_cfg["calibrate"]:
        return 1.0, None, None

    strategy = method_cfg["calibration_strategy"]
    cal_dir  = ensure_dir_structure(
        get_experiment_dir(ROBOT_NAME, CONF_LEVEL), method_name
    )["calibration"]

    if strategy == "scaling":
        p = torch.load(cal_dir / f"{method_name}_scaling.pt", weights_only=False)
        return p["scaling_factor"], None, None
    elif strategy == "covariance_fit":
        p = torch.load(cal_dir / f"{method_name}_covariance.pt", weights_only=False)
        return None, None, p["covariance"].double()
    elif strategy == "mean_covariance_fit":
        p = torch.load(cal_dir / f"{method_name}_mean_covariance.pt", weights_only=False)
        return None, p["bias"].double(), p["covariance"].double()
    elif strategy == "covariance_fit_cp":
        p = torch.load(cal_dir / f"{method_name}_covariance_cp.pt", weights_only=False)
        return p["scaling_factor"], None, p["covariance"].double()


def _coverage_and_bands(vio_pose, gt_pose, vio_cov, method_cfg, gamma, bias, global_cov):

    err = compute_se3_error(vio_pose, gt_pose, params["error_form"])  # (N, 6)

    if bias is not None:
        err = err - bias.to(err.device, dtype=err.dtype)

    strategy = method_cfg["calibration_strategy"]

    if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
        if strategy == "covariance_fit_cp":
            g = gamma if gamma is not None else 1.0
            xi_sigma = (global_cov * g).unsqueeze(0).expand(err.shape[0], -1, -1)
        elif strategy in ("covariance_fit", "mean_covariance_fit"):
            xi_sigma = global_cov.unsqueeze(0).expand(err.shape[0], -1, -1)
        else:
            xi_sigma = vio_cov * (gamma if gamma is not None else 1.0) 

        inside_6d = (per_sample_mahalanobis_sq(err, xi_sigma) <= CHI2_6D).numpy()
        pos_cov = xi_sigma[:, 0:3, 0:3]  
        inside_pos = (per_sample_mahalanobis_sq(err[:, 0:3], pos_cov) <= CHI2_3D).numpy()

        pos_diag = pos_cov[:, [0, 1, 2], [0, 1, 2]].detach().numpy() 
        band_half= np.sqrt(CHI2_3D * pos_diag)

    else: 
        q_hat = gamma if gamma is not None else 0.0
        inside_6d = (_l2_norm(err, dim=-1) <= q_hat).numpy()
        inside_pos = inside_6d  
        band_half = np.full((err.shape[0], 3), q_hat)

    return inside_pos, band_half, inside_6d.mean(), inside_pos.mean()


def plot_all_methods(val_csv: str, methods_to_plot: list, align: bool = True) -> None:
    vio_pose, gt_pose, vio_cov, vio_np, gt_np = _load_trajectory(val_csv, align)
    vio_cov = regularize_covariance(vio_cov)
    n = len(vio_np)
    t = np.arange(n)
    n_methods = len(methods_to_plot)
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    fig, axes = plt.subplots(3, n_methods, figsize=(5.5 * n_methods, 9), sharex=True)
    if n_methods == 1:
        axes = axes[:, None]

    fig.suptitle( f"VIO Uncertainty Difficult Env — {Path(val_csv).stem}\n", fontsize=11)

    for col, method_name in enumerate(methods_to_plot):
        method_cfg  = methods[method_name]
        gamma, bias, global_cov = _load_artifact(method_name, method_cfg)
        inside, band_half, cov_6d, cov_pos = _coverage_and_bands( vio_pose, gt_pose, vio_cov, method_cfg, gamma, bias, global_cov,)
        gamma_str = f"γ={gamma:.2f}" if isinstance(gamma, float) else "emp. cov"

        for row, ylabel in enumerate(axis_labels):
            ax = axes[row, col]
            j  = row
            ax.fill_between(t, vio_np[:, j] - band_half[:, j], vio_np[:, j] + band_half[:, j], alpha=0.25, color="orange", zorder=1)
            ax.plot(t, vio_np[:, j], color="orange", lw=1.4, zorder=2)

            for i in range(n - 1):
                c = "green" if inside[i] else "red"
                ax.plot(t[i:i+2], gt_np[i:i+2, j], color=c, lw=1.4, zorder=3)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3, lw=0.5)
            if row == 0:
                ax.set_title(f"{method_cfg['name']}   {gamma_str}\n"  f"6D cov={cov_6d:.3f}", fontsize=9)
            if row == 2:
                ax.set_xlabel("Timestep", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], color="orange", lw=1.5, label="VIO estimate"),
        mpatches.Patch(color="orange", alpha=0.3, label="Position confidence band (3D)"),
        plt.Line2D([0], [0], color="green", lw=1.5, label="GT inside position band"),
        plt.Line2D([0], [0], color="red", lw=1.5, label="GT outside position band"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

if __name__ == "__main__":
    plot_all_methods(VAL_CSV, METHODS_TO_PLOT)
