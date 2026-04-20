import warnings
warnings.filterwarnings("ignore", "torch.set_default_tensor_type.*", UserWarning)

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
import yaml

from data_creator import _load_trajectory
from pymatlie.se3 import SE3
from minimal_run import (
    SE3_DOF,
    ensure_dir_structure,
    get_experiment_dir,
    methods,
    regularize_covariance,
)
from utils.algorithms import NonconformityScore
from utils.conformal_prediction.base import critical_mahalanobis_distance

VAL_CSV = "data/VIO/V1_01_easy_merged.csv"
ROBOT_NAME = "EuRoC_Drone"
CONF_LEVEL = "default"

params = yaml.safe_load(open("systems.yaml"))[ROBOT_NAME]
CHI2_6D = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF) ** 2

METHOD_COLORS = {
    "CLAPS": "blue",
    "BASELINE2": "green",
    "BASELINE4": "red",
    "BASELINE6": "purple",
    "BASELINE7": "brown",
    "BASELINE8": "cyan",
}

def _load_artifact(method_name: str, method_cfg: dict):
    if not method_cfg["calibrate"]:
        return 1.0, None, None
    strategy = method_cfg["calibration_strategy"]
    cal_dir = ensure_dir_structure(get_experiment_dir(ROBOT_NAME, CONF_LEVEL), method_name)["calibration"]
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


def _get_cov3_at(vio_cov, method_cfg, gamma, global_cov, idx) -> np.ndarray:
    if method_cfg["r_metric"] != NonconformityScore.MAHALANOBIS:
        return None
    strategy = method_cfg["calibration_strategy"]
    if strategy == "covariance_fit_cp":
        g = gamma if gamma is not None else 1.0
        cov6 = (global_cov * g).numpy()
    elif strategy in ("covariance_fit", "mean_covariance_fit"):
        cov6 = global_cov.numpy()
    else:
        g = gamma if gamma is not None else 1.0
        cov6 = (vio_cov[idx] * g).numpy()
    return cov6[0:3, 0:3]


def _draw_ellipse_2d(ax, center_2d, cov_2x2, chi2_val, color, label=None, **kwargs):
    vals, vecs = np.linalg.eigh(cov_2x2)
    vals  = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    e = Ellipse( xy = center_2d,width  = 2.0 * np.sqrt(chi2_val * vals[-1]),height = 2.0 * np.sqrt(chi2_val * vals[0]), angle  = angle, color  = color, label  = label, **kwargs)
    ax.add_patch(e)
    return e

def plot_geometry_comparison(val_csv: str, methods_to_plot: list,timestep: int  = 250, align: bool = True,) -> None:

    vio_pose, _, vio_cov, vio_np, gt_np = _load_trajectory(val_csv, align)
    vio_cov = regularize_covariance(vio_cov)
    idx       = min(timestep, len(vio_np) - 1)
    vio_point = vio_np[idx, :3]
    gt_point  = gt_np[idx, :3]

    planes = [
        ((0, 1), "X (m)", "Y (m)", "XY — top view"),
        ((0, 2), "X (m)", "Z (m)", "XZ — side view"),
        ((1, 2), "Y (m)", "Z (m)", "YZ — front view"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [methods[m]["name"] for m in methods_to_plot]
    fig.suptitle(
        f"Confidence ellipse geometry at timestep {idx} — {Path(val_csv).stem}\n",
        fontsize=10,
    )

    for method_name in methods_to_plot:
        method_cfg = methods[method_name]
        gamma, _, global_cov = _load_artifact(method_name, method_cfg)
        color = METHOD_COLORS.get(method_name, "gray")
        label  = method_cfg["name"]

        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            cov3 = _get_cov3_at(vio_cov, method_cfg, gamma, global_cov, idx)
            for col, ((i, j), xl, yl, title) in enumerate(planes):
                cov2 = cov3[np.ix_([i, j], [i, j])]
                _draw_ellipse_2d(axes[col], vio_point[[i, j]], cov2, CHI2_6D, color, label = f"{label}", alpha = 0.15, edgecolor = color, linewidth = 2.0)
        else: 
            q_hat = gamma if gamma is not None else 0.0
            for col, ((i, j), xl, yl, title) in enumerate(planes):
                axes[col].add_patch(plt.Circle( vio_point[[i, j]], q_hat, color=color, alpha=0.12, linewidth=2.0, label=f"{label}"))

    for col, ((i, j), xl, yl, title) in enumerate(planes):
        ax = axes[col]
        ax.plot(*vio_point[[i, j]], "ko",  ms=5,  zorder=10, label="VIO")
        ax.plot(*gt_point[[i, j]],  "r*",  ms=5, zorder=10, label="GT")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", "datalim")
        ax.margins(0.3)
        ax.grid(True, alpha=0.3)
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(), fontsize=5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    TIMESTEP = 300 
    plot_geometry_comparison( VAL_CSV, methods_to_plot=["BASELINE2", "BASELINE6", "BASELINE7", "BASELINE8"], timestep=TIMESTEP)
