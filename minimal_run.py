
import warnings
warnings.filterwarnings("ignore", "torch.set_default_tensor_type.*", UserWarning)

import time
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from helpers import _l2_norm, estimate_invariant_error_covariance, estimate_mean_and_covariance
from pymatlie.se3 import SE3
from utils.algorithms import NonconformityScore, Q_Space
from utils.conformal_prediction.base import (
    splitCP,
    critical_mahalanobis_distance,
)

SE3_DOF = 6

methods = {
    # Mahalanobis + conformal scaling of per-sample VIO covariance
    "CLAPS": {
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "calibration_strategy": "scaling",
        "name": "CLAPS-VIO",
    },
    # L2 ball radius calibrated via conformal prediction (ignores VIO covariance shape)
    "BASELINE2": {
        "calibrate": True,
        "r_metric": NonconformityScore.L2,
        "calibration_strategy": "scaling",
        "name": "L2 + CP",
    },
    # Raw VIO covariance, no calibration (γ = 1)
    "BASELINE4": {
        "calibrate": False,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "calibration_strategy": "scaling",
        "name": "VIO raw cov",
    },
    # Replace per-sample VIO cov with empirical covariance from calibration set
    "BASELINE6": {
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "calibration_strategy": "covariance_fit",
        "name": "Empirical cov",
    },
    # Empirical covariance + bias correction
    "BASELINE7": {
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "calibration_strategy": "mean_covariance_fit",
        "name": "Empirical cov + bias",
    },
    # Empirical covariance shape + conformal scaling (CP applied on top of emp. cov)
    "BASELINE8": {
        "calibrate": True,
        "r_metric": NonconformityScore.MAHALANOBIS,
        "calibration_strategy": "covariance_fit_cp",
        "name": "Empirical cov + CP",
    },
}


def get_experiment_dir(robot_name: str, confidence_level: str = "default") -> Path:
    return Path(f"data/{robot_name}/experiments/{confidence_level}")


def ensure_dir_structure(base_path: Path, method: str = None) -> dict:
    paths = {
        "base": base_path,
        "calibration": base_path / "calibration",
        "validation": base_path / "validation" / "metrics",
    }
    if method:
        method_base = base_path / method
        paths = {
            "base": method_base,
            "calibration": method_base / "calibration",
            "validation": method_base / "validation" / "metrics",
        }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


@contextmanager
def record_time(metrics: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.setdefault(key, []).append(time.perf_counter() - start)


def regularize_covariance(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Make (N,D,D) or (D,D) covariance symmetric positive-definite."""
    if cov.ndim == 2:
        cov = 0.5 * (cov + cov.T)
        min_eig = torch.linalg.eigvalsh(cov).min()
        if min_eig < eps:
            cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * (eps - min_eig + eps)
        return cov

    cov = 0.5 * (cov + cov.transpose(-1, -2))
    min_eigs = torch.linalg.eigvalsh(cov).min(dim=-1).values
    if (min_eigs < eps).any():
        D = cov.shape[-1]
        eye = torch.eye(D, dtype=cov.dtype, device=cov.device).unsqueeze(0)
        shifts = torch.clamp(eps - min_eigs, min=0.0) + eps
        cov = cov + shifts[:, None, None] * eye
    return cov


def compute_se3_error(
    vio_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    error_form: str,
) -> torch.Tensor:
    """SE3 Lie-algebra error between VIO estimate and ground truth. Returns (N, 6)."""
    g_vio = SE3.map_q_to_configuration(vio_pose)
    g_gt  = SE3.map_q_to_configuration(gt_pose)
    if error_form == "RI":
        return SE3.log(SE3.right_invariant_error(estimated_state=g_vio, true_state=g_gt))
    elif error_form == "LI":
        return SE3.log(SE3.left_invariant_error(estimated_state=g_vio, true_state=g_gt))
    else:
        raise ValueError(f"Unknown error_form: {error_form!r}")

def per_sample_mahalanobis_sq(err: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    if cov.ndim == 2:
        cov = cov.unsqueeze(0).expand(err.shape[0], -1, -1)
    cov_inv = torch.linalg.inv(cov)
    return torch.einsum("ni,nij,nj->n", err, cov_inv, err)

def process_single_validation_file(validation_file_path, shared_params):
    p = Path(validation_file_path)

    method_cfg = shared_params["method_cfg"]
    method_paths = shared_params["method_paths"]
    loaded_scaling_factor = shared_params["loaded_scaling_factor"]
    loaded_covariance = shared_params.get("loaded_covariance")
    loaded_bias = shared_params.get("loaded_bias")
    params = shared_params["params"]
    calibration_strategy = shared_params.get("calibration_strategy", "scaling")
    method_name = shared_params["method_name"]

    val_time_metrics = {}
    data = torch.load(p, map_location=torch.get_default_device(), weights_only=False)

    vio_pose = data["vio_pose"]  
    gt_pose  = data["gt_pose"] 
    vio_cov  = regularize_covariance(data["vio_cov"])

    with record_time(val_time_metrics, "error_total"):
        err = compute_se3_error(vio_pose, gt_pose, params["error_form"])

    if loaded_bias is not None:
        err = err - loaded_bias.to(err.device, dtype=err.dtype)

    chi2_crit_sq = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF) ** 2

    with record_time(val_time_metrics, "calibration_total"):
        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            if calibration_strategy == "scaling":
                gamma = loaded_scaling_factor if loaded_scaling_factor is not None else 1.0
                xi_sigma = vio_cov * gamma  
            elif calibration_strategy == "covariance_fit_cp":
                gamma    = loaded_scaling_factor if loaded_scaling_factor is not None else 1.0
                xi_sigma = loaded_covariance.to(err.device, dtype=err.dtype) * gamma
                if xi_sigma.ndim == 2:
                    xi_sigma = xi_sigma.unsqueeze(0).expand(err.shape[0], -1, -1)
            elif calibration_strategy in ("covariance_fit", "mean_covariance_fit"):
                xi_sigma = loaded_covariance.to(err.device, dtype=err.dtype)
                if xi_sigma.ndim == 2:
                    xi_sigma = xi_sigma.unsqueeze(0).expand(err.shape[0], -1, -1)

            md_sq = per_sample_mahalanobis_sq(err, xi_sigma) 
            inside = md_sq <= chi2_crit_sq
            empirical_coverage = inside.float().mean().item()

        elif method_cfg["r_metric"] == NonconformityScore.L2:
            q_hat = loaded_scaling_factor if loaded_scaling_factor is not None else 0.0
            distances = _l2_norm(err, dim=-1)
            inside = distances <= q_hat
            empirical_coverage = inside.float().mean().item()

    metrics = {
        "file": p.name,
        "empirical_coverage_LA": empirical_coverage,
        "err": err.detach().cpu().numpy(),
        "mask_lie_algebra": inside.cpu().numpy(),
        "vio_pose": vio_pose[0].detach().cpu(),
        "gt_pose": gt_pose[0].detach().cpu(),
    }
    metrics.update(val_time_metrics)

    metrics_path = method_paths["validation"] / f"{p.stem}_{method_name}.pt"
    torch.save(metrics, metrics_path)
    print(f" {p.name} coverage={empirical_coverage:.4f} of {metrics_path.name}")
    return metrics


if __name__ == "__main__":

    robot_name = "EuRoC_Drone"
    method = "CLAPS" 
    confidence_level = "default"
    run_calibration = True
    run_validation = True
    limit = None 

    systems_config = yaml.load(open("systems.yaml"), Loader=yaml.FullLoader)
    params = systems_config[robot_name]

    method_cfg = methods[method]
    calibration_strategy = method_cfg.get("calibration_strategy", "scaling")

    experiment_dir = get_experiment_dir(robot_name, confidence_level)
    method_paths = ensure_dir_structure(experiment_dir, method)

    scaling_factor = None
    bias = None
    loaded_covariance = None

    if run_calibration and method_cfg["calibrate"]:
        print("=" * 60)
        print(f"Calibration  [{method_cfg['name']}]")
        print("=" * 60)

        cal_files = sorted(Path("data/vio/calibration").glob("*.pt"))
        if not cal_files:
            raise FileNotFoundError("No calibration .pt files found. Run data_creator.py first.")

        all_vio_pose, all_gt_pose, all_vio_cov = [], [], []
        for file in tqdm(cal_files, desc="Loading calibration data"):
            loaded = torch.load(file, map_location=torch.get_default_device(), weights_only=False)
            all_vio_pose.append(loaded["vio_pose"])
            all_gt_pose.append(loaded["gt_pose"])
            all_vio_cov.append(loaded["vio_cov"])

        vio_pose = torch.cat(all_vio_pose, dim=0)
        gt_pose  = torch.cat(all_gt_pose,  dim=0)
        vio_cov  = regularize_covariance(torch.cat(all_vio_cov, dim=0))

        print(f"Calibration set: {vio_pose.shape[0]} samples")

        err = compute_se3_error(vio_pose, gt_pose, params["error_form"]) 

        if calibration_strategy == "scaling":
            if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
                md_sq = per_sample_mahalanobis_sq(err, vio_cov) 
                scores = md_sq.sqrt()
            else:
                scores = _l2_norm(err, dim=-1)

            q_hat = splitCP(scores=scores, alpha=params["failure_rate"])

            if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
                chi2_crit = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF)
                scaling_factor = (q_hat ** 2 / chi2_crit ** 2).item()
            else:
                scaling_factor = q_hat.item()

        elif calibration_strategy == "covariance_fit":
            loaded_covariance = estimate_invariant_error_covariance(err)

        elif calibration_strategy == "mean_covariance_fit":
            bias, loaded_covariance = estimate_mean_and_covariance(err)

        elif calibration_strategy == "covariance_fit_cp":
            loaded_covariance = estimate_invariant_error_covariance(err)
            sigma_exp = loaded_covariance.unsqueeze(0).expand(err.shape[0], -1, -1)
            scores = per_sample_mahalanobis_sq(err, sigma_exp).sqrt()
            q_hat = splitCP(scores=scores, alpha=params["failure_rate"])
            chi2_crit = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF)
            scaling_factor = (q_hat ** 2 / chi2_crit ** 2).item()

        chi2_crit_sq = critical_mahalanobis_distance(params["failure_rate"], D=SE3_DOF) ** 2
        if method_cfg["r_metric"] == NonconformityScore.MAHALANOBIS:
            if calibration_strategy == "scaling":
                xi_cal = vio_cov * scaling_factor
            elif calibration_strategy == "covariance_fit_cp":
                xi_cal = loaded_covariance.unsqueeze(0).expand(err.shape[0], -1, -1) * scaling_factor
            else:
                xi_cal = loaded_covariance.unsqueeze(0).expand(err.shape[0], -1, -1)
            err_adj = err if bias is None else err - bias.to(err.device)
            inside  = per_sample_mahalanobis_sq(err_adj, xi_cal) <= chi2_crit_sq
        else:
            inside = _l2_norm(err, dim=-1) <= scaling_factor

        empirical_coverage = inside.float().mean().item()
        print(f"Calibration coverage: {empirical_coverage:.4f}  "
              f"(target ≥ {1 - params['failure_rate']:.2f})")
        if scaling_factor is not None:
            print(f"Scaling factor: {scaling_factor:.4f}")

        if calibration_strategy == "covariance_fit":
            path = method_paths["calibration"] / f"{method}_covariance.pt"
            torch.save({"covariance": loaded_covariance.cpu()}, path)
        elif calibration_strategy == "mean_covariance_fit":
            path = method_paths["calibration"] / f"{method}_mean_covariance.pt"
            torch.save({"bias": bias.cpu(), "covariance": loaded_covariance.cpu()}, path)
        elif calibration_strategy == "covariance_fit_cp":
            path = method_paths["calibration"] / f"{method}_covariance_cp.pt"
            torch.save({"covariance": loaded_covariance.cpu(), "scaling_factor": scaling_factor}, path)
        else:
            path = method_paths["calibration"] / f"{method}_scaling.pt"
            torch.save({"scaling_factor": scaling_factor}, path)
        print(f"Calibration artifact: {path}")

    if run_validation:
        print("=" * 60)
        print(f"Validation  [{method_cfg['name']}]")
        print("=" * 60)

        loaded_scaling_factor = None
        loaded_covariance = None
        loaded_bias = None

        if method_cfg["calibrate"]:
            if calibration_strategy == "scaling":
                payload = torch.load(
                    method_paths["calibration"] / f"{method}_scaling.pt",
                    map_location=torch.get_default_device(), weights_only=False,
                )
                loaded_scaling_factor = payload["scaling_factor"]
            elif calibration_strategy == "covariance_fit":
                payload = torch.load(
                    method_paths["calibration"] / f"{method}_covariance.pt",
                    map_location=torch.get_default_device(), weights_only=False,
                )
                loaded_covariance = payload["covariance"]
            elif calibration_strategy == "mean_covariance_fit":
                payload = torch.load(
                    method_paths["calibration"] / f"{method}_mean_covariance.pt",
                    map_location=torch.get_default_device(), weights_only=False,
                )
                loaded_bias = payload["bias"]
                loaded_covariance = payload["covariance"]
            elif calibration_strategy == "covariance_fit_cp":
                payload = torch.load(
                    method_paths["calibration"] / f"{method}_covariance_cp.pt",
                    map_location=torch.get_default_device(), weights_only=False,
                )
                loaded_covariance = payload["covariance"]
                loaded_scaling_factor = payload["scaling_factor"]

        shared_params = {
            "method_cfg": method_cfg,
            "method_paths": method_paths,
            "loaded_scaling_factor":loaded_scaling_factor,
            "loaded_covariance":loaded_covariance,
            "loaded_bias":loaded_bias,
            "params":params,
            "calibration_strategy": calibration_strategy,
            "method_name":method,
        }

        val_files = sorted(Path("data/vio/validation").glob("*.pt"))
        if limit is not None:
            val_files = val_files[:limit]
        if not val_files:
            raise FileNotFoundError("No validation .pt files found. Run data_creator.py first.")

        results = []
        for vf in tqdm(val_files, desc="Validation"):
            results.append(process_single_validation_file(str(vf), shared_params))

        if results:
            coverages = [r["empirical_coverage_LA"] for r in results]
            print("-" * 40)
            print(f"Validation coverage  avg={sum(coverages)/len(coverages):.4f}  "
                  f"min={min(coverages):.4f}  max={max(coverages):.4f}")
            print(f"Target coverage      ≥ {1 - params['failure_rate']:.2f}")
