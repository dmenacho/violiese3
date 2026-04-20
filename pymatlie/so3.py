from dataclasses import dataclass, field
from typing import Tuple

import torch

from pymatlie.base_group import MatrixLieGroup
from pymatlie.vecops import sincu, versine_over_x

@dataclass(frozen=True)
class SO3(MatrixLieGroup):
    g_dim: int = 3
    matrix_size: tuple = (3, 3)
    B: torch.Tensor = torch.eye(3, dtype=torch.float64)
    u_dim: int = 3

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SO3.matrix_size, f"vee requires shape (N, {SO3.matrix_size}), got {tau_hat.shape}"
        # Tau = [tau_x, tau_y, tau_z]
        return torch.stack(
            [tau_hat[..., 2, 1], tau_hat[..., 0, 2], tau_hat[..., 1, 0]],
            dim=-1,
        )
    
    #tau=theta*u
    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SO3.g_dim, f"hat requires shape (N, {SO3.g_dim}), got {tau.shape}"
        out = torch.zeros((tau.shape[0], 3, 3), device=tau.device, dtype=tau.dtype)
        # SKEW - SYMMETRIC MATRIX
        out[..., 0, 1] = -tau[..., 2]
        out[..., 0, 2] = tau[..., 1]
        out[..., 1, 0] = tau[..., 2]
        out[..., 1, 2] = -tau[..., 0]
        out[..., 2, 0] = -tau[..., 1]
        out[..., 2, 1] = tau[..., 0]
        return out

    @staticmethod
    def _theta(tau: torch.Tensor) -> torch.Tensor:
        # theta = norm(tau)
        return torch.linalg.norm(tau, dim=-1)

    @staticmethod
    def _A(theta: torch.Tensor) -> torch.Tensor:
        # sin(theta)/theta
        return sincu(theta)
    
    @staticmethod
    def _B(theta: torch.Tensor) -> torch.Tensor:
        theta2 = theta * theta
        out = torch.empty_like(theta)
        small = theta.abs() < 1e-7
        # (1-cos(theta))/theta^2 = 1/2 - theta^2/24 + theta^4/720 - ...
        out[small] = 0.5 - theta2[small] / 24.0 + theta2[small] * theta2[small] / 720.0
        out[~small] = (1.0 - torch.cos(theta[~small])) / theta2[~small]
        return out
    
    @staticmethod
    def _C(theta: torch.Tensor) -> torch.Tensor:
        theta2 = theta * theta
        out = torch.empty_like(theta)
        small = theta.abs() < 1e-7
        # (theta-sin(theta))/theta^3 = 1/6 - theta^2/120 + theta^4/5040 - ...
        out[small] = 1.0 / 6.0 - theta2[small] / 120.0 + theta2[small] * theta2[small] / 5040.0
        out[~small] = (theta[~small] - torch.sin(theta[~small])) / (theta2[~small] * theta[~small])
        return out

    @staticmethod
    def _D(theta: torch.Tensor) -> torch.Tensor:
        theta2 = theta * theta
        out = torch.empty_like(theta)
        small = theta.abs() < 1e-7
        # 1/theta^2 * (1 - sinc(theta)/(2*B(theta))) = 1/12 + theta^2/720 + ...
        out[small] = 1.0 / 12.0 + theta2[small] / 720.0 + theta2[small] * theta2[small] / 30240.0
        A = SO3._A(theta[~small])
        B = SO3._B(theta[~small])
        # Equation 146
        out[~small] = (1.0 / theta2[~small]) * (1.0 - A / (2.0 * B))
        return out
    
    @staticmethod
    def exp(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SO3.g_dim, f"exp requires shape (N, {SO3.g_dim}), got {tau.shape}"
        theta = SO3._theta(tau)
        tau_hat = SO3.hat(tau)
        tau_hat_sq = tau_hat @ tau_hat
        I = torch.eye(3, device=tau.device, dtype=tau.dtype).expand(tau.shape[0], 3, 3)
        A = SO3._A(theta)[..., None, None]
        B = SO3._B(theta)[..., None, None]
        # R = Exp(theta*u) Equation 134
        return I + A * tau_hat + B * tau_hat_sq
    

    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SO3.matrix_size, f"log requires shape (N, {SO3.matrix_size}), got {g.shape}"
        # Trace (R) / g = R
        tr = g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2]
        cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
        theta = torch.arccos(cos_theta)
        # R - R^T
        vee_part = torch.stack(
            [
                g[..., 2, 1] - g[..., 1, 2],
                g[..., 0, 2] - g[..., 2, 0],
                g[..., 1, 0] - g[..., 0, 1],
            ],
            dim=-1,
        )

        out = torch.empty((g.shape[0], 3), device=g.device, dtype=g.dtype)
        small = theta.abs() < 1e-7
        near_pi = (~small) & ((torch.pi - theta).abs() < 1e-4)
        regular = (~small) & (~near_pi)

        #Unstable approxiamtion 1/2(R - R^T)
        out[small] = 0.5 * vee_part[small]

        #Stable case
        if regular.any():
            scale = theta[regular] / (2.0 * torch.sin(theta[regular]))
            # Equation 135
            out[regular] = scale.unsqueeze(-1) * vee_part[regular]

        #Close to pi since sin(pi)=0 and this provokes that the model increase exponential
        if near_pi.any():
            Rn = g[near_pi]
            thetan = theta[near_pi]
            phi = torch.zeros((Rn.shape[0], 3), device=g.device, dtype=g.dtype)
            diag = torch.stack([Rn[:, 0, 0], Rn[:, 1, 1], Rn[:, 2, 2]], dim=-1)
            idx = torch.argmax(diag, dim=-1)
            for i in range(Rn.shape[0]):
                k = idx[i].item()
                denom = torch.sqrt(
                    torch.clamp(
                        1.0 + Rn[i, k, k] - Rn[i, (k + 1) % 3, (k + 1) % 3] - Rn[i, (k + 2) % 3, (k + 2) % 3],
                        min=1e-12,
                    )
                )
                axis = torch.zeros(3, device=g.device, dtype=g.dtype)
                axis[k] = 0.5 * denom
                axis[(k + 1) % 3] = (Rn[i, (k + 1) % 3, k] + Rn[i, k, (k + 1) % 3]) / (2.0 * denom)
                axis[(k + 2) % 3] = (Rn[i, (k + 2) % 3, k] + Rn[i, k, (k + 2) % 3]) / (2.0 * denom)
                axis = axis / torch.linalg.norm(axis).clamp(min=1e-12)
                phi[i] = thetan[i] * axis
            out[near_pi] = phi

        return out
    
    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        return SO3.hat(SO3.log(g))
    
    @staticmethod
    def left_jacobian(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SO3.g_dim, f"left_jacobian requires shape (N, {SO3.g_dim}), got {tau.shape}"
        theta = SO3._theta(tau)
        # tau_hat = [theta]_x
        tau_hat = SO3.hat(tau)
        tau_hat_sq = tau_hat @ tau_hat
        I = torch.eye(3, device=tau.device, dtype=tau.dtype).expand(tau.shape[0], 3, 3)
        B = SO3._B(theta)[..., None, None]
        C = SO3._C(theta)[..., None, None]
        # Equation 145
        return I + B * tau_hat + C * tau_hat_sq

    @staticmethod
    def left_jacobian_inverse(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SO3.g_dim, f"left_jacobian_inverse requires shape (N, {SO3.g_dim}), got {tau.shape}"
        theta = SO3._theta(tau)
        tau_hat = SO3.hat(tau)
        tau_hat_sq = tau_hat @ tau_hat
        I = torch.eye(3, device=tau.device, dtype=tau.dtype).expand(tau.shape[0], 3, 3)
        D = SO3._D(theta)[..., None, None]
        # Equation 146
        return I - 0.5 * tau_hat + D * tau_hat_sq   
    
    @staticmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SO3.matrix_size, f"adjoint_matrix requires shape (N, {SO3.matrix_size}), got {g.shape}"
        # Equation 139
        return g
    
    @staticmethod
    def ad_operator(xi: torch.Tensor) -> torch.Tensor:
        return SO3.hat(xi)

    @staticmethod
    def coadjoint_operator(xi: torch.Tensor) -> torch.Tensor:
        return SO3.ad_operator(xi).transpose(-2, -1)
    
    @staticmethod
    def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[-1] == 4, f"quaternion_normalize requires shape (N, 4), got {q.shape}"
        return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        q = SO3.quaternion_normalize(q)
        x, y, z, w = q.unbind(dim=-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
        # Equation 138 to compute R(q)
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)
        return R
    
    @staticmethod
    def map_q_to_configuration(q: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[-1] == 4, f"map_q_to_configuration requires shape (N, 4), got {q.shape}"
        return SO3.quaternion_to_matrix(q)

    @staticmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        return SO3.matrix_to_quaternion(g)

    @staticmethod   
    def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        assert R.ndim == 3 and R.shape[-2:] == (3, 3), (
            f"matrix_to_quaternion requires shape (N, 3, 3), got {R.shape}"
        )

        q = torch.empty((R.shape[0], 4), device=R.device, dtype=R.dtype)
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        pos = trace > 0
        if pos.any():
            t = torch.sqrt(trace[pos] + 1.0) * 2.0
            q[pos, 3] = 0.25 * t
            q[pos, 0] = (R[pos, 2, 1] - R[pos, 1, 2]) / t
            q[pos, 1] = (R[pos, 0, 2] - R[pos, 2, 0]) / t
            q[pos, 2] = (R[pos, 1, 0] - R[pos, 0, 1]) / t

        neg = ~pos
        if neg.any():
            Rn = R[neg]
            idx = torch.argmax(
                torch.stack([Rn[:, 0, 0], Rn[:, 1, 1], Rn[:, 2, 2]], dim=-1),
                dim=-1,
            )
            qn = torch.empty((Rn.shape[0], 4), device=R.device, dtype=R.dtype)

            for i in range(Rn.shape[0]):
                if idx[i] == 0:
                    t = torch.sqrt(1.0 + Rn[i, 0, 0] - Rn[i, 1, 1] - Rn[i, 2, 2]) * 2.0
                    qn[i, 0] = 0.25 * t
                    qn[i, 1] = (Rn[i, 0, 1] + Rn[i, 1, 0]) / t
                    qn[i, 2] = (Rn[i, 0, 2] + Rn[i, 2, 0]) / t
                    qn[i, 3] = (Rn[i, 2, 1] - Rn[i, 1, 2]) / t
                elif idx[i] == 1:
                    t = torch.sqrt(1.0 + Rn[i, 1, 1] - Rn[i, 0, 0] - Rn[i, 2, 2]) * 2.0
                    qn[i, 0] = (Rn[i, 0, 1] + Rn[i, 1, 0]) / t
                    qn[i, 1] = 0.25 * t
                    qn[i, 2] = (Rn[i, 1, 2] + Rn[i, 2, 1]) / t
                    qn[i, 3] = (Rn[i, 0, 2] - Rn[i, 2, 0]) / t
                else:
                    t = torch.sqrt(1.0 + Rn[i, 2, 2] - Rn[i, 0, 0] - Rn[i, 1, 1]) * 2.0
                    qn[i, 0] = (Rn[i, 0, 2] + Rn[i, 2, 0]) / t
                    qn[i, 1] = (Rn[i, 1, 2] + Rn[i, 2, 1]) / t
                    qn[i, 2] = 0.25 * t
                    qn[i, 3] = (Rn[i, 1, 0] - Rn[i, 0, 1]) / t

            q[neg] = qn

        return SO3.quaternion_normalize(q)