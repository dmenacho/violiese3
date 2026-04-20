from dataclasses import dataclass, field
from typing import Tuple

import torch

from pymatlie.base_group import MatrixLieGroup
from pymatlie.so3 import SO3

@dataclass(frozen=True)
class SE3(MatrixLieGroup):
    """Special Euclidean group SE(3)."""

    inertia_matrix: torch.Tensor
    g_dim: int = 6
    matrix_size: tuple = (4, 4)
    B: torch.Tensor = torch.eye(6, dtype=torch.float64)
    u_dim: int = 6
    g_names: Tuple[str, ...] = field(
        default=(
            r"$R_{11}$", r"$R_{12}$", r"$R_{13}$", r"$x$",
            r"$R_{21}$", r"$R_{22}$", r"$R_{23}$", r"$y$",
            r"$R_{31}$", r"$R_{32}$", r"$R_{33}$", r"$z$",
            r"$0$", r"$0$", r"$0$", r"$1$",
        ),
        init=False,
    )
    xi_names: Tuple[str, ...] = field(
        default=(r"v_x", r"v_y", r"v_z", r"\omega_x", r"\omega_y", r"\omega_z"),
        init=False,
    )

    @staticmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        assert tau_hat.ndim == 3 and tau_hat.shape[-2:] == SE3.matrix_size, f"vee requires shape (N, {SE3.matrix_size}), got {tau_hat.shape}"
        rho = tau_hat[..., :3, 3]
        phi = SO3.vee(tau_hat[..., :3, :3])
        # [rho, phi] e R^6
        return torch.cat([rho, phi], dim=-1)
    
    @staticmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"hat requires shape (N, {SE3.g_dim}), got {tau.shape}"
        out = torch.zeros((tau.shape[0], 4, 4), device=tau.device, dtype=tau.dtype)
        out[..., :3, :3] = SO3.hat(tau[..., 3:])
        out[..., :3, 3] = tau[..., :3]
        # [[[theta]_x, phi], [0, 0, 0, 0]] ]
        return out

    @staticmethod
    def exp(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"exp requires shape (N, {SE3.g_dim}), got {tau.shape}"
        rho = tau[..., :3]
        phi = tau[..., 3:]
        # Equation 172 M = Exp(thau)
        R = SO3.exp(phi)
        V = SO3.left_jacobian(phi)
        t = torch.bmm(V, rho.unsqueeze(-1)).squeeze(-1)
        T = torch.eye(4, device=tau.device, dtype=tau.dtype).expand(tau.shape[0], 4, 4).clone()
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        return T

    @staticmethod
    def log(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"log requires shape (N, {SE3.matrix_size}), got {g.shape}"
        # Equation 173 thau = Log(M)
        R = g[..., :3, :3]
        t = g[..., :3, 3]
        phi = SO3.log(R)
        V_inv = SO3.left_jacobian_inverse(phi)
        rho = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)
        return torch.cat([rho, phi], dim=-1)

    @staticmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        return SE3.hat(SE3.log(g))
    
    @staticmethod
    def _Q_matrix(rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Barfoot/Sola Q block used in the SE(3) left Jacobian.

        Implements the formula shown in Sola Appendix D, Eq. (180).
        """
        assert rho.shape == phi.shape and rho.shape[-1] == 3
        dtype = rho.dtype
        device = rho.device
        B = rho.shape[0]

        rx = SO3.hat(rho)
        px = SO3.hat(phi)
        px2 = px @ px
        theta = torch.linalg.norm(phi, dim=-1)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta5 = theta4 * theta
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        # Stable coefficients with Taylor expansions near zero.
        c1 = torch.empty_like(theta)
        c2 = torch.empty_like(theta)
        c3 = torch.empty_like(theta)
        small = theta.abs() < 1e-6

        c1[small] = 1.0 / 6.0 - theta2[small] / 120.0 + theta2[small] * theta2[small] / 5040.0
        c2[small] = 1.0 / 24.0 - theta2[small] / 720.0 + theta2[small] * theta2[small] / 40320.0
        c3[small] = 1.0 / 120.0 - theta2[small] / 2520.0 + theta2[small] * theta2[small] / 120960.0

        c1[~small] = (theta[~small] - sin_t[~small]) / (theta2[~small] * theta[~small])
        c2[~small] = (1.0 - 0.5 * theta2[~small] - cos_t[~small]) / theta4[~small]
        c3[~small] = (
            1.0
            - 0.5 * theta2[~small]
            - cos_t[~small]
            - 3.0 * (theta[~small] - sin_t[~small] - theta[~small] ** 3 / 6.0) / theta[~small]
        ) / theta5[~small]

        term1 = 0.5 * rx
        term2 = c1[..., None, None] * (px @ rx + rx @ px + px @ rx @ px)
        term3 = -c2[..., None, None] * (px2 @ rx + rx @ px2 - 3.0 * (px @ rx @ px))
        term4 = -0.5 * c3[..., None, None] * (px @ rx @ px2 + px2 @ rx @ px)
        Q = term1 + term2 + term3 + term4
        return Q.to(device=device, dtype=dtype).reshape(B, 3, 3)

    @staticmethod
    def left_jacobian(tau: torch.Tensor) -> torch.Tensor:
        """SE(3) left Jacobian.

        J_l(rho, phi) = [[J_l(phi), Q(rho,phi)], [0, J_l(phi)]].
        """
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"left_jacobian requires shape (N, {SE3.g_dim}), got {tau.shape}"
        rho = tau[..., :3]
        phi = tau[..., 3:]
        # Equation 179a
        J = SO3.left_jacobian(phi)
        Q = SE3._Q_matrix(rho, phi)
        out = torch.zeros((tau.shape[0], 6, 6), device=tau.device, dtype=tau.dtype)
        out[..., :3, :3] = J
        out[..., :3, 3:] = Q
        out[..., 3:, 3:] = J
        return out

    @staticmethod
    def left_jacobian_inverse(tau: torch.Tensor) -> torch.Tensor:
        assert tau.ndim == 2 and tau.shape[-1] == SE3.g_dim, f"left_jacobian_inverse requires shape (N, {SE3.g_dim}), got {tau.shape}"
        rho = tau[..., :3]
        phi = tau[..., 3:]
        # Equation 179b
        J_inv = SO3.left_jacobian_inverse(phi)
        Q = SE3._Q_matrix(rho, phi)
        out = torch.zeros((tau.shape[0], 6, 6), device=tau.device, dtype=tau.dtype)
        out[..., :3, :3] = J_inv
        out[..., :3, 3:] = -torch.bmm(torch.bmm(J_inv, Q), J_inv)
        out[..., 3:, 3:] = J_inv
        return out
    
    @staticmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"adjoint_matrix requires shape (N, {SE3.matrix_size}), got {g.shape}"
        R = g[..., :3, :3]
        t = g[..., :3, 3]
        # [t]_x
        t_hat = SO3.hat(t)
        # Equarion 175
        out = torch.zeros((g.shape[0], 6, 6), device=g.device, dtype=g.dtype)
        out[..., :3, :3] = R
        out[..., :3, 3:] = torch.bmm(t_hat, R)
        out[..., 3:, 3:] = R
        return out

    @staticmethod
    def ad_operator(xi: torch.Tensor) -> torch.Tensor:
        assert xi.ndim == 2 and xi.shape[-1] == SE3.g_dim, f"ad_operator requires shape (N, {SE3.g_dim}), got {xi.shape}"
        rho = xi[..., :3]
        phi = xi[..., 3:]
        out = torch.zeros((xi.shape[0], 6, 6), device=xi.device, dtype=xi.dtype)
        out[..., :3, :3] = SO3.hat(phi)
        out[..., :3, 3:] = SO3.hat(rho)
        out[..., 3:, 3:] = SO3.hat(phi)
        return out

    @staticmethod
    def coadjoint_operator(xi: torch.Tensor) -> torch.Tensor:
        return SE3.ad_operator(xi).transpose(-2, -1)

    @staticmethod
    def map_q_to_configuration(q: torch.Tensor) -> torch.Tensor:
        """Raw state pose -> SE(3) matrix.

        q = [x, y, z, qx, qy, qz, qw]
        """
        assert q.ndim == 2 and q.shape[-1] == 7, f"map_q_to_configuration requires shape (N, 7), got {q.shape}"
        T = torch.eye(4, device=q.device, dtype=q.dtype).expand(q.shape[0], 4, 4).clone()
        T[..., :3, :3] = SO3.quaternion_to_matrix(q[..., 3:])
        T[..., :3, 3] = q[..., :3]
        return T

    @staticmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        assert g.ndim == 3 and g.shape[-2:] == SE3.matrix_size, f"map_configuration_to_q requires shape (N, {SE3.matrix_size}), got {g.shape}"
        t = g[..., :3, 3]
        q = SO3.matrix_to_quaternion(g[..., :3, :3])
        return torch.cat([t, q], dim=-1)

    @staticmethod
    def map_dq_to_velocity(q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Initial minimal convention for SE(3).

        Assumes dq already stores a 6D twist:
            [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        """
        assert q.ndim == 2 and q.shape[-1] == 7, f"map_dq_to_velocity requires q shape (N, 7), got {q.shape}"
        assert dq.ndim == 2 and dq.shape[-1] == 6, f"map_dq_to_velocity requires dq shape (N, 6), got {dq.shape}"
        return dq

    @staticmethod
    def map_velocity_to_dq(q: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        assert q.ndim == 2 and q.shape[-1] == 7, f"map_velocity_to_dq requires q shape (N, 7), got {q.shape}"
        assert velocity.ndim == 2 and velocity.shape[-1] == 6, f"map_velocity_to_dq requires velocity shape (N, 6), got {velocity.shape}"
        return velocity