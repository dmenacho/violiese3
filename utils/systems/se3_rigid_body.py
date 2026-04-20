from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from pymatlie.se3 import SE3
from utils.systems.base import SecondOrderSystem


@dataclass(frozen=True)
class SE3RigidBody(SecondOrderSystem):
    inertia_matrix: torch.Tensor

    q_dim: int = field(default=7, init=False)
    u_dim: int = field(default=6, init=False)

    q_names: Tuple[str, ...] = field(
        default=("x", "y", "z", "qx", "qy", "qz", "qw"),
        init=False,
    )
    dq_names: Tuple[str, ...] = field(
        default=("vx", "vy", "vz", "wx", "wy", "wz"),
        init=False,
    )

    def __post_init__(self):
        if self.inertia_matrix is None:
            object.__setattr__(self, "inertia_matrix", torch.eye(6, dtype=torch.float64))

        if self.inertia_matrix.shape != (6, 6):
            raise ValueError(
                f"inertia_matrix must have shape (6, 6), got {self.inertia_matrix.shape}"
            )

        super().__post_init__()

    @property
    def DOF(self) -> int:
        """
        Lie-algebra pose dimension for SE(3).
        Override the base implementation, which would return q_dim = 7.
        """
        return 6

    @property
    def state_dim(self) -> int:
        """
        Raw state dimension = q + dq = 7 + 6 = 13.
        """
        return 13

    def force_map(self, q: torch.Tensor) -> torch.Tensor:
        """
        Identity force map for the minimal model.

        Returns B(q) of shape (N, 6, 6), so that:
            generalized_force = B(q) @ u = u
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        N = q.shape[0]
        return torch.eye(
            6,
            device=q.device,
            dtype=q.dtype,
        ).unsqueeze(0).expand(N, -1, -1)

    def potential_forces(self, q: torch.Tensor) -> torch.Tensor:
        """
        No potential forces in the minimal model.
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        return torch.zeros((q.shape[0], 6), device=q.device, dtype=q.dtype)

    def dissipative_forces(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """
        No damping in the minimal model.
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        assert dq.ndim == 2 and dq.shape[1] == 6, (
            f"dq must be of shape (N, 6), got {dq.shape}"
        )
        return torch.zeros_like(dq)

    def f(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            d_conf : shape (N, 6)
            d_vel  : shape (N, 6)

        Here:
            d_conf = dq
            d_vel  = M^{-1} u
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        assert dq.ndim == 2 and dq.shape[1] == 6, (
            f"dq must be of shape (N, 6), got {dq.shape}"
        )
        assert u.ndim == 2 and u.shape[1] == self.u_dim, (
            f"u must be of shape (N, {self.u_dim}), got {u.shape}"
        )

        q = q.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)
        dq = dq.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)
        u = u.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)

        d_conf = dq
        d_vel = u @ self.inertia_matrix_inv.T
        return d_conf, d_vel

    def update_configuration(self, q: torch.Tensor, dq: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Update pose using SE(3) right-plus:
            g_next = g * Exp(dq * dt)

        Then map back to raw pose coordinates [x, y, z, qx, qy, qz, qw].
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        assert dq.ndim == 2 and dq.shape[1] == 6, (
            f"dq must be of shape (N, 6), got {dq.shape}"
        )
        assert dt > 0, "dt must be positive"

        g = SE3.map_q_to_configuration(q)
        g_next = SE3.right_plus(g, dq * dt)
        return SE3.map_configuration_to_q(g_next)

    def update_velocity(self, dq: torch.Tensor, ddq: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Euclidean twist update.
        """
        assert dq.ndim == 2 and dq.shape[1] == 6, (
            f"dq must be of shape (N, 6), got {dq.shape}"
        )
        assert ddq.ndim == 2 and ddq.shape[1] == 6, (
            f"ddq must be of shape (N, 6), got {ddq.shape}"
        )
        assert dt > 0, "dt must be positive"
        return dq + ddq * dt

    def get_linearized_dynamics_matrices(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        u: torch.Tensor,
        error_form: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Continuous-time linearized error dynamics for the minimal SE(3) model.

        Error state:
            delta_x = [delta_pose(6), delta_velocity(6)] in R^12

        We keep the important Lie-theoretic structure from the old SE(2) code:
        pose-error dynamics depend on the adjoint operator ad(dq).

        RI:
            d/dt(delta_pose) =  ad(dq) delta_pose - delta_velocity

        LI:
            d/dt(delta_pose) = -ad(dq) delta_pose + delta_velocity

        Velocity-error dynamics are kept minimal for now:
            d/dt(delta_velocity) = u_error
        so A22 = 0 and B injects into the velocity block.

        Returns
        -------
        A : (N, 12, 12)
        B : (12, 6)
            Same B for all batch elements, matching the style of the old SE(2) code.
        """
        assert q.ndim == 2 and q.shape[1] == self.q_dim, (
            f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        )
        assert dq.ndim == 2 and dq.shape[1] == 6, (
            f"dq must be of shape (N, 6), got {dq.shape}"
        )
        assert u.ndim == 2 and u.shape[1] == self.u_dim, (
            f"u must be of shape (N, {self.u_dim}), got {u.shape}"
        )

        q = q.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)
        dq = dq.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)
        u = u.to(device=self.inertia_matrix.device, dtype=self.inertia_matrix.dtype)

        N = q.shape[0]
        D = self.DOF

        A = torch.zeros(
            (N, 2 * D, 2 * D),
            device=self.inertia_matrix.device,
            dtype=self.inertia_matrix.dtype,
        )

        I6 = torch.eye(
            D,
            device=self.inertia_matrix.device,
            dtype=self.inertia_matrix.dtype,
        )

        ad_xi = SE3.ad_operator(dq)

        if error_form == "LI":
            A[:, :D, :D] = -ad_xi
            A[:, :D, D:] = I6
        elif error_form == "RI":
            A[:, :D, :D] = ad_xi
            A[:, :D, D:] = -I6
        else:
            raise ValueError(f"Invalid error_form: {error_form}")

        # Minimal model: no detailed velocity Jacobian yet.
        # A[:, D:, D:] remains zero.

        B = torch.zeros(
            (2 * D, self.u_dim),
            device=self.inertia_matrix.device,
            dtype=self.inertia_matrix.dtype,
        )
        B[D:, :] = self.inertia_matrix_inv

        return A, B