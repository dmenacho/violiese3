"""Base class for 2nd-order dynamical systems in Euclidean space."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch


@dataclass(frozen=True)
class SecondOrderSystem(ABC):
    """Abstract class for 2nd-order dynamical systems in Euclidean space."""

    # Must be set in subclass
    inertia_matrix: torch.Tensor  # Inertia matrix
    inertia_matrix_inv: torch.Tensor = field(init=False, repr=False)  # Inverse inertia matrix
    q_dim: int
    u_dim: int

    q_names: Tuple[str, ...] = field(init=False)
    dq_names: Tuple[str, ...] = field(init=False)

    def __post_init__(self):
        assert self.q_dim > 0, "q_dim must be > 0"
        assert len(self.q_names) == self.q_dim, "q_names must match q_dim"

        object.__setattr__(self, "inertia_matrix_inv", torch.linalg.inv(self.inertia_matrix))

    @property
    def DOF(self) -> int:
        """Degrees of freedom.

        Shared with other system types.
        """
        return self.q_dim

    @property
    def state_dim(self) -> int:
        """State dimension: q + dq"""
        return self.q_dim * 2
    
    def project_to_motion_constraints(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Project dq into the motion constraints."""
        return dq

    def f(self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns d_conf, d_vel = f(q, dq, u)."""
        d_conf = dq # Kinematics
        d_vel = self.total_forces(q, dq, u) @ self.inertia_matrix_inv.T # Dynamics
        return d_conf, d_vel
    
    def total_forces(self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Returns the total forces acting on the system."""
        assert q.ndim == 2 and q.shape[1] == self.q_dim, f"q must be of shape (N, {self.q_dim})"
        assert dq.ndim == 2 and dq.shape[1] == self.q_dim, f"dq must be of shape (N, {self.q_dim})"
        assert u.ndim == 2 and u.shape[1] == self.u_dim, f"Action must be of shape (N, {self.u_dim})"
        
        # External Control Forces
        B = self.force_map(q)  # (N, q_dim, u_dim)
        Bu = torch.einsum("nqu,nu->nq", B, u)  # (N, q_dim)

        F_potential = self.potential_forces(q)  # (N, q_dim) = -∇V(q)
        F_dissipative = self.dissipative_forces(q, dq)  # (N, q_dim) = D(q, dq)
        # TODO: add coriolis forces

        F_total = Bu + F_potential + F_dissipative  # (N, q_dim)

        return F_total

    def update_configuration(self, q: torch.Tensor, dq: torch.Tensor, dt: float) -> torch.Tensor:
        """Update q given dq."""
        assert q.ndim == 2 and q.shape[1] == self.q_dim, f"q must be of shape (N, {self.q_dim}), got {q.shape}"
        assert dq.ndim == 2 and dq.shape[1] == self.q_dim, f"dq must be of shape (N, {self.q_dim}), got {dq.shape}"
        assert dt > 0, "dt must be positive"
        # TODO: think i should change logic so that dq.shape == self.dq_dim ..
        return q + dq * dt
    
    def update_velocity(self, dq: torch.Tensor, ddq: torch.Tensor, dt: float) -> torch.Tensor:
        """Update dq given d_vel."""
        assert dq.ndim == 2 and dq.shape[1] == self.q_dim, f"dq must be of shape (N, {self.q_dim})"
        assert ddq.ndim == 2 and ddq.shape[1] == self.q_dim, f"ddq must be of shape (N, {self.q_dim})"
        assert dt > 0, "dt must be positive"
        return dq + ddq * dt

    @abstractmethod
    def force_map(self, q: torch.Tensor) -> torch.Tensor:
        """Maps control inputs to generalized forces."""
        raise NotImplementedError

    @abstractmethod
    def potential_forces(self, q: torch.Tensor) -> torch.Tensor:
        """Maps configuration to potential forces."""
        raise NotImplementedError

    @abstractmethod
    def dissipative_forces(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Maps configuration to potential forces."""
        raise NotImplementedError

    @abstractmethod
    def get_linearized_dynamics_matrices(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        u: torch.Tensor,
        error_form: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the discrete‐time Jacobians (A, B) where
          A = ∂f/∂x   ∈ ℝ^{(2n×2n)},
          B = ∂f/∂u   ∈ ℝ^{(2n×m)}.
        Here x = [q; dq], f gives Δx/dt (or the discrete step).
        """
        raise NotImplementedError

class SecondOrderNonholonomicSystem(SecondOrderSystem):
    """Second-order nonholonomic system."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_Pfaffian_A(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Returns the Pfaffian matrix A."""
        raise NotImplementedError

    @abstractmethod
    def get_Pfaffian_A_dot(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Returns the time derivative of the Pfaffian matrix A."""
        raise NotImplementedError

    def f(self, q: torch.Tensor, dq: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (q̇, q̈) solving the nonholonomic constraints
          A(q)·q̈ + Ȧ(q,dq)·dq = 0
        by eliminating λ via
          A M⁻¹ Aᵀ λ = −[A M⁻¹ F_ext + Ȧ·dq].
        """
        # Like holonomic system
        d_conf = dq # Kinematics
        acc_unproj = self.total_forces(q, dq, u) @ self.inertia_matrix_inv.T # Dynamics

        # Imposing nonholonomic constraints
        A     = self.get_Pfaffian_A(q, dq)                              # (N, k, n)
        A_dot = self.get_Pfaffian_A_dot(q, dq)                      # (N, k, n)

        # 3) RHS for λ‐solve:  −(A q̈_free + Ȧ dq)
        rhs = - (A @ acc_unproj.unsqueeze(-1) + A_dot @ dq.unsqueeze(-1))

        # 4) Solve (A M⁻¹ Aᵀ) λ = rhs  →  lam: (N, k)
        AMAt = A @ (self.inertia_matrix_inv @ A.transpose(-2, -1))        # (N, k, k)
        lam  = torch.linalg.solve(AMAt, rhs).squeeze(-1)                  # (N, k)

        # 5) Constraint correction:  q̈_corr = M⁻¹ Aᵀ λ
        corr = (self.inertia_matrix_inv @ (A.transpose(-2, -1) @ lam.unsqueeze(-1))).squeeze(-1)  # (N, n)
        d_vel = acc_unproj + corr

        return d_conf, d_vel


    def project_dq_and_compute_lagrange_multipliers(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Solve the Lagrange multipliers for the constraints."""
        assert hasattr(self, "get_Pfaffian_A"), "get_Pfaffian_A must be defined in subclass"
        A_mat = self.get_Pfaffian_A(q, dq)
        A_mat_transpose = A_mat.transpose(-2, -1)
        M = A_mat @ (self.inertia_matrix_inv @ A_mat_transpose)
        rhs = A_mat @ dq.unsqueeze(-1)  # Adq
        lam = torch.linalg.solve(M, rhs)

        correction = (self.inertia_matrix_inv @ (A_mat_transpose @ lam)).squeeze(-1)  # Jinv·Aᵀ·λ
        dq_proj = dq - correction
        return dq_proj, lam
    
    def project_to_motion_constraints(self, q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """
        Project dq into the nonholonomic constraint kernel via
          P = J^{-1} A^T (A J^{-1} A^T)^{-1} A,
        so that
          dq_proj = (I - P) dq.

        Here `correction = P @ dq`, and we subtract it from dq.
        """
        dq_proj, _ = self.project_dq_and_compute_lagrange_multipliers(q, dq)
        return dq_proj