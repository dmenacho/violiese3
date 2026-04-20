"""Base class for matrix Lie groups."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import torch


@dataclass(frozen=True)
class MatrixLieGroup(ABC):
    """Base class for matrix Lie groups. Serves as interface for implementing
    specific groups like SE(2), SO(2), etc.

    Reference:
    [1] https://arxiv.org/pdf/1812.01537
    """

    # Must be set in subclass
    inertia_matrix: torch.Tensor  # Inertia matrix
    inertia_matrix_inv: torch.Tensor = field(init=False, repr=False)  # Inverse of inertia matrix (precomputed)
    u_dim: int  # Control input dimension
    B: torch.Tensor  # Force Map (Linear Map from Control Inputs to Body-Fixed Forces/Torques)
    g_dim: int  # Degrees of freedom of the Lie group = dimension of the Lie algebra; e.g. 3 for SE(2)
    matrix_size: Tuple[int, int]  # Size of the matrix; e.g. (3, 3) for SE(2)
    B_T: torch.Tensor = field(init=False, repr=False)  # Transpose of B (precomputed)

    g_names: Tuple[str, ...] = field(init=False)
    xi_names: Tuple[str, ...] = field(init=False)

    def __post_init__(self):
        assert self.g_dim > 0, "DIM must be positive"
        assert self.matrix_size[0] == self.matrix_size[1], "Matrix must be square"
        assert self.matrix_size[0] > 0, "Matrix size must be positive"
        assert len(self.g_names) == self.SIZE, f"g_names must have length {self.SIZE}, got {len(self.g_names)}"
        assert len(self.xi_names) == self.g_dim, f"xi_names must have length {self.g_dim}, got {len(self.xi_names)}"

        assert self.inertia_matrix.shape == (
            self.g_dim,
            self.g_dim,
        ), f"dynamics_step: inertia_matrix must be ({self.g_dim}, {self.g_dim}), got {self.inertia_matrix.shape}"

        object.__setattr__(self, "inertia_matrix_inv", torch.linalg.inv(self.inertia_matrix))
        assert self.B.shape == (self.g_dim, self.u_dim), f"dynamics_step: B must be ({self.g_dim}, {self.u_dim}), got {self.B.shape}"
        object.__setattr__(self, "B_T", self.B.T)

    def project_to_motion_constraints(self, _g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Project the state to the motion constraints."""
        return xi

    @property
    def SIZE(self) -> int:
        """Size of the Lie group element matrix."""
        return self.matrix_size[0] * self.matrix_size[1]

    @property
    def DOF(self) -> int:
        """Degrees of freedom of the Lie group.

        Shared with other libraries/utils.
        """
        return self.g_dim

    @property
    def state_dim(self) -> int:
        """State dimension: g + xi"""
        return self.g_dim + self.g_dim

    @classmethod
    def get_identity(cls) -> torch.Tensor:
        """For all Matrix Lie groups, the identity element is the identity
        matrix.

        identity @ g = g @ identity = g
        """
        assert cls.matrix_size is not None, "Matrix size is not defined."
        return torch.eye(cls.matrix_size[0])

    @staticmethod
    def inverse(g: torch.Tensor) -> torch.Tensor:
        """Inverse of Lie group element."""
        return torch.linalg.inv(g)

    @staticmethod
    @abstractmethod
    def vee(tau_hat: torch.Tensor) -> torch.Tensor:
        """Converts a Lie algebra element (matrix) to a vector."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hat(tau: torch.Tensor) -> torch.Tensor:
        """Converts a vector to a Lie algebra element (matrix)."""
        raise NotImplementedError

    @classmethod
    def wedge(cls, tau: torch.Tensor) -> torch.Tensor:
        """Alias for `hat` function."""
        return cls.hat(tau)

    @staticmethod
    def expm(tau_hat: torch.Tensor) -> torch.Tensor:
        """Computes the matrix exponential of a Lie algebra element."""
        return torch.linalg.matrix_exp(tau_hat)

    @staticmethod
    @abstractmethod
    def logm(g: torch.Tensor) -> torch.Tensor:
        """Computes the matrix logarithm of a Lie group element."""
        raise NotImplementedError

    @classmethod
    def exp(cls, tau: torch.Tensor) -> torch.Tensor:
        """Eq.

        (21) in [1]
        u: Lie algebra element
        exp(u) = expm(u^)
        """
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"exp requires shape (N, {cls.g_dim}), got {tau.shape}"
        return cls.expm(cls.hat(tau))

    @classmethod
    def log(cls, g: torch.Tensor) -> torch.Tensor:
        """Eq.

        (22) in [1]
        g: Lie group element
        Log(g) = vee(log(g))
        """
        assert g.ndim == 3 and g.shape[-2:] == cls.matrix_size, f"log requires shape (N, {cls.matrix_size}), got {g.shape}"
        return cls.vee(cls.logm(g))

    @staticmethod
    @abstractmethod
    def left_jacobian(tau: torch.Tensor) -> torch.Tensor:
        """Computes the left Jacobian of the Lie group."""
        raise NotImplementedError

    @classmethod
    def right_jacobian(cls, tau: torch.Tensor) -> torch.Tensor:
        """Eq.

        (76) in [1]
        """
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"right_jacobian requires shape (N, {cls.g_dim}), got {tau.shape}"
        return cls.left_jacobian(-tau)

    @staticmethod
    @abstractmethod
    def adjoint_matrix(g: torch.Tensor) -> torch.Tensor:
        """Computes the adjoint matrix of a Lie group element."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def coadjoint_operator(xi: torch.Tensor) -> torch.Tensor:
        """Computes the coadjoint operator of a Lie algebra element."""
        raise NotImplementedError

    @classmethod
    def right_plus(cls, g: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Eq. (25) in [1]

        g: Lie group element
        tau: Lie algebra element
        g o+ delta = g @ exp(tau) = g @ expm(tau^)
        """
        assert g.ndim == 3 and g.shape[-2:] == cls.matrix_size, f"right_plus: g must be (N, {cls.matrix_size}), got {g.shape}"
        assert tau.ndim == 2 and tau.shape[-1] == cls.g_dim, f"right_plus: tau must be (N, {cls.g_dim}), got {tau.shape}"

        return torch.matmul(g, cls.exp(tau))

    @classmethod
    def right_minus(cls, g_start: torch.Tensor, g_end: torch.Tensor) -> torch.Tensor:
        """Eq.

        (26) in [1] tau = g_start o- g_end = Log(g_start^-1 @ g_end) tau
        goes from g_start to g_end
        """
        assert (
            g_start.ndim == 3 and g_start.shape[-2:] == cls.matrix_size
        ), f"right_minus: g_start must be (N, {cls.matrix_size}), got {g_start.shape}"
        assert g_end.ndim == 3 and g_end.shape[-2:] == cls.matrix_size, f"right_minus: g_end must be (N, {cls.matrix_size}), got {g_end.shape}"
        g_diff = torch.matmul(cls.inverse(g_start), g_end)
        return cls.log(g_diff)

    @classmethod
    def right_invariant_error(cls, estimated_state: torch.Tensor, true_state: torch.Tensor) -> torch.Tensor:
        """Computes the right invariant error between the estimated state and
        the true state."""
        # assert estimated_state.shape == true_state.shape, f"right_invariant_error: mismatched shapes {estimated_state.shape} vs {true_state.shape}"
        # return estimated_state @ cls.inverse(true_state)
        return true_state @ cls.inverse(estimated_state)

    @classmethod
    def left_invariant_error(cls, estimated_state: torch.Tensor, true_state: torch.Tensor) -> torch.Tensor:
        """Computes the left invariant error between the estimated state and
        the true state."""
        assert estimated_state.shape == true_state.shape, f"right_invariant_error: mismatched shapes {estimated_state.shape} vs {true_state.shape}"
        return cls.inverse(true_state) @ estimated_state

    def f(self, _g: torch.Tensor, xi: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update using Euler-Poincare equations."""
        D = self.g_dim
        assert xi.ndim == 2 and xi.shape[1] == D, f"xi must be (N, {D}), got {xi.shape}"
        assert u.shape[1] == self.u_dim and u.ndim == 2, f"u must be (N, {self.u_dim}), got {u.shape}"
        u = u.to(device=self.B_T.device, dtype=self.B_T.dtype)
        Bu = u @ self.B_T  # Bu in batched form

        coad = self.coadjoint_operator(xi)  # (N, D, D)
        inertia_matrix_xi = xi @ self.inertia_matrix.T  # (N, D) inertia_matrixb @ xi
        rhs = torch.bmm(coad, inertia_matrix_xi[..., None])[..., 0] + Bu  # Broadcasting same inertia_matrix for all xi
        xi_dot = (self.inertia_matrix_inv @ rhs.T).T
        return xi, xi_dot

    def lie_bracket(self, xi_1: torch.Tensor, xi_2: torch.Tensor) -> torch.Tensor:
        """Lie bracket of two Lie algebra elements."""
        assert xi_1.ndim == 2 and xi_1.shape[-1] == self.g_dim, f"lie_bracket: xi_1 must be (N, {self.g_dim}), got {xi_1.shape}"
        assert xi_2.ndim == 2 and xi_2.shape[-1] == self.g_dim, f"lie_bracket: xi_2 must be (N, {self.g_dim}), got {xi_2.shape}"
        h1 = self.hat(xi_1)
        h2 = self.hat(xi_2)
        return self.vee(h1 @ h2 - h2 @ h1)

    def update_configuration(self, g: torch.Tensor, xi: torch.Tensor, dt: float) -> torch.Tensor:
        """Updates the configuration (group element g) using the Lie algebra
        element xi.

        Called 'reconstruction_step' in literature.
        """
        assert g.ndim == 3 and g.shape[-2:] == self.matrix_size, f"update_q: g must be (N, {self.matrix_size}), got {g.shape}"
        assert xi.ndim == 2 and xi.shape[-1] == self.g_dim, f"update_q: xi must be (N, {self.g_dim}), got {xi.shape}"
        xi = xi.to(device=g.device, dtype=g.dtype)
        return g @ self.exp(xi * dt)

    def update_velocity(self, xi: torch.Tensor, dxi: torch.Tensor, dt: float) -> torch.Tensor:
        """Updates the velocity (Lie algebra element xi) using the Lie algebra
        element dxi."""
        assert xi.ndim == 2 and xi.shape[-1] == self.g_dim, f"update_velocity: xi must be (N, {self.g_dim}), got {xi.shape}"
        assert dxi.ndim == 2 and dxi.shape[-1] == self.g_dim, f"update_velocity: dxi must be (N, {self.g_dim}), got {dxi.shape}"
        return xi + dxi * dt

    @staticmethod
    @abstractmethod
    def map_q_to_configuration(q: torch.Tensor) -> torch.Tensor:
        """Map the configuration vector to the Lie group element."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def map_configuration_to_q(g: torch.Tensor) -> torch.Tensor:
        """Map the Lie Group element to configuration space."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def map_dq_to_velocity(q: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """Map the velocity in configuration space to the Lie Algebra
        velocity."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def map_velocity_to_dq(q: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Map the velocity in Lie Algebra to the configuration space
        velocity."""
        raise NotImplementedError


@dataclass(frozen=True)
class NonholonomicGroup(MatrixLieGroup):
    """Base class for nonholonomic matrix Lie groups."""

    constraint_projection_matrix_velocity: torch.Tensor = field(init=False, repr=False)  # Enforces A @ xi = 0 constraint
    constraint_projection_matrix_wrench: torch.Tensor = field(init=False, repr=False)  # Enforces

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        zeros = torch.zeros(
            (1, self.g_dim),
            device=self.inertia_matrix.device,
            dtype=self.inertia_matrix.dtype,
        )
        A_matrix = self.get_Pfaffian_A(zeros, zeros)[0]
        assert (
            A_matrix.ndim == 2 and A_matrix.shape[0] < self.g_dim and A_matrix.shape[1] == self.g_dim
        ), f"get_Pfaffian_A must return (k, {self.g_dim}), got {A_matrix.shape}"

        Ainertia_matrixinv = A_matrix @ self.inertia_matrix_inv  # A @ inertia_matrix^-1
        lambda_solver = torch.linalg.inv(Ainertia_matrixinv @ A_matrix.T)  # (A @ inertia_matrix^-1 @ A^T)^-1
        P = self.inertia_matrix_inv @ A_matrix.T @ lambda_solver @ A_matrix  # inertia_matrix^-1 @ A^T @ (A @ inertia_matrix^-1 @ A^T)^-1 @ A
        I_minus_P = torch.eye(
            self.g_dim,
            device=self.inertia_matrix.device,
            dtype=self.inertia_matrix.dtype,
        ) - P

        PI = (
            torch.eye(
                self.g_dim,
                device=self.inertia_matrix.device,
                dtype=self.inertia_matrix.dtype,
            )
            - A_matrix.T @ torch.linalg.inv(A_matrix @ self.inertia_matrix_inv @ A_matrix.T) @ A_matrix @ self.inertia_matrix_inv
        )

        object.__setattr__(self, "constraint_projection_matrix_velocity", I_minus_P)  # Storing transpose for batch multiplication purposes
        object.__setattr__(self, "constraint_projection_matrix_wrench", PI)

    def project_to_motion_constraints(self, _g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Project the state to the motion constraints."""
        return xi @ self.constraint_projection_matrix_velocity.T

    @abstractmethod
    def get_Pfaffian_A(self, g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Computes the Pfaffian A of the nonholonomic group."""
        raise NotImplementedError

    def f(self, g: torch.Tensor, xi: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update using Euler-Poincare equations."""
        # _, xi_dot = super().f(g, xi, u)
        D = self.g_dim
        assert xi.ndim == 2 and xi.shape[1] == D, f"xi must be (N, {D}), got {xi.shape}"
        assert u.shape[1] == self.u_dim and u.ndim == 2, f"u must be (N, {self.u_dim}), got {u.shape}"
        u = u.to(device=self.B_T.device, dtype=self.B_T.dtype)
        Bu = u @ self.B_T  # Bu in batched form

        coad = self.coadjoint_operator(xi)  # (N, D, D)
        inertia_matrix_xi = xi @ self.inertia_matrix.T  # (N, D) inertia_matrixb @ xi
        rhs = torch.bmm(coad, inertia_matrix_xi[..., None])[..., 0] + Bu  # Broadcasting same inertia_matrix for all xi
        xi_dot = (self.inertia_matrix_inv @ rhs.T).T
        xi_dot_proj = self.project_to_motion_constraints(g, xi_dot)

        return xi, xi_dot_proj
