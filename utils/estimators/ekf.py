import torch

from utils.estimators.base import BaseEstimator


class EKFEstimator(BaseEstimator):

    def __init__(self, system, integrator, dt: float, error_form: str):
        super().__init__(system=system)
        self.integrator = integrator
        self.dt: float = dt
        self.error_form: str = error_form
        assert self.integrator.__name__ == "fe"

    def predict_step(self, state_estimate: dict, u: torch.Tensor) -> dict:
        """Single substep prediction (original implementation)."""
        p_hat = state_estimate["pose"]
        v_hat = state_estimate["velocity"]

        # Jacobians are evaluated at
        A, G = self.system.get_linearized_dynamics_matrices(p_hat, v_hat, u, error_form=self.error_form)

        P = state_estimate["P"]
        Q = state_estimate["Q"]
        A = A.to(device=P.device, dtype=P.dtype)
        G = G.to(device=P.device, dtype=P.dtype)

        N = P.shape[0]
        noise_dim = G.shape[-1]
        if Q.ndim == 2:
            if Q.shape != (noise_dim, noise_dim):
                raise ValueError(
                    f"Q must have shape {(noise_dim, noise_dim)}, got {Q.shape}"
                )
            Qb = Q.unsqueeze(0).expand(N, -1, -1)
        elif Q.ndim == 3:
            if Q.shape != (N, noise_dim, noise_dim):
                raise ValueError(
                    f"Batched Q must have shape {(N, noise_dim, noise_dim)}, got {Q.shape}"
                )
            Qb = Q
        else:
            raise ValueError(f"Q must be 2D or 3D, got shape {Q.shape}")

        A_discrete = torch.eye(*A.shape[1:], device=P.device, dtype=P.dtype) + A * self.dt

        Pk1 = A_discrete @ P @ A_discrete.transpose(-1, -2) + G @ Qb @ G.transpose(-1, -2) * self.dt

        p_hat_next, v_hat_next = self.integrator(self.system, p_hat, v_hat, u, self.dt)

        state_estimate["pose"] = p_hat_next
        state_estimate["velocity"] = v_hat_next
        state_estimate["P"] = Pk1

        return state_estimate

    def predict_planning_step(self, state_estimate: dict, u: torch.Tensor, n_substeps: int) -> dict:
        """Multi-substep prediction with proper discrete-time noise handling."""
        for i in range(n_substeps):
            state_estimate = self.predict_step(state_estimate, u)
        return state_estimate