import torch
from utils.systems.base import  SecondOrderSystem
from typing import Callable, Optional, Tuple

def fe(
    system: SecondOrderSystem,
    q: torch.Tensor,
    dq: torch.Tensor,
    u: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward Euler integration for dynamical systems"""
    d_conf, d_vel = system.f(q, dq, u)
    q_next = system.update_configuration(q, d_conf, dt)
    dq_next = system.update_velocity(dq, d_vel, dt)
    return q_next, dq_next