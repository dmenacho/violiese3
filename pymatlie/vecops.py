"""Shared vector operations for Lie groups and Lie algebras."""

import torch


def sincu(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized sinc function: sin(x) / x"""
    return torch.special.sinc(x / x.new_tensor(torch.pi))


def versine_over_x(x: torch.Tensor) -> torch.Tensor:
    """Computes the versine function divided by x:
    versine(x) = 1 - cos(x)
    versine_over_x(x) = (1 - cos(x)) / x
    """
    return x.new_tensor(0.5) * x * torch.square(sincu(x / x.new_tensor(2.0)))
