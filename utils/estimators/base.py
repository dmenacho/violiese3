from abc import ABC, abstractmethod
import torch

class BaseEstimator(ABC):
    def __init__(self, system):
        self.system = system

    @abstractmethod
    def predict_step(self, state_estimate: dict, u: torch.Tensor) -> dict:
        pass

