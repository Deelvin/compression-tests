from enum import Enum
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import torch

from quantization import Statistics

class ClippingStrategy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def optimize_boundaries(
        self, 
        weights: torch.Tensor, 
        activations: torch.Tensor, 
        stat: Statistics
    ) -> Statistics:
        ...

class KLDivergenceClipping(ClippingStrategy):
    def __init__(self, name: str, threshold: float) -> None:
        pass
    
    def optimize_boundaries(
        self, 
        weights: torch.Tensor, 
        activations: torch.Tensor, 
        stat: Statistics
    ) -> Statistics:
        pass

class PercentileClippingClipping(ClippingStrategy):
    def __init__(self, name: str, p: float) -> None:
        pass
    
    def optimize_boundaries(
        self, 
        weights: torch.Tensor, 
        activations: torch.Tensor, 
        stat: Statistics
    ) -> Statistics:
        pass

class LossOptimizationClipping(ClippingStrategy):
    def __init__(self, name: str, objective: Callable) -> None:
        pass
    
    def optimize_boundaries(
        self, 
        weights: torch.Tensor, 
        activations: torch.Tensor, 
        stat: Statistics
    ) -> Statistics:
        pass