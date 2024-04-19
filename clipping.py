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
        values_type: str,
        stat: Statistics
    ) -> Statistics:
        ...

class KLDivergenceClipping(ClippingStrategy):
    def __init__(self, name: str, threshold: float) -> None:
        pass
    
    def optimize_boundaries(
        self,
        values_type: str,
        stat: Statistics
    ) -> Statistics:
        pass

class PercentileClipping(ClippingStrategy):
    def __init__(self, name: str, p: float) -> None:
        self.name = name
        self.p = p
    
    def optimize_boundaries(
        self,
        values_type: str,
        stat: Statistics
    ) -> Statistics:
        clipped_stat = stat
        updated_chosen_stat = getattr(clipped_stat, values_type)
        for channel in range(len(updated_chosen_stat.min_vlaues)):
            updated_chosen_stat.min_values = np.percentile(updated_chosen_stat.min_values, self.p)
            updated_chosen_stat.max_values = np.percentile(updated_chosen_stat.max_values, self.p)
        setattr(clipped_stat, values_type, updated_chosen_stat)

        return clipped_stat

class LossOptimizationClipping(ClippingStrategy):
    def __init__(self, name: str, objective: Callable) -> None:
        pass
    
    def optimize_boundaries(
        self,
        values_type: str,
        stat: Statistics
    ) -> Statistics:
        pass