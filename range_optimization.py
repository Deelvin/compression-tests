from enum import Enum
from typing import Callable, List
from abc import ABC, abstractmethod

import numpy as np
import torch
from ml_dtypes import float8_e4m3fn, float8_e5m2

from quantization import Statistics, QuantizationGranularity, dtype_boundaries

class ClippingStrategy(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def optimize_boundaries(
        self,
        values_type: str,
        weights: torch.Tensor,
        activations: torch.Tensor,
        stat: Statistics
    ) -> Statistics:
        ...

class KLDivergenceClipping(ClippingStrategy):
    def __init__(
        self, name: str,
        weights_only: bool,
        granularity: QuantizationGranularity
    ) -> None:
        self.name = name
        self.weights_only = weights_only
        self.granularity = granularity
    
    def optimize_boundaries(
        self,
        values_type: str,
        weights: torch.Tensor,
        activations: torch.Tensor,
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
        weights: torch.Tensor,
        activations: torch.Tensor,
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
    def __init__(self, name: str, loss_func: Callable) -> None:
        pass
    
    def optimize_boundaries(
        self,
        values_type: str,
        weights: torch.Tensor,
        activations: torch.Tensor,
        stat: Statistics
    ) -> Statistics:
        pass

class ScaleOptimizationStrategy:
    def __init__(self, name: str, quantization_scheme: str) -> None:
        self.name = name
        self.quantization_scheme
    
    def optimize_scale(
        self,
        tensor: torch.Tensor,
    ) -> List[np.float16]:
        ...
class GridScaleOptimizaion(ScaleOptimizationStrategy):
    def __init__(self, name: str, quantization_scheme: str) -> None:
        self.name = name
        self.quantization_scheme = quantization_scheme
        if "e4m3" in quantization_scheme:
            self.quantization_dtype = float8_e4m3fn
        elif "e5m2" in quantization_scheme:
            self.quantization_dtype = float8_e5m2
        else:
            self.quantization_dtype= np.int8

    def optimize_scale(
        self,
        tensor: torch.Tensor,
    ) -> List[np.float16]:
        def _calculate_loss(tensor_fp32: torch.Tensor, scale: np.float32) -> np.float32:
            tensor_fp8 = (tensor_fp32 / scale).astype(dtype=self.quantization_dtype)
            tensor_fp32_after_fp8 = tensor_fp8.astype(dtype=np.float32) * scale

            return np.sum(np.abs(tensor_fp32 - tensor_fp32_after_fp8))

        tensor = tensor.cpu().detach().numpy()
        scale_grid = np.linspace(0.01, 10, 15)
        losses = []
        for scale in scale_grid:
            losses.append(_calculate_loss(tensor, scale))
        # TODO: extend to all quantization schemes
        tensor = torch.Tensor(tensor).to("cuda")
        return torch.tensor(scale_grid[np.argmin(losses)]).to("cuda")