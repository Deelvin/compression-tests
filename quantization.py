import os
from typing import Union, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

@dataclass
class QuantizationDTypeBoundaries:
    qmin: Union[int, float]
    qmax: Union[int, float]

dtype_boundaries = {
    "float8_e4m3": QuantizationDTypeBoundaries(qmin=-448, qmax=448),
    "float8_e5m2": QuantizationDTypeBoundaries(qmin=-57344, qmax=57344),
    "int8": QuantizationDTypeBoundaries(qmin=-127, qmax=127),
}

@dataclass
class DataMinMax:
    min_values: List[torch.float16]
    max_values: List[torch.float16]

class Statistics:
    weights: DataMinMax
    activations: DataMinMax

class QuantizationType(Enum):
    SYMMETRIC = 1
    ASYMMETRIC = 2

class QuantizationGranularity(Enum):
    PER_TENSOR = 1
    PER_CHANNEL = 2

def _extract_perchannel_minmax_values(
    path_to_file: str, 
    values_type: str
) -> Tuple[List[torch.float16]]:
    assert values_type in ["weights", "activations"]

    dim = 0 if values_type == "weights" else 1
    values = torch.load(path_to_file)

    return torch.min(values, dim).values.tolist(), torch.max(values, dim).values.tolist()


def get_statistics_from_files(path_to_files: str, layer_name: str) -> Statistics:
    stat = Statistics()
    for values_type in ["weights", "activations"]:
        for filename in os.listdir(path_to_files):
            if values_type in filename and layer_name in filename:
                min_stats, max_stats = _extract_perchannel_minmax_values(
                    os.path.join(path_to_files, filename),
                    values_type
                )
                if not hasattr(stat, values_type):
                    setattr(
                        stat, 
                        values_type, 
                        DataMinMax(min_values=min_stats, max_values=max_stats)
                    )
                else:
                    getattr(
                        stat,
                        values_type
                    ).min_values.extend(min_stats)
                    getattr(
                        stat,
                        values_type
                    ).max_values.extend(max_stats)

    return stat
            

def prepare_quantization_params(
    statistics: Statistics,
    values_type: str = "weights",
    dtype: str = "fp8_e4m3",
    quantizaion_type: QuantizationType = QuantizationType.SYMMETRIC,
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR
) -> Tuple:
    max_values = np.array(getattr(statistics, values_type).max_values)
    min_values = np.array(getattr(statistics, values_type).min_values)
    zp_dtype = np.int8 if dtype == "int8" else np.float16
    if quantizaion_type == QuantizationType.SYMMETRIC:
        scale = np.maximum(
            np.abs(max_values), np.abs(min_values)
        ) / dtype_boundaries[dtype].qmax
        zp = np.zeros_like(scale, dtype=zp_dtype)
    else:
        scale = np.array([
            (np.max(max_values) - np.min(min_values)) / 
            (dtype_boundaries[dtype].qmax - dtype_boundaries[dtype].qmin)
        ] * len(getattr(statistics, values_type).max_values))
        zp = (-np.round(np.min(min_values) / scale) + dtype_boundaries[dtype].qmin).astype(zp_dtype)

    return zp, scale

def fake_quantize(
    original_data: torch.Tensor,
    zp: Union[np.ndarray[np.float16], np.ndarray[np.int8]],
    scale: np.float16,
    values_type: str = "weights",
    qtype: str = "fp8_e4m3",
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR
) -> torch.Tensor:
    if granularity == QuantizationGranularity.PER_TENSOR:
        quantized_data = torch.clamp(
            original_data / scale + zp,
            min=dtype_boundaries[qtype].qmin,
            max=dtype_boundaries[qtype].qmax
        )
    else:
        dim = 0 if values_type == "weights" else 1
        quantized_data = torch.zeros(original_data.shape)
        for channel in range(original.size(dim)):
            # TODO: rewrite this awful "if-else"
            if dim == 0:
                quantized_data[channel] = torch.clamp(
                    original_data[channel] / scale + zp[channel],
                    min=dtype_boundaries[qtype].qmin,
                    max=dtype_boundaries[qtype].qmax
                )
            else:
                quantized_data[:, channel] = torch.clamp(
                    original_data[:, channel] / scale + zp[channel],
                    min=dtype_boundaries[qtype].qmin,
                    max=dtype_boundaries[qtype].qmax
                )

    return quantized_data

def smooth(
    original_data: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    pass

def calculate_error(
    original_weights: torch.Tensor,
    original_activations: torch.Tensor,
    quantized_weights: torch.Tensor,
    quantized_activations: torch.Tensor
) -> torch.float16:
    diff_tensor = original_weights @ original_activations - quantized_weights @ quantized_activations
    return torch.sqrt(torch.sum(diff_tensor * diff_tensor)).item()