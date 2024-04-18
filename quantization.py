import os
from typing import Union, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from utils import _get_tensor_channel


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
    min_values: np.ndarray[torch.float16]
    max_values: np.ndarray[torch.float16]


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
    path_to_file: str, values_type: str
) -> Tuple[List[torch.float16]]:
    assert values_type in ["weights", "activations"]

    dim = 1
    values = torch.load(path_to_file)
    if values_type == "activations":
        values = values.squeeze()
        dim = 0

    return torch.min(values, dim).values.tolist(), torch.max(values, dim).values.tolist()


def get_statistics_from_files(path_to_files: str, layer_name: str) -> Statistics:
    stat = Statistics()
    for values_type in ["weights", "activations"]:
        for filename in os.listdir(path_to_files):
            if values_type in filename and layer_name in filename:
                min_stats, max_stats = _extract_perchannel_minmax_values(
                    os.path.join(path_to_files, filename), values_type
                )
                min_stats, max_stats = np.array([[val] for val in min_stats]), np.array(
                    [[val] for val in max_stats]
                )
                if not hasattr(stat, values_type):
                    setattr(
                        stat, values_type, DataMinMax(min_values=min_stats, max_values=max_stats)
                    )
                else:
                    getattr(stat, values_type).min_values = np.concatenate(
                        [getattr(stat, values_type).min_values, min_stats], axis=1
                    )
                    getattr(stat, values_type).max_values = np.concatenate(
                        [getattr(stat, values_type).max_values, max_stats], axis=1
                    )

    return stat


def prepare_quantization_params(
    statistics: Statistics,
    values_type: str = "weights",
    dtype: str = "fp8_e4m3",
    quantizaion_type: QuantizationType = QuantizationType.SYMMETRIC,
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR,
) -> Tuple:
    max_values = getattr(statistics, values_type).max_values
    min_values = getattr(statistics, values_type).min_values
    if quantizaion_type == QuantizationType.SYMMETRIC:
        if granularity == QuantizationGranularity.PER_TENSOR:
            scale = np.array(
                [
                    np.maximum(np.max(np.abs(max_values)), np.max(np.abs(min_values)))
                    / dtype_boundaries[dtype].qmax
                    for _ in range(max_values.shape[0])
                ]
            )
        else:
            scale = np.array(
                [
                    np.maximum(
                        np.max(np.abs(max_values[channel])), np.max(np.abs(min_values[channel]))
                    )
                    / dtype_boundaries[dtype].qmax
                    for channel in range(max_values.shape[0])
                ]
            )
        zp = np.zeros_like(scale)
    else:
        scale = np.array(
            [
                (np.max(max_values) - np.min(min_values))
                / (dtype_boundaries[dtype].qmax - dtype_boundaries[dtype].qmin)
            ]
            * len(getattr(statistics, values_type).max_values)
        )
        zp = -np.round(np.min(min_values) / scale) + dtype_boundaries[dtype].qmin

    return zp, scale


def fake_quantize(
    original_data: torch.Tensor,
    zp: Union[np.ndarray[np.float16], np.ndarray[np.int8]],
    scale: np.float16,
    values_type: str = "weights",
    qtype: str = "fp8_e4m3",
    granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR,
) -> torch.Tensor:
    if granularity == QuantizationGranularity.PER_TENSOR:
        if values_type == "weights":
            data = original_data.T
        else:
            data = original_data.squeeze()
        quantized_data = torch.clamp(
            data / scale + zp, min=dtype_boundaries[qtype].qmin, max=dtype_boundaries[qtype].qmax
        )
        if values_type == "weights":
            quantized_data = quantized_data.T
    else:
        dim = 0 if values_type == "weights" else 1
        quantized_data = torch.zeros(original_data.shape)
        for channel in range(original_data.size(dim)):
            # TODO: rewrite this awful "if-else"
            if dim == 0:
                quantized_data[:, channel] = torch.clamp(
                    original_data[:, channel] / scale + zp[channel],
                    min=dtype_boundaries[qtype].qmin,
                    max=dtype_boundaries[qtype].qmax,
                )
            else:
                quantized_data[channel, :] = torch.clamp(
                    original_data[channel, :] / scale + zp[channel],
                    min=dtype_boundaries[qtype].qmin,
                    max=dtype_boundaries[qtype].qmax,
                )

    return quantized_data


def smooth(
    original_weights: torch.Tensor,
    original_activations: torch.Tensor,
    stats: Statistics,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert original_weights.shape[1] == original_activations.T.shape[0]

    scale_coef = torch.max(
        torch.abs(stats.activations.min_values + stats.activations.max_values)
    ) ** alpha / torch.max(torch.abs(stats.weights.min_values + stats.weights.max_values)) ** (
        1 - alpha
    )
    scale_matrix = torch.diag(scale_coef)

    smoothed_weights = original_weights @ torch.linalg.inv(scale_matrix)
    smoothed_activaions = scale_matrix @ original_activations
    assert (
        smoothed_weights.shape == original_weights.shape
        and smoothed_activaions.shape == original_activations.shape
    )

    return smoothed_weights, smoothed_activaions


def calculate_loss(
    original_weights: torch.Tensor,
    original_activations: torch.Tensor,
    quantized_weights: torch.Tensor,
    quantized_activations: torch.Tensor,
) -> torch.float16:
    diff_tensor = (
        original_weights @ original_activations - quantized_weights @ quantized_activations
    )
    return torch.sqrt(torch.sum(diff_tensor * diff_tensor)).item()
