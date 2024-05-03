from typing import Optional

from dataclasses import dataclass
from quantization import QuantizationType, QuantizationGranularity
from range_optimization import (
    ClippingStrategy,
    PercentileClipping,
    ScaleOptimizationStrategy,
    GridScaleOptimizaion,
)


@dataclass
class QuantizationScheme:
    weights_quantizaion_type: QuantizationType
    weights_quantization_granularity: QuantizationGranularity
    activations_quantizaion_type: QuantizationType
    activations_quantization_granularity: QuantizationGranularity
    smooth: bool
    clipping_strategy: Optional[ClippingStrategy]
    scale_optimization: Optional[ScaleOptimizationStrategy]
    target_dtype: str


qschemes = {
    "float8_e4m3_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        scale_optimization=None,
        target_dtype="float8_e4m3",
    ),
    "float8_e4m3_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=True,
        clipping_strategy=None,
        scale_optimization=None,
        target_dtype="float8_e4m3",
    ),
    "float8_e4m3_no_smooth_0_optimized_scale": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        scale_optimization=GridScaleOptimizaion("grid", "float8_e4m3"),
        target_dtype="float8_e4m3",
    ),
    "float8_e4m3_smooth_0_optimized_scale": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=True,
        clipping_strategy=None,
        scale_optimization=GridScaleOptimizaion("grid", "float8_e4m3"),
        target_dtype="float8_e4m3",
    ),
    "int8_no_smooth_2": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.ASYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        activations_quantizaion_type=QuantizationType.ASYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        smooth=False,
        clipping_strategy=None,
        scale_optimization=None,
        target_dtype="int8",
    ),
    "int8_smooth_2": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.ASYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        activations_quantizaion_type=QuantizationType.ASYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_CHANNEL,
        smooth=True,
        clipping_strategy=None,
        scale_optimization=None,
        target_dtype="int8",
    ),
}
