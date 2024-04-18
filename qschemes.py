from typing import Optional

from dataclasses import dataclass
from quantization import QuantizationType, QuantizationGranularity
from clipping import ClippingStrategy, PercentileClipping

@dataclass
class QuantizationScheme:
    weights_quantizaion_type: QuantizationType
    weights_quantization_granularity: QuantizationGranularity
    activations_quantizaion_type: QuantizationType
    activations_quantization_granularity: QuantizationGranularity
    smooth: bool
    clipping_strategy: Optional[ClippingStrategy]
    target_dtype: str

qschemes = {
    "float8_e4m3_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e4m3"
    ),
    "float8_e5m2_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e5m2"
    ),
    "int8_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="int8"
    ),
    "float8_e4m3_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e4m3"
    ),
    "float8_e5m2_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e5m2"
    ),
    "int8_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="int8"
    ),
    "float8_e4m3_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e4m3"
    ),
    "float8_e5m2_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="float8_e5m2"
    ),
    "int8_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None,
        target_dtype="int8"
    ),
    "float8_e4m3_smooth_0_percentile_clipping": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=PercentileClipping(name="percentile_clipping", p=0.98),
        target_dtype="float8_e4m3"
    ),
    "float8_e5m2_smooth_0_percentile_clipping": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=PercentileClipping(name="percentile_clipping", p=0.98),
        target_dtype="float8_e5m2"
    ),
    "int8_smooth_0_percentile_clipping": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=PercentileClipping(name="percentile_clipping", p=0.98),
        target_dtype="int8"
    ),
}