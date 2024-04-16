from dataclasses import dataclass


@dataclass
class QuantizationScheme:
    weights_quantizaion_type: QuantizationType
    weights_quantization_granularity: QuantizationGranularity
    activations_quantizaion_type: QuantizationType
    activations_quantization_granularity: QuantizationGranularity
    smooth: bool
    clipping_strategy: Optional[ClippingStrategy]

qschemes = {
    "float8_e4m3_no_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=False,
        clipping_strategy=None
    ),
    "float8_e4m3_smooth_0": QuantizationScheme(
        weights_quantizaion_type=QuantizationType.SYMMETRIC,
        weights_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        activations_quantizaion_type=QuantizationType.SYMMETRIC,
        activations_quantization_granularity=QuantizationGranularity.PER_TENSOR,
        smooth=True,
        clipping_strategy=None
    ),
}