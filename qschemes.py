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
    ...
}