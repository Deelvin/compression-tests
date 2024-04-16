from quantization import QuantizationScheme

class SingleLayerQuantizationExperiment:
    def __init__(
        self,
        original_weights: torch.Tensor,
        original_activations: torch.Tensor,
        quantization_scheme: QuantizationScheme,
        layer_name: str
    ) -> None:
        assert original_weights.shape[1] == original_activations.T.shape[0]
        self.original_weights = original_weights
        self.original_activations = original_activations
        self.quantization_scheme = quantization_scheme'
    
    def quantize(self) -> torch.Tensor:
        pass
    
    def calculate_quantization_error(self) -> torch.float16:
        assert self.quantized_weights and self.quantized_activations, "Use quantize() method first"

        return calculate_error(
            self.original_weights, 
            self.original_activations, 
            self.quantized_weights, 
            self.quantized_activations
        )

class SingleQuantizationSchemeExperiment:
    def __init__(
        self,
        path_to_model_data: str,
        quantization_scheme: QuantizationScheme,
        dump_quantized: bool = True,
        plot_distributions: bool = True
    ) -> None:
        pass
    
    def run(self, verbose: bool = True) -> torch.float16:
        # TODO: perform quantization (dump quantized tensors if needed);
        #       plot 2D per-channel (and 3D possibly per-tensor) distributions;
        #       plot distribution of quantization errors depending on layer;
        #       calculate total quantization error and return it
        pass

