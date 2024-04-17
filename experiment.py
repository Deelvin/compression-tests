import os

import torch
import numpy

from tqdm import tqdm

from quantization import (
    prepare_quantization_params,
    get_statistics_from_files,
    fake_quantize,
)
from qschemes import QuantizationScheme
from utils import plot_distribution, plot_distributions_comparison

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
        self.quantization_scheme = quantization_scheme
    
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
        model_name: str,
        path_to_model_data: str,
        path_to_save_results: str,
        quantization_scheme: Optional[QuantizationScheme] = None,
        dump_quantized: bool = True,
        plot_distributions: bool = True,
        artifact_suffix: str = ""
    ) -> None:

        self.model_name = model_name
        self.path_to_model_data = path_to_model_data
        self.path_to_save_results = path_to_save_results
        self.dump_quantized = dump_quantized
        self.quantization_scheme = quantization_scheme
        self.plot_distributions = plot_distributions
        self.artifact_suffix = artifact_suffix

        self.path_to_fp16_results = os.path.join(path_to_save_results, "fp16")
        os.makedirs(self.path_to_fp16_results, exist_ok=True)

        self.weights_files = []
        self.activations_files = []

        for filename in os.listdir(self.path_to_model_data):
            if filename.startswith(self.model_name):
                if filename.endswith("_weights.pt"):
                    self.weights_files.append(filename)
                else:
                    self.activations_files.append(filename)

    def run(self, verbose: bool = True) -> torch.float16:
        disable_tqdm = not verbose
        for (values_type, files) in zip(["weights", "activations"], [self.weights_files, self.activations_files]):
            if self.quantization_scheme:
                for data_file in tqdm(
                    files, 
                    disable=disable_tqdm, 
                    desc=f"Quantizing {values_type} of each Linear layer of {self.model_name}..."
                ):
                    for target_dtype in ["float8_e4m3", "float8_e5m2", "int8"]:
                        path_to_quantized_results = os.path.join(self.path_to_save_results, target_dtype + self.artifact_suffix)
                        os.makedirs(path_to_quantized_results, exist_ok=True)
                        stats = get_statistics_from_files(self.path_to_model_data, layer_name)
                        zp, scale = prepare_quantization_params(
                            stats, 
                            values_type=values_type,
                            dtype=target_dtype,
                        )
                        quantized_data = fake_quantize(
                            data, 
                            zp, scale,
                            values_type=values_type, 
                            qtype=target_dtype
                        )
                        if self.dump_quantized:
                            torch.save(
                                quantized_data, 
                                os.path.join(path_to_quantized_results, layer_name) + ".pt"
                            )
                        if self.plot_distributions:
                            plot_distributions_comparison(
                                data, 
                                quantized_data, 
                                zp,
                                layer_name,
                                path_to_quantized_results, 
                                values_type=values_type
                            )
            else:
                for data_file in tqdm(
                    files, 
                    disable=disable_tqdm, 
                    desc=f"Processing {values_type} of each Linear layer of {self.model_name} inf fp16..."
                ):
                    data_path = os.path.join(self.path_to_model_data, data_file)
                    data = torch.load(data_path).squeeze()
                    layer_name = data_file.replace(f'{self.model_name}_', '').replace('.pt', '')
                    if self.plot_distributions:
                        plot_distribution(data, self.path_to_fp16_results, layer_name, values_type=values_type)
            

        # TODO: plot distribution of quantization errors depending on layer;
        #       calculate total quantization error and return it
        return None