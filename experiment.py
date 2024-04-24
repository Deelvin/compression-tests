import os
import gc
from typing import Optional

import torch
import numpy as np

from tqdm import tqdm

from quantization import (
    prepare_quantization_params,
    get_statistics_from_files,
    fake_quantize,
    calculate_loss,
    smooth
)
from qschemes import QuantizationScheme
from utils import (
    plot_distribution, 
    plot_distributions_comparison,
    _find_operand
)

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
    
    def calculate_quantization_loss(self) -> torch.float16:
        assert self.quantized_weights and self.quantized_activations, "Use quantize() method first"

        return calculate_loss(
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
        artifact_name: str = ""
    ) -> None:

        self.model_name = model_name
        self.path_to_model_data = path_to_model_data
        self.path_to_save_results = path_to_save_results
        self.dump_quantized = dump_quantized
        self.quantization_scheme = quantization_scheme
        self.plot_distributions = plot_distributions
        self.artifact_name = artifact_name

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
        
        self.per_layer_loss = []

    def run(self, verbose: bool = True) -> torch.float16:
        disable_tqdm = not verbose
        if self.quantization_scheme:
            for rhs_file in tqdm(
                self.activations_files, 
                disable=disable_tqdm, 
                desc=f"Quantizing weights and activations of each Linear layer of {self.model_name}..."
            ):
                rhs_path = os.path.join(self.path_to_model_data, rhs_file)
                rhs = torch.load(rhs_path).squeeze().to("cuda")
                lhs_path = _find_operand(self.path_to_model_data, rhs_file)
                lhs_file = lhs_path.split('/')[-1]
                lhs = torch.load(lhs_path).to("cuda")
                layer_name = os.path.commonprefix([lhs_file, rhs_file]).replace(f"{self.model_name}_", '')
                path_to_quantized_results = os.path.join(self.path_to_save_results, self.artifact_name)
                os.makedirs(path_to_quantized_results, exist_ok=True)
                stats = get_statistics_from_files(self.path_to_model_data, layer_name)
                if self.quantization_scheme.smooth:
                    original_lhs, original_rhs = torch.clone(lhs), torch.clone(rhs)
                    lhs, rhs = smooth(lhs, rhs, stats)
                    torch.save(lhs, lhs_path), torch.save(rhs, rhs_path)
                    stats = get_statistics_from_files(self.path_to_model_data, layer_name)
                if self.quantization_scheme.clipping_strategy:
                    for values_type in ["activations", "weights"]:
                        stats = self.quantization_scheme.clipping_strategy.optimize_boundaries(
                            values_type, stats
                        )
                rhs_zp, rhs_scale = prepare_quantization_params(
                    stats, 
                    values_type="activations",
                    dtype=self.quantization_scheme.target_dtype,
                )
                # TODO: add scale optimization for activations
                lhs_zp, lhs_scale = prepare_quantization_params(
                    stats, 
                    values_type="weights",
                    dtype=self.quantization_scheme.target_dtype,
                )
                if self.quantization_scheme.scale_optimization is not None:
                    lhs_scale = self.quantization_scheme.scale_optimization.optimize_scale(lhs)
                quantized_rhs = fake_quantize(
                    rhs, 
                    rhs_zp, rhs_scale,
                    values_type="activations", 
                    qtype=self.quantization_scheme.target_dtype
                )
                quantized_lhs = fake_quantize(
                    lhs, 
                    lhs_zp, lhs_scale,
                    values_type="weights", 
                    qtype=self.quantization_scheme.target_dtype
                )
                if self.dump_quantized:
                    dump_path = os.path.join(path_to_quantized_results, layer_name)
                    torch.save(
                        quantized_rhs, 
                        dump_path + "_activations.pt"
                    )
                    if not os.path.exists(dump_path + "_weights.pt"):
                        torch.save(
                            quantized_rhs, 
                            dump_path + "_weights.pt"
                        )
                if self.plot_distributions:
                    plot_distributions_comparison(
                        rhs, 
                        quantized_rhs, 
                        rhs_zp,
                        layer_name,
                        path_to_quantized_results, 
                        values_type="activations"
                    )
                    plot_distributions_comparison(
                        lhs, 
                        quantized_lhs, 
                        lhs_zp,
                        layer_name,
                        path_to_quantized_results, 
                        values_type="weights"
                    )
                if self.quantization_scheme.smooth:
                    torch.save(original_lhs, lhs_path), torch.save(original_rhs, rhs_path)
                self.per_layer_loss.append(calculate_loss(
                    rhs, lhs, quantized_rhs, quantized_lhs
                ))
                gc.collect()
        else:
            for rhs_file in tqdm(
                self.activations_files, 
                disable=disable_tqdm, 
                desc=f"Processing weigths and activations of each Linear layer of {self.model_name} in fp16..."
            ):
                rhs_path = os.path.join(self.path_to_model_data, rhs_file)
                rhs = torch.load(rhs_path).squeeze()
                lhs_path = _find_operand(self.path_to_model_data, rhs_file)
                lhs_file = lhs_path.split('/')[-1]
                lhs = torch.load(lhs_path)
                layer_name = os.path.commonprefix([lhs_file, rhs_file]).replace(f"{self.model_name}_", '')
                sample = rhs_file.replace(os.path.commonprefix([lhs_file, rhs_file]), "").replace(f"{self.model_name}_", "").replace(f"_activations.pt", "")
                if self.plot_distributions:
                    plot_distribution(rhs, self.path_to_fp16_results, layer_name, values_type="activations", sample=sample)
                    plot_distribution(lhs, self.path_to_fp16_results, layer_name, values_type="weights")
                self.per_layer_loss.append(calculate_loss(
                    rhs, lhs, rhs, lhs
                ))
                gc.collect()

        # TODO: plot distribution of quantization loss depending on layer;
        return np.sum(self.per_layer_loss)