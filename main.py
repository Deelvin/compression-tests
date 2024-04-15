from typing import NoReturn, Tuple
import shutil
import os
import argparse

import torch

from quantization import (
    prepare_quantization_params,
    get_statistics_from_files,
    fake_quantize, 
    dequantize
)
from utils import (
    collect_weights_and_activations, 
    plot_distribution,
    plot_distributions_comparison,
    DUMMY_DATASET
)

def prepare(args: argparse.ArgumentParser) -> Tuple[str, str]:
    pwd = os.path.dirname(__file__)
    path_to_model_data = os.path.join(pwd, "model_data")
    path_to_results = os.path.join(pwd, "results")

    if args.model_path.endswith('/'):
        args.model_path = args.model_path[:-1]
    args.model_name = args.model_path.split('/')[-1]

    os.makedirs(path_to_model_data, exist_ok=args.use_cached)
    os.makedirs(path_to_results, exist_ok=True)

    if not args.use_cached:
        try:
            collect_weights_and_activations(
                args.model_path, 
                path_to_model_data, 
                file_prefix=args.model_name,
                prune_ratio=args.prune_ratio,
                filter_layers=args.filter_layers
            )
        except Exception as exc:
            shutil.rmtree(path_to_model_data)
            raise exc

    return path_to_model_data, path_to_results


def run(path_to_model_data: str, path_to_results: str, args: argparse.ArgumentParser) -> None:
    # original fp16
    path_to_fp16_results = os.path.join(path_to_results, "fp16")
    os.makedirs(path_to_fp16_results, exist_ok=True)

    weights_files = []
    activations_files = []
    for filename in os.listdir(path_to_model_data):
        if filename.startswith(args.model_name):
            if filename.endswith("_weights.pt"):
                weights_files.append(filename)
            else:
                activations_files.append(filename)

    for weights_file in weights_files:
        weights_path = os.path.join(path_to_model_data, weights_file)
        weights = torch.load(weights_path)
        layer_name = weights_file.replace(f'{args.model_name}_', '').replace('.pt', '')
        plot_distribution(weights, path_to_fp16_results, layer_name, values_type="weights")
    
    for activations_file in activations_files:
        activations_path = os.path.join(path_to_model_data, activations_file)
        activations = torch.load(activations_path)
        layer_name = activations_file.replace(f'{args.model_name}_', '').replace('.pt', '')
        plot_distribution(activations, path_to_fp16_results, layer_name, values_type="activations")

    # quantized: fp8 (e4m3, e5m2), int8
    for target_dtype in ["float8_e4m3", "float8_e5m2", "int8"]:
        path_to_quantized_results = os.path.join(path_to_results, target_dtype)
        os.makedirs(path_to_quantized_results, exist_ok=True)

        for fp16_weights_file in weights_files:
            original_weights_path = os.path.join(path_to_model_data, fp16_weights_file)
            original_weights = torch.load(original_weights_path)
            layer_name = fp16_weights_file.replace(f'{args.model_name}_', '').replace('.pt', '')
            stats = get_statistics_from_files(path_to_model_data, layer_name)
            zp, scale = prepare_quantization_params(
                stats, 
                values_type="weights",
                dtype=target_dtype,
            )
            quantized_weights = fake_quantize(
                original_weights, 
                zp, scale,
                values_type="weights", 
                qtype=target_dtype
            )
            torch.save(
                quantized_weights, 
                os.path.join(path_to_quantized_results, layer_name) + ".pt"
            )
            plot_distributions_comparison(
                original_weights, 
                quantized_weights, 
                zp,
                layer_name,
                path_to_quantized_results, 
                values_type="weights"
            )
        
        for fp16_activations_file in activations_files:
            original_activations_path = os.path.join(path_to_model_data, fp16_activations_file)
            original_activations = torch.load(original_activations_path)
            layer_name = fp16_activations_file.replace(f'{args.model_name}_', '').replace('.pt', '')
            stats = get_statistics_from_files(path_to_model_data, layer_name)
            zp, scale = prepare_quantization_params(
                stats, 
                values_type="activations",
                dtype=target_dtype,
            )
            quantized_activations = fake_quantize(
                original_activations, 
                zp, scale,
                values_type="activations", 
                qtype=target_dtype
            )
            torch.save(
                quantized_activations, 
                os.path.join(path_to_quantized_results, layer_name) + ".pt"
            )
            plot_distributions_comparison(
                original_activations, 
                quantized_activations, 
                zp,
                layer_name,
                path_to_quantized_results, 
                values_type="activations"
            )


def clean(path_to_model_data: str, args: argparse.ArgumentParser) -> None:
    pass

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--use-cached", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--auto-clip", choices=["percentile", "mse", "kl"])
    parser.add_argument("--prune-ratio", type=float, default=0.0)
    parser.add_argument("--filter-layers", type=str, default="")
    args = parser.parse_args()

    model_data_dir, results_dir = prepare(args)

    run(model_data_dir, results_dir, args)

    if args.clean:
        clean(data_dir, args)
    
    print("Done")

if __name__ == "__main__":
    main()