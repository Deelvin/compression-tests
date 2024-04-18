from typing import NoReturn, Tuple, Dict
import shutil
import os
import argparse

import torch

from utils import collect_weights_and_activations, DUMMY_DATASET
from qschemes import qschemes
from experiment import SingleLayerQuantizationExperiment, SingleQuantizationSchemeExperiment

EPS = 1e-6


def prepare(args: argparse.ArgumentParser) -> Tuple[str, str]:
    pwd = os.path.dirname(__file__)
    path_to_model_data = os.path.join(pwd, "model_data")
    path_to_save_results = os.path.join(pwd, "results")

    if args.model_path.endswith("/"):
        args.model_path = args.model_path[:-1]
    args.model_name = args.model_path.split("/")[-1]

    os.makedirs(path_to_model_data, exist_ok=args.use_cached)
    os.makedirs(path_to_save_results, exist_ok=True)

    if not args.use_cached:
        try:
            collect_weights_and_activations(
                args.model_path,
                path_to_model_data,
                file_prefix=args.model_name,
                prune_ratio=args.prune_ratio,
                filter_layers=args.filter_layers,
            )
        except Exception as exc:
            shutil.rmtree(path_to_model_data)
            raise exc

    return path_to_model_data, path_to_save_results


def run(path_to_model_data: str, path_to_save_results: str, args: argparse.ArgumentParser) -> None:
    fp16_loss = SingleQuantizationSchemeExperiment(
        model_name=args.model_name,
        path_to_model_data=path_to_model_data,
        path_to_save_results=path_to_save_results,
        quantization_scheme=None,
        plot_distributions=True,
    ).run(verbose=True)

    assert fp16_loss < EPS

    qschemes_to_test = list(qschemes.keys())
    print(f"Running experiments with the following quantization schemes: {qschemes_to_test}")

    quantization_loss: Dict[str, float] = {}
    for experiment_id, scheme in enumerate(qschemes_to_test):
        print(f"{experiment_id + 1}) {scheme}")
        cur_scheme_loss = SingleQuantizationSchemeExperiment(
            model_name=args.model_name,
            path_to_model_data=path_to_model_data,
            path_to_save_results=path_to_save_results,
            quantization_scheme=qschemes[scheme],
            dump_quantized=True,
            plot_distributions=True,
            artifact_name=scheme,
        ).run(verbose=True)
        quantization_loss[scheme] = cur_scheme_loss

    print(quantization_loss)


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
