import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import LightSource

DUMMY_DATASET = [
    "The capital of Canada is",
    "2+2=?",
    "What is the capital of France?",
    "Who is the president of the USA?",
]


def collect_weights_and_activations(
    path_to_model: str,
    path_to_save_data: str,
    file_prefix: str = "mistral",
    prune_ratio: float = 0.0,
    filter_layers: str = "",
) -> None:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(path_to_model)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    inputs = DUMMY_DATASET.copy()

    print(model)

    model.eval()
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input.detach()

        return hook

    count_limit = int(1 / (1 - prune_ratio))
    current_count = 0
    for i, (name, module) in tqdm(enumerate(model.named_modules()), desc="Collecting weights"):
        if isinstance(module, torch.nn.Linear):
            if filter_layers in name:
                if current_count == 0:
                    module.register_forward_hook(get_activation(f"{name}"))
                    file_name = os.path.join(path_to_save_data, f"{file_prefix}_{name}")
                    torch.save(module.weight.data, f"{file_name}_weights.pt")
                current_count = (current_count + 1) % count_limit

    for input_str in tqdm(inputs, desc="Collecting activations"):
        inputs = tokenizer(input_str, return_tensors="pt")

        activations.clear()

        with torch.no_grad():
            model(**inputs)

        for i, (name, activation) in enumerate(activations.items()):
            file_name = os.path.join(
                path_to_save_data,
                f"{file_prefix}_{name}_{input_str.replace(' ', '_')}_activations.pt",
            )
            torch.save(activation, file_name)


def _get_tensor_channel(tensor: torch.Tensor, channel: int, dim: int) -> torch.Tensor:
    return tensor[channel, :] if dim == 0 else tensor[:, channel]


def _find_operand(path_to_model_data: str, first_operand_file: str) -> str:
    for filename in os.listdir(path_to_model_data):
        splitted_filename, splitted_first_operand_file = filename.split(
            "_"
        ), first_operand_file.split("_")
        if (
            splitted_filename[0:3] == splitted_first_operand_file[0:3]
            and filename != first_operand_file
            and filename.endswith("weights.pt")
            and first_operand_file.endswith("activations.pt")
            or filename.endswith("activations.pt")
            and first_operand_file.endswith("weights.pt")
        ):
            return os.path.join(path_to_model_data, filename)
    raise FileNotFoundError(f"No operand found for matmul with {first_operand_file}")


def plot_distribution(
    data: torch.Tensor, path_to_save_plot: str, layer_name: str, values_type: str = "weights", sample: str = None
) -> None:
    assert values_type in ["weights", "activations"]

    os.makedirs(path_to_save_plot, exist_ok=True)

    dim = 1 if values_type == "weights" else 0
    # 2D per-channel
    for channel in range(data.size(dim)):
        # TODO: use --prune parameter value to reduce number of graphs
        if values_type == "weights" and channel % 1024 == 0 or values_type == "activations":
            plt.figure(figsize=(10, 4))
            channel_values = _get_tensor_channel(data, channel, dim).squeeze()
            plt.hist(channel_values.numpy(), bins=50, color="skyblue", edgecolor="black")
            plt.title(f"{layer_name}, channel {channel} in Layer {layer_name}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plot_file_name = f"histogram_layer_{layer_name}_{values_type}_{sample if sample is not None else ''}_channel_{channel}.png"
            plot_file_path = os.path.join(path_to_save_plot, plot_file_name)
            plt.savefig(plot_file_path)
            plt.close()

    # TODO: Fix 3d plots
    # 3D for the whole tensor
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(data.shape[0])
    # y = np.arange(data.shape[1])
    # X, Y = np.meshgrid(x, y)
    # Z = data.flatten()
    # # scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='.')
    # ls = LightSource(270, 45)
    # rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(
    #     X, Y, Z,
    #     rstride=1,
    #     cstride=1,
    #     facecolors=rgb,
    #     linewidth=0,
    #     antialiased=False,
    #     shade=False
    # )
    # ax.set_xlabel('axis 1')
    # ax.set_ylabel('axis 0')
    # ax.set_zlabel('Frequency')
    # ax.set_title(f'3D Histogram of data in Layer {layer_name}')
    # ax.view_init(elev=20, azim=-60)
    # print("DEBUG3")
    # fig.colorbar(scatter, label='Value')

    # plot_file_name = f'3d_histogram_layer_{layer_name}.png'
    # plot_file_path = os.path.join(path_to_save_plot, plot_file_name)
    # plt.savefig(plot_file_path)
    # plt.close()


def plot_distributions_comparison(
    original_tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
    zp: np.ndarray,
    layer_name: str,
    path_to_save_plot: str,
    values_type: str = "weights"
) -> None:
    assert values_type in ["weights", "activations"]
    assert (
        original_tensor.shape == quantized_tensor.shape
    ), f"{original_tensor.shape} vs {quantized_tensor.shape}"

    dim = 0 if values_type == "weights" else 1
    num_channels = original_tensor.size(dim)

    if not os.path.exists(path_to_save_plot):
        os.makedirs(path_to_save_plot)

    for channel in range(num_channels):
        if channel % 1024 == 0:
            original_channel = _get_tensor_channel(original_tensor, channel, dim).flatten().numpy()
            quantized_channel = (
                _get_tensor_channel(quantized_tensor, channel, dim).flatten().numpy()
            )

            fig, axs = plt.subplots(2, 1, figsize=(10, 5))

            # TODO: plot unknown number of graphs (to compare different quantization schemas in a single plot)
            axs[0].hist(
                original_channel,
                bins="auto",
                color="blue",
                alpha=0.7,
                rwidth=0.85,
                label="Original",
            )

            axs[1].hist(
                quantized_channel,
                bins="auto",
                color="red",
                alpha=0.7,
                rwidth=0.85,
                label="Quantized",
            )

            axs[1].axvline(x=zp[channel], color="black", linestyle="--", label="Zero Point")

            plt.title(f"Original vs Quantized in {layer_name} comparison")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(path_to_save_plot, f"{layer_name}_{values_type}_channel_{channel}.png"))
            plt.close()

def plot_loss(qschemes: List[str], losses: List[np.float16], path_to_save_plot) -> None:
    plt.bar(qschemes, losses)
    plt.title(f"Loss of different quantization schemes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_plot, "loss.png"))
    plt.close()