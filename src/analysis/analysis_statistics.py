# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch_cka
except ImportError:
    import sys
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "torch_cka"], check=False)

def format_layer_name(layer_name):
    mapping = {
        "features.0.conv": "1st conv",
        "features.1.conv": "2nd conv",
        "features.3.conv": "3rd conv",
        "features.4.conv": "4th conv",
        "features.0": "1st conv",
        "features.1": "2nd conv",
        "features.3": "3rd conv",
        "features.4": "4th conv",
    }
    if layer_name in mapping:
        return mapping[layer_name]
    import re
    match = re.search(r'features\.(\d+)', str(layer_name))
    if match:
        idx = match.group(1)
        idx_map = {'0': '1st conv', '1': '2nd conv', '3': '3rd conv', '4': '4th conv'}
        if idx in idx_map:
            return idx_map[idx]
    return layer_name

def plot_cka_matrix(cka_obj, title_suffix="", save_path=None):
    cka_results = cka_obj.export()
    matrix = cka_results['CKA']
    model1_layers = cka_results['model1_layers']
    model2_layers = cka_results['model2_layers']
    m1_name = cka_results.get('model1_name', 'Model 1')
    m2_name = cka_results.get('model2_name', 'Model 2')

    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().numpy()

    display_m1_layers = [format_layer_name(l) for l in model1_layers]
    display_m2_layers = [format_layer_name(l) for l in model2_layers]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(matrix, cmap='magma', origin='lower')

    ax.set_xticks(np.arange(len(model2_layers)))
    ax.set_yticks(np.arange(len(model1_layers)))
    ax.set_xticklabels(display_m2_layers, fontsize=10)
    ax.set_yticklabels(display_m1_layers, fontsize=10)
    
    ax.set_xlabel(m2_name, fontsize=11, labelpad=10)
    ax.set_ylabel(m1_name, fontsize=11, labelpad=10)
    
    title = f"CKA Matrix ({title_suffix})" if title_suffix else "Representational Similarity Matrix (CKA)"
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')

    min_val, max_val = matrix.min(), matrix.max()
    val_range = max_val - min_val

    for i in range(len(model1_layers)):       
        for j in range(len(model2_layers)):   
            score = matrix[i, j]
            norm_score = (score - min_val) / val_range if val_range > 1e-8 else 0.5
            text_color = "white" if norm_score < 0.5 else "black"
            ax.text(j, i, f"{score:.4f}", ha="center", va="center", 
                    color=text_color, fontweight="bold", fontsize=12)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_all_cka_pairs(cka_bf, cka_bs, cka_fs, cka_nb=None, cka_uc=None, save_dir=None):
    import os
    def _get_path(filename):
        return os.path.join(save_dir, filename) if save_dir else filename

    plot_cka_matrix(cka_bf, title_suffix="Baseline vs Fine-Tuned", save_path=_get_path("cka_baseline_vs_finetuned.png"))
    plot_cka_matrix(cka_bs, title_suffix="Baseline vs Scratch", save_path=_get_path("cka_baseline_vs_scratch.png"))
    plot_cka_matrix(cka_fs, title_suffix="Fine-Tuned vs Scratch", save_path=_get_path("cka_finetuned_vs_scratch.png"))
    if cka_nb is not None:
        plot_cka_matrix(cka_nb, title_suffix="Noise Baseline", save_path=_get_path("cka_noise_baseline.png"))
    if cka_uc is not None:
        plot_cka_matrix(cka_uc, title_suffix="Untrained Control of natural images", save_path=_get_path("cka_untrained_control.png"))


def _extract_conv_weights(model, layer_num, layer_name=None):
    import torch
    import torch.nn as nn

    # 1. Try direct indexing into model.features[layer_num]
    if hasattr(model, "features") and isinstance(layer_num, int):
        if 0 <= layer_num < len(model.features):
            target = model.features[layer_num]
            if hasattr(target, "conv") and hasattr(target.conv, "weight") and target.conv.weight is not None:
                return target.conv.weight.detach().cpu().numpy().flatten()
            elif hasattr(target, "weight") and target.weight is not None:
                return target.weight.detach().cpu().numpy().flatten()

    # 2. Collect all convolutional modules in model.features or model
    conv_modules = []
    if hasattr(model, "features"):
        for m in model.features:
            if hasattr(m, "conv") and hasattr(m.conv, "weight") and m.conv.weight is not None:
                conv_modules.append(m.conv)
            elif hasattr(m, "weight") and m.weight is not None:
                conv_modules.append(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.weight is not None:
                conv_modules.append(m)

    # 3. Handle ordinal indexing (0=1st conv, 1=2nd conv, 2=3rd conv, 3=4th conv)
    if isinstance(layer_num, int) and 0 <= layer_num < len(conv_modules):
        return conv_modules[layer_num].weight.detach().cpu().numpy().flatten()

    # 4. Handle string matching if layer_name provided
    if layer_name:
        clean = str(layer_name).lower().replace(" ", "").replace("_", "")
        for idx, m in enumerate(conv_modules):
            if f"{idx+1}" in clean or f"conv{idx+1}" in clean:
                return m.weight.detach().cpu().numpy().flatten()

    if conv_modules:
        return conv_modules[-1].weight.detach().cpu().numpy().flatten()

    raise ValueError(f"Could not extract weights for layer_num={layer_num}, layer_name={layer_name}")


def plot_layer_weight_diagnostic(models, model_names, layer_num, layer_name, threshold=0.001, save_path=None):
    colors = ['blue', 'orange', 'green']
    num_models = len(models)
    
    fig, axs = plt.subplots(1, num_models, figsize=(15, 4), sharey=True, dpi=150)
    
    if num_models == 1:
        axs = [axs]
        
    sparsity_info = []
    for i, (model, name) in enumerate(zip(models, model_names)):
        weights = _extract_conv_weights(model, layer_num, layer_name)
        sparsity = np.mean(np.abs(weights) < threshold) * 100
        sparsity_info.append((name, sparsity))
        
        axs[i].hist(weights, bins=100, color=colors[i % len(colors)], alpha=0.7)
        axs[i].set_xlabel("Weight Value")
        if i == 0:
            axs[i].set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.show()

    # Print summary after displaying plot (no titles on the figure itself)
    print("\n" + "=" * 76)
    print(f"📊 WEIGHT SPARSITY DIAGNOSTIC - {layer_name.upper()}")
    print("=" * 76)
    for name, sparsity in sparsity_info:
        print(f"  • {name:<20} | Sparsity (< {threshold}): {sparsity:.2f}%")
    print("=" * 76 + "\n")