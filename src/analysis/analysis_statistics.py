# %% [code]
# %% [code]
# %% [code]
import matplotlib.pyplot as plt
import numpy as np

import subprocess
subprocess.run(["pip", "install", "torch_cka"], check=True)

def plot_cka_matrix(cka_obj, title_suffix=""):
    cka_results = cka_obj.export()
    matrix = cka_results['CKA']
    model1_layers = cka_results['model1_layers']
    model2_layers = cka_results['model2_layers']
    m1_name = cka_results.get('model1_name', 'Model 1')
    m2_name = cka_results.get('model2_name', 'Model 2')

    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(matrix, cmap='magma', origin='lower')

    ax.set_xticks(np.arange(len(model2_layers)))
    ax.set_yticks(np.arange(len(model1_layers)))
    ax.set_xticklabels(model2_layers, fontsize=10)
    ax.set_yticklabels(model1_layers, fontsize=10)
    
    ax.set_xlabel(m2_name, fontsize=11, labelpad=10)
    ax.set_ylabel(m1_name, fontsize=11, labelpad=10)
    
    title = f"CKA Matrix ({title_suffix})" if title_suffix else "Representational Similarity Matrix (CKA)"
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')

    for i in range(len(model1_layers)):       
        for j in range(len(model2_layers)):   
            score = matrix[i, j]
            text_color = "white" if score < 0.67 else "black"
            ax.text(j, i, f"{score:.4f}", ha="center", va="center", 
                    color=text_color, fontweight="bold", fontsize=12)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()

def plot_all_cka_pairs(cka_bf, cka_bs, cka_fs, cka_nb=None, cka_uc=None):
    plot_cka_matrix(cka_bf, title_suffix="Baseline vs Fine-Tuned")
    plot_cka_matrix(cka_bs, title_suffix="Baseline vs Scratch")
    plot_cka_matrix(cka_fs, title_suffix="Fine-Tuned vs Scratch")
    if cka_nb is not None:
        plot_cka_matrix(cka_nb, title_suffix="Noise Baseline")
    if cka_uc is not None:
        plot_cka_matrix(cka_uc, title_suffix="Untrained Control of natural images")


def plot_layer_weight_diagnostic(models, model_names, layer_num, layer_name, threshold=0.001):
    colors = ['blue', 'orange', 'green']
    num_models = len(models)
    
    fig, axs = plt.subplots(1, num_models, figsize=(15, 4), sharey=True, dpi=150)
    
    if num_models == 1:
        axs = [axs]
        
    for i, (model, name) in enumerate(zip(models, model_names)):
        weights = model.features[layer_num].conv.weight.detach().cpu().numpy().flatten()
        sparsity = np.mean(np.abs(weights) < threshold) * 100
        
        axs[i].hist(weights, bins=100, color=colors[i % len(colors)], alpha=0.7)
        axs[i].set_title(f"{name}\nSparsity (<{threshold}): {sparsity:.2f}%", fontsize=10)
        axs[i].set_xlabel("Weight Value")
        if i == 0:
            axs[i].set_ylabel("Count")
            
    plt.suptitle(f"Weight Sparsity Diagnostic - {layer_name.upper()}", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()