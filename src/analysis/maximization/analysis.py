# %% [code]
# %% [code]
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


# analyzing SIMM, mse, mae
def analyze_layer_drift(patterns_a, patterns_b, layer_name="Layer", pair_labels=("Pattern A", "Pattern B")):
    common_indices = sorted(list(set(patterns_a.keys()).intersection(set(patterns_b.keys()))))
    
    results = {}
    mse_list, mae_list, ssim_list = [], [], []
    
    for idx in common_indices:
        tensor_a = patterns_a[idx]
        tensor_b = patterns_b[idx]
        
        # Bilinear interpolation fallback to handle cross-domain resolution mismatches
        if tensor_a.shape != tensor_b.shape:
            tensor_b = F.interpolate(
                tensor_b.unsqueeze(0), 
                size=(tensor_a.shape[1], tensor_a.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
        img_a_rgb = tensor_a.permute(1, 2, 0).numpy()
        img_b_rgb = tensor_b.permute(1, 2, 0).numpy()
        
        img_a = 0.2989 * img_a_rgb[:,:,0] + 0.5870 * img_a_rgb[:,:,1] + 0.1140 * img_a_rgb[:,:,2]
        img_b = 0.2989 * img_b_rgb[:,:,0] + 0.5870 * img_b_rgb[:,:,1] + 0.1140 * img_b_rgb[:,:,2]
        
        mse = np.mean((img_a - img_b) ** 2)
        mae = np.mean(np.abs(img_a - img_b))
        score = ssim(img_a, img_b, data_range=1.0)
        
        results[idx] = {
            'mse': mse,
            'mae': mae,
            'ssim': score
        }
        
        mse_list.append(mse)
        mae_list.append(mae)
        ssim_list.append(score)

    summary_stats = {
        'avg_mse': np.mean(mse_list),
        'avg_mae': np.mean(mae_list),
        'avg_ssim': np.mean(ssim_list),
        'most_drifted_filter': common_indices[np.argmin(ssim_list)],
        'least_drifted_filter': common_indices[np.argmax(ssim_list)]
    }
    
    print(f"\n======== STATISTICAL ANALYSIS FOR {layer_name.upper()} ({pair_labels[0]} vs {pair_labels[1]}) ========")
    print(f"Total Filters Evaluated : {len(common_indices)}")
    print(f"Layer Average SSIM      : {summary_stats['avg_ssim']:.4f}")
    print(f"Layer Average MSE       : {summary_stats['avg_mse']:.6f}")
    print(f"Layer Average MAE       : {summary_stats['avg_mae']:.6f}")
    print(f"Most Drifted Filter     : Index {summary_stats['most_drifted_filter']} (SSIM: {min(ssim_list):.4f})")
    print(f"Least Drifted Filter    : Index {summary_stats['least_drifted_filter']} (SSIM: {max(ssim_list):.4f})")
    print("=====================================================")
    
    return results, summary_stats


def analyze_all_pairs_drift(nat_patterns, ft_patterns, scr_patterns, layer_name="Layer"):
    print(f"\n>>> EXECUTING THREE-WAY PAIRWISE STRUCTURAL RUN FOR: {layer_name.upper()} <<<")
    
    nat_vs_ft_res, nat_vs_ft_sum = analyze_layer_drift(
        nat_patterns, ft_patterns, layer_name=layer_name, pair_labels=("Natural", "Fine-Tuned")
    )
    
    nat_vs_scr_res, nat_vs_scr_sum = analyze_layer_drift(
        nat_patterns, scr_patterns, layer_name=layer_name, pair_labels=("Natural", "Scratch")
    )
    
    ft_vs_scr_res, ft_vs_scr_sum = analyze_layer_drift(
        ft_patterns, scr_patterns, layer_name=layer_name, pair_labels=("Fine-Tuned", "Scratch")
    )
    
    all_results = {
        'nat_vs_ft': (nat_vs_ft_res, nat_vs_ft_sum),
        'nat_vs_scr': (nat_vs_scr_res, nat_vs_scr_sum),
        'ft_vs_scr': (ft_vs_scr_res, ft_vs_scr_sum)
    }
    
    return all_results

#analyze the absolute difference masks
def plot_difference_grid_pairwise(patterns_a, patterns_b, title_suffix, nrow=8, padding=4):
    common_indices = sorted(list(set(patterns_a.keys()).intersection(set(patterns_b.keys()))))
    
    diff_tensors = []
    for idx in common_indices:
        tensor_a = patterns_a[idx]
        tensor_b = patterns_b[idx]
        
        if tensor_a.shape != tensor_b.shape:
            tensor_b = F.interpolate(
                tensor_b.unsqueeze(0), 
                size=(tensor_a.shape[1], tensor_a.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
        # Convert both tensors to grayscale first to isolate luminance structure
        gray_a = 0.2989 * tensor_a[0] + 0.5870 * tensor_a[1] + 0.1140 * tensor_a[2]
        gray_b = 0.2989 * tensor_b[0] + 0.5870 * tensor_b[1] + 0.1140 * tensor_b[2]
        
        diff_gray = torch.abs(gray_a - gray_b)
        diff_tensors.append(diff_gray.unsqueeze(0))
        
    batch_tensor = torch.stack(diff_tensors, dim=0)
    grid = vutils.make_grid(batch_tensor, nrow=nrow, padding=padding, normalize=False)
    grid_np = grid[0].cpu().numpy()
    
    plt.figure(figsize=(14, 10), dpi=200)
    im = plt.imshow(grid_np, cmap='hot', vmin=0.0, vmax=0.5)
    
    plt.title(f"Absolute Structural Difference Masks - {title_suffix.upper()}", fontsize=12, fontweight='bold', pad=15)
    plt.axis('off')
    plt.colorbar(im, shrink=0.6, label='Magnitude of Structural Representation Change')
    plt.tight_layout()
    plt.show()

def compare_all_domain_states(nat_patterns, ft_patterns, scr_patterns, layer_name="Layer"):
    # Comparison 1: Natural vs Fine-Tuned
    plot_difference_grid_pairwise(
        nat_patterns, 
        ft_patterns, 
        title_suffix=f"{layer_name} (Natural vs Fine-Tuned)"
    )
    
    # Comparison 2: Natural vs Screen-from-Scratch
    plot_difference_grid_pairwise(
        nat_patterns, 
        scr_patterns, 
        title_suffix=f"{layer_name} (Natural vs Screen Scratch)"
    )
    
    # Comparison 3: Fine-Tuned vs Screen-from-Scratch
    plot_difference_grid_pairwise(
        ft_patterns, 
        scr_patterns, 
        title_suffix=f"{layer_name} (Fine-Tuned vs Screen Scratch)"
    )

#  Fourier analysis & Normalized Difference Anisotropy Index (NDI)
def _step1_preprocess(image_data):
    """
    Step 1: Convert image to grayscale and crop to an 80x80 central square.
    This guarantees equal grid dimensions for balanced 2D FFT frequency sampling.
    """
    if torch.is_tensor(image_data):
        image_data = image_data.detach().cpu().numpy()
        
    if image_data.ndim == 3:
        if image_data.shape[0] == 3:  # CHW format
            gray = 0.2989 * image_data[0] + 0.5870 * image_data[1] + 0.1140 * image_data[2]
        else:                         # HWC format
            gray = 0.2989 * image_data[:, :, 0] + 0.5870 * image_data[:, :, 1] + 0.1140 * image_data[:, :, 2]
    else:
        gray = image_data

    # Crop to central square of size min(h, w)
    h, w = gray.shape
    size = min(h, w)
    dy = (h - size) // 2
    dx = (w - size) // 2
    return gray[dy:dy+size, dx:dx+size]

def compute_fourier_spectrum(image_gray, dc_radius=5):
    """
    Step 2: Apply a circular spatial mask, compute 2D FFT, and calculate
    Normalized Difference Index (NDI) across cardinal and oblique sectors.
    """
    h, w = image_gray.shape
    cy, cx = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) / 2.0
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # 1. Circular spatial crop
    circular_mask = dist_from_center <= radius
    masked_image = image_gray * circular_mask
    
    # 2. 2D Fast Fourier Transform
    f_transform = np.fft.ifftshift(masked_image)
    f_transform = np.fft.fft2(f_transform)
    f_shift = np.fft.fftshift(f_transform)
    
    magnitude = np.abs(f_shift)
    log_magnitude = np.log(1 + magnitude)
    
    # 3. Frequency sector masks (exclude DC center component)
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = dist_from_center
    
    mask_dc = R > dc_radius
    mask_cardinal = ((theta < 10) | (theta > 170) | ((theta > 80) & (theta < 100))) & mask_dc
    mask_oblique = (((theta > 35) & (theta < 55)) | ((theta > 125) & (theta < 145))) & mask_dc
    
    cardinal_energy = np.sum(magnitude[mask_cardinal])
    oblique_energy = np.sum(magnitude[mask_oblique])
    
    total_energy = cardinal_energy + oblique_energy
    if total_energy == 0:
        ndi = 0.0
    else:
        ndi = (cardinal_energy - oblique_energy) / total_energy
        
    return magnitude, log_magnitude, ndi

def compute_rotational_anisotropy(image_data, dc_radius=5):
    """
    Preprocesses image data (grayscale + square crop) and computes Fourier NDI metrics.
    """
    image_gray = _step1_preprocess(image_data)
    _, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius)
    return log_mag, ndi

def plot_fourier_diagnostic(image_data, title="Fourier Diagnostic", dc_radius=5, log_scale=False, figsize=(18, 4.5)):
    """
    Generates a 4-panel visual diagnostic figure:
    1. Spatial Image (Circular Masked)
    2. Log Magnitude Fourier Spectrum
    3. Cardinal (Red) vs Oblique (Blue) Sector Overlay
    4. 1D Angular Energy Distribution Plot across 0°, 45°, 90°, 135°, 180° (with optional log_scale)
    """
    image_gray = _step1_preprocess(image_data)
    magnitude, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius)
    
    h, w = image_gray.shape
    cy, cx = h // 2, w // 2
    
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
    
    mask_dc = R > dc_radius
    mask_cardinal = ((theta < 10) | (theta > 170) | ((theta > 80) & (theta < 100))) & mask_dc
    mask_oblique = (((theta > 35) & (theta < 55)) | ((theta > 125) & (theta < 145))) & mask_dc
    
    # Calculate 1D Angular Energy Profile
    angles = np.arange(0, 180, 2)
    angular_energy = []
    for a in angles:
        bin_mask = (np.abs(theta - a) <= 2) & mask_dc
        angular_energy.append(np.sum(magnitude[bin_mask]))
    angular_energy = np.array(angular_energy)
    if np.max(angular_energy) > 0:
        angular_energy = angular_energy / np.max(angular_energy)
        
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Panel 1: Masked Spatial Image
    circular_mask = R <= (min(h, w) / 2.0)
    masked_img = image_gray * circular_mask
    axes[0].imshow(masked_img, cmap='gray')
    axes[0].set_title(f"Spatial Image\n({title})", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Log Magnitude Spectrum
    im1 = axes[1].imshow(log_mag, cmap='magma')
    axes[1].set_title("Log Magnitude Spectrum", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Sector Overlay (Red = Cardinal, Blue = Oblique)
    overlay = np.zeros((h, w, 3))
    norm_log = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    overlay[:, :, 0] = norm_log
    overlay[:, :, 1] = norm_log
    overlay[:, :, 2] = norm_log
    
    overlay[mask_cardinal, 0] = 1.0 
    overlay[mask_cardinal, 1] *= 0.3
    overlay[mask_cardinal, 2] *= 0.3
    
    overlay[mask_oblique, 2] = 1.0  
    overlay[mask_oblique, 0] *= 0.3
    overlay[mask_oblique, 1] *= 0.3
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Sector Overlay\nNDI = {ndi:.4f}", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    # Panel 4: 1D Angular Energy Distribution
    axes[3].plot(angles, angular_energy, color='darkgreen', lw=2)
    axes[3].axvspan(0, 10, color='red', alpha=0.2, label='Cardinal (0°/180°)')
    axes[3].axvspan(80, 100, color='red', alpha=0.2, label='Cardinal (90°)')
    axes[3].axvspan(170, 180, color='red', alpha=0.2)
    
    axes[3].axvspan(35, 55, color='blue', alpha=0.2, label='Oblique (45°)')
    axes[3].axvspan(125, 145, color='blue', alpha=0.2, label='Oblique (135°)')
    
    axes[3].set_xticks([0, 45, 90, 135, 180])
    axes[3].set_xlim(0, 180)
    
    if log_scale:
        axes[3].set_yscale('log')
        pos_vals = angular_energy[angular_energy > 0]
        min_pos = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
        axes[3].set_ylim(bottom=max(1e-4, min_pos * 0.5), top=1.2)
        axes[3].set_ylabel("Norm. Energy (Log Scale)", fontsize=9)
    else:
        axes[3].set_ylabel("Norm. Energy", fontsize=9)

    axes[3].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[3].set_title("Angular Energy Distribution", fontsize=11, fontweight='bold')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    axes[3].legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    return fig

def analyze_layer_filters(filter_images, dc_radius=5):
    """Averages spectral transformations and NDI anisotropy across all channels in a target layer."""
    if len(filter_images) == 0:
        return np.zeros((80, 80)), 0.0

    sample_log, _ = compute_rotational_anisotropy(filter_images[0], dc_radius=dc_radius)
    h, w = sample_log.shape
    
    avg_log_spectrum = np.zeros((h, w), dtype=np.float64)
    anisotropy_scores = []
    
    for img in filter_images:
        log_mag, ndi = compute_rotational_anisotropy(img, dc_radius=dc_radius)
        avg_log_spectrum += log_mag
        anisotropy_scores.append(ndi)
        
    avg_log_spectrum /= len(filter_images)
    avg_anisotropy = np.mean(anisotropy_scores)
    
    return avg_log_spectrum, avg_anisotropy


def compute_layer_angular_distribution(filter_images, dc_radius=5):
    """
    Computes averaged 1D angular energy distribution profile and NDI anisotropy score
    across all filter images in a target layer.
    """
    angles = np.arange(0, 180, 2)
    if len(filter_images) == 0:
        return angles, np.zeros_like(angles, dtype=np.float64), 0.0

    sample_gray = _step1_preprocess(filter_images[0])
    h, w = sample_gray.shape
    cy, cx = h // 2, w // 2
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
    mask_dc = R > dc_radius

    total_angular_energy = np.zeros_like(angles, dtype=np.float64)
    anisotropy_scores = []

    for img in filter_images:
        image_gray = _step1_preprocess(img)
        magnitude, _, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius)
        anisotropy_scores.append(ndi)

        img_energy = []
        for a in angles:
            bin_mask = (np.abs(theta - a) <= 2) & mask_dc
            img_energy.append(np.sum(magnitude[bin_mask]))
        total_angular_energy += np.array(img_energy)

    avg_angular_energy = total_angular_energy / len(filter_images)
    if np.max(avg_angular_energy) > 0:
        avg_angular_energy = avg_angular_energy / np.max(avg_angular_energy)

    avg_anisotropy = np.mean(anisotropy_scores)
    return angles, avg_angular_energy, avg_anisotropy


def plot_fourier_cmap_grid(filter_data, models, layers, dc_radius=5, log_scale=False, figsize=(14, 11), dpi=150):
    """
    Plots a grid (len(models) x len(layers)) of averaged 2D Fourier Log Magnitude Spectra (CMAP).
    Optionally applies log-scale color normalization via log_scale=True.
    Prints a summary text table of Anisotropy Index (NDI) for each model & layer.
    """
    anisotropy_results = {m: {} for m in models}

    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)
    fig.suptitle("Layer-wise 2D Fourier Spectra Analysis & Anisotropy Index Metrics", fontsize=16, y=0.96)

    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]

            # Process structural metrics
            avg_spectrum, anisotropy_index = analyze_layer_filters(images_list, dc_radius=dc_radius)
            anisotropy_results[model_name][layer_name] = anisotropy_index

            # Plot spectrum maps
            ax = axs[i, j] if len(models) > 1 and len(layers) > 1 else (axs[i] if len(models) > 1 else axs[j])
            
            if log_scale:
                pos_vals = avg_spectrum[avg_spectrum > 0]
                vmin = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
                norm = LogNorm(vmin=max(1e-4, vmin), vmax=max(np.max(avg_spectrum), vmin * 10))
                im = ax.imshow(avg_spectrum, cmap='magma', norm=norm)
            else:
                im = ax.imshow(avg_spectrum, cmap='magma')

            ax.set_title(f"Model: {model_name.upper()}\nFeatures.{layer_name}.Conv\n[AI Index: {anisotropy_index:.3f}]", fontsize=10)
            ax.axis('off')

            # Single colorbar anchor per row to prevent visual cluttering
            if j == len(layers) - 1:
                fig.colorbar(im, ax=ax, shrink=0.7, label='Log Spectral Intensity')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Print out summary text table
    print("\n" + "="*54)
    print(f"{'LAYER CONFIGURATION':<25} | {'MODEL':<12} | {'ANISOTROPY INDEX':<10}")
    print("="*54)
    for l in layers:
        for m in models:
            score = anisotropy_results[m][l]
            print(f"FEATURES.{l}.CONV{' ':<11} | {m:<12} | {score:.4f}")
        print("-"*54)

    return anisotropy_results


def plot_fourier_angular_distribution_grid(filter_data, models, layers, dc_radius=5, log_scale=False, figsize=(16, 11), dpi=150):
    """
    Plots a grid (len(models) x len(layers)) of averaged 1D Angular Energy Distributions.
    Highlights Cardinal (0°/180°, 90°) and Oblique (45°, 135°) angular sectors.
    Optionally sets Y-axis to logarithmic scale when log_scale=True.
    Prints a summary text table of Anisotropy Index (NDI) for each model & layer.
    """
    anisotropy_results = {m: {} for m in models}

    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)
    fig.suptitle("Layer-wise 1D Angular Energy Distribution & Anisotropy Index Metrics", fontsize=16, y=0.96)

    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]

            angles, avg_angular_energy, anisotropy_index = compute_layer_angular_distribution(images_list, dc_radius=dc_radius)
            anisotropy_results[model_name][layer_name] = anisotropy_index

            ax = axs[i, j] if len(models) > 1 and len(layers) > 1 else (axs[i] if len(models) > 1 else axs[j])

            # Plot 1D energy profile
            ax.plot(angles, avg_angular_energy, color='darkgreen', lw=2)

            # Highlight Cardinal (red) and Oblique (blue) sectors
            ax.axvspan(0, 10, color='red', alpha=0.2, label='Cardinal (0°/180°)' if (i == 0 and j == 0) else "")
            ax.axvspan(80, 100, color='red', alpha=0.2, label='Cardinal (90°)' if (i == 0 and j == 0) else "")
            ax.axvspan(170, 180, color='red', alpha=0.2)

            ax.axvspan(35, 55, color='blue', alpha=0.2, label='Oblique (45°)' if (i == 0 and j == 0) else "")
            ax.axvspan(125, 145, color='blue', alpha=0.2, label='Oblique (135°)' if (i == 0 and j == 0) else "")

            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlim(0, 180)

            if log_scale:
                ax.set_yscale('log')
                pos_vals = avg_angular_energy[avg_angular_energy > 0]
                min_pos = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
                ax.set_ylim(bottom=max(1e-4, min_pos * 0.5), top=1.2)
            else:
                ax.set_ylim(0, 1.05)

            ax.set_title(f"Model: {model_name.upper()}\nFeatures.{layer_name}.Conv\n[AI Index: {anisotropy_index:.3f}]", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

            if j == 0:
                ax.set_ylabel("Norm. Energy (Log Scale)" if log_scale else "Norm. Energy", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("Angle θ (degrees)", fontsize=9)

    # Add single legend for the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Print out summary text table
    print("\n" + "="*54)
    print(f"{'LAYER CONFIGURATION':<25} | {'MODEL':<12} | {'ANISOTROPY INDEX':<10}")
    print("="*54)
    for l in layers:
        for m in models:
            score = anisotropy_results[m][l]
            print(f"FEATURES.{l}.CONV{' ':<11} | {m:<12} | {score:.4f}")
        print("-"*54)

    return anisotropy_results



