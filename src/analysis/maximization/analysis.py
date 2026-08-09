# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
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
    appeared_mae_list, disappeared_mae_list = [], []
    
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
        appeared_mae = np.mean(np.maximum(0.0, img_b - img_a))
        disappeared_mae = np.mean(np.maximum(0.0, img_a - img_b))
        score = ssim(img_a, img_b, data_range=1.0)
        
        results[idx] = {
            'mse': mse,
            'mae': mae,
            'appeared_mae': appeared_mae,
            'disappeared_mae': disappeared_mae,
            'ssim': score
        }
        
        mse_list.append(mse)
        mae_list.append(mae)
        appeared_mae_list.append(appeared_mae)
        disappeared_mae_list.append(disappeared_mae)
        ssim_list.append(score)

    summary_stats = {
        'avg_mse': np.mean(mse_list),
        'avg_mae': np.mean(mae_list),
        'avg_appeared_mae': np.mean(appeared_mae_list),
        'avg_disappeared_mae': np.mean(disappeared_mae_list),
        'avg_ssim': np.mean(ssim_list),
        'most_drifted_filter': common_indices[np.argmin(ssim_list)],
        'least_drifted_filter': common_indices[np.argmax(ssim_list)]
    }
    
    print(f"\n======== STATISTICAL ANALYSIS FOR {layer_name.upper()} ({pair_labels[0]} vs {pair_labels[1]}) ========")
    print(f"Total Filters Evaluated : {len(common_indices)}")
    print(f"Layer Average SSIM      : {summary_stats['avg_ssim']:.4f}")
    print(f"Layer Average Total MAE : {summary_stats['avg_mae']:.6f}")
    print(f"  └─ Appeared Gain MAE  : {summary_stats['avg_appeared_mae']:.6f}  (ReLU({pair_labels[1]} - {pair_labels[0]}))")
    print(f"  └─ Disappeared Loss MAE: {summary_stats['avg_disappeared_mae']:.6f} (ReLU({pair_labels[0]} - {pair_labels[1]}))")
    print(f"Layer Average MSE       : {summary_stats['avg_mse']:.6f}")
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

def compute_difference_masks(patterns_a, patterns_b, mode="absolute"):
    """
    Computes directional or absolute difference masks for matching filter indices
    between two pattern dictionaries/lists:
    - mode="absolute"              : |gray_b - gray_a| (Unsigned total difference)
    - mode="appeared" / "gained"   : ReLU(gray_b - gray_a) (Newly introduced features in B)
    - mode="disappeared" / "lost"  : ReLU(gray_a - gray_b) (Erased features from baseline A)
    """
    if isinstance(patterns_a, list) and isinstance(patterns_b, list):
        patterns_a = {i: img for i, img in enumerate(patterns_a)}
        patterns_b = {i: img for i, img in enumerate(patterns_b)}

    common_indices = sorted(list(set(patterns_a.keys()).intersection(set(patterns_b.keys()))))
    diff_masks = {}

    for idx in common_indices:
        tensor_a = patterns_a[idx]
        tensor_b = patterns_b[idx]

        if torch.is_tensor(tensor_a):
            tensor_a = tensor_a.detach().cpu()
        else:
            tensor_a = torch.from_numpy(np.array(tensor_a))

        if torch.is_tensor(tensor_b):
            tensor_b = tensor_b.detach().cpu()
        else:
            tensor_b = torch.from_numpy(np.array(tensor_b))

        if tensor_a.ndim == 3 and tensor_a.shape[0] != 3 and tensor_a.shape[2] == 3:
            tensor_a = tensor_a.permute(2, 0, 1)
        if tensor_b.ndim == 3 and tensor_b.shape[0] != 3 and tensor_b.shape[2] == 3:
            tensor_b = tensor_b.permute(2, 0, 1)

        if tensor_a.shape != tensor_b.shape:
            tensor_b = F.interpolate(
                tensor_b.unsqueeze(0),
                size=(tensor_a.shape[1], tensor_a.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        gray_a = 0.2989 * tensor_a[0] + 0.5870 * tensor_a[1] + 0.1140 * tensor_a[2]
        gray_b = 0.2989 * tensor_b[0] + 0.5870 * tensor_b[1] + 0.1140 * tensor_b[2]

        if mode in ["appeared", "gained", "new"]:
            diff_gray = torch.clamp(gray_b - gray_a, min=0.0)
        elif mode in ["disappeared", "lost", "gone"]:
            diff_gray = torch.clamp(gray_a - gray_b, min=0.0)
        else:
            diff_gray = torch.abs(gray_b - gray_a)

        diff_rgb = diff_gray.unsqueeze(0).repeat(3, 1, 1)
        diff_masks[idx] = diff_rgb

    return diff_masks


def plot_difference_grid_pairwise(patterns_a, patterns_b, title_suffix, mode="both", nrow=8, padding=4):
    """
    Plots directional spatial difference mask grids across matching filter indices.
    When mode="both" (default), displays two separate grid plots:
    1. ✨ Appeared / Newly Learned Features (ReLU(I_B - I_A))
    2. 🍂 Disappeared / Erased Features (ReLU(I_A - I_B))
    """
    modes_to_plot = ["appeared", "disappeared"] if mode in ["both", "separate", "all"] else [mode]
    
    for m in modes_to_plot:
        diff_masks = compute_difference_masks(patterns_a, patterns_b, mode=m)
        if not diff_masks:
            continue

        diff_tensors = [mask[0].unsqueeze(0) for mask in diff_masks.values()]
        batch_tensor = torch.stack(diff_tensors, dim=0)
        grid = vutils.make_grid(batch_tensor, nrow=nrow, padding=padding, normalize=False)
        grid_np = grid[0].cpu().numpy()
        
        mode_label = "APPEARED FEATURES (ReLU(I_B - I_A))" if m in ["appeared", "gained", "new"] else \
                     ("DISAPPEARED FEATURES (ReLU(I_A - I_B))" if m in ["disappeared", "lost", "gone"] else "ABSOLUTE DIFFERENCE (|I_B - I_A|)")
        
        plt.figure(figsize=(14, 10), dpi=200)
        im = plt.imshow(grid_np, cmap='hot', vmin=0.0, vmax=0.5)
        
        plt.title(f"{mode_label} - {title_suffix.upper()}", fontsize=12, fontweight='bold', pad=15)
        plt.axis('off')
        plt.colorbar(im, shrink=0.6, label='Structural Representation Change Magnitude')
        plt.tight_layout()
        plt.show()


def compare_all_domain_states(nat_patterns, ft_patterns, scr_patterns, layer_name="Layer", mode="both"):
    """
    Executes pairwise spatial difference grid plotting across all domain pairs,
    displaying both Appeared and Disappeared feature grids for each pair.
    """
    # Comparison 1: Natural vs Fine-Tuned
    plot_difference_grid_pairwise(
        nat_patterns, 
        ft_patterns, 
        title_suffix=f"{layer_name} (Natural vs Fine-Tuned)",
        mode=mode
    )
    
    # Comparison 2: Natural vs Screen-from-Scratch
    plot_difference_grid_pairwise(
        nat_patterns, 
        scr_patterns, 
        title_suffix=f"{layer_name} (Natural vs Screen Scratch)",
        mode=mode
    )
    
    # Comparison 3: Fine-Tuned vs Screen-from-Scratch
    plot_difference_grid_pairwise(
        ft_patterns, 
        scr_patterns, 
        title_suffix=f"{layer_name} (Fine-Tuned vs Screen Scratch)",
        mode=mode
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

def compute_fourier_spectrum(image_gray, dc_radius=5, use_log=False):
    """
    Step 2: Apply a circular spatial mask, compute 2D FFT, and calculate
    Normalized Difference Index (NDI) across cardinal and oblique sectors.
    Optionally computes NDI using log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
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
    
    # 3. Frequency sector masks (exclude DC center component and corner un-inscribed frequencies)
    # Using canonical literature measurement windows (+/- 21° around target centers):
    # Cardinal: 0°/180° ± 21° and 90° ± 21° (Total bandwidth = 84°)
    # Oblique: 45° ± 21° and 135° ± 21°    (Total bandwidth = 84°)
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = dist_from_center
    
    mask_dc = (R > dc_radius) & (R <= radius)
    mask_cardinal = ((theta < 21) | (theta > 159) | ((theta > 69) & (theta < 111))) & mask_dc
    mask_oblique = (((theta >= 24) & (theta <= 66)) | ((theta >= 114) & (theta <= 156))) & mask_dc
    
    spectrum = log_magnitude if use_log else magnitude
    
    cardinal_energy = np.sum(spectrum[mask_cardinal])
    oblique_energy = np.sum(spectrum[mask_oblique])
    
    total_energy = cardinal_energy + oblique_energy
    if total_energy == 0:
        ndi = 0.0
    else:
        ndi = (cardinal_energy - oblique_energy) / total_energy
        
    return magnitude, log_magnitude, ndi

def compute_rotational_anisotropy(image_data, dc_radius=5, use_log=True):
    """
    Preprocesses image data (grayscale + square crop) and computes Fourier NDI metrics.
    Optionally returns log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
    """
    image_gray = _step1_preprocess(image_data)
    mag, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, use_log=use_log)
    spectrum = log_mag if use_log else mag
    return spectrum, ndi

def plot_fourier_diagnostic(image_data, title="Fourier Diagnostic", dc_radius=5, log_scale=False, use_log=False, figsize=(18, 4.5)):
    """
    Generates a 4-panel visual diagnostic figure:
    1. Spatial Image (Circular Masked)
    2. Log Magnitude Fourier Spectrum
    3. Cardinal (Red) vs Oblique (Blue) Sector Overlay
    4. 1D Angular Energy Distribution Plot across 0°, 45°, 90°, 135°, 180° (with optional log_scale/use_log)
    """
    image_gray = _step1_preprocess(image_data)
    magnitude, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, use_log=use_log)
    spectrum = log_mag if use_log else magnitude
    
    h, w = image_gray.shape
    cy, cx = h // 2, w // 2
    
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
    max_radius = min(h, w) / 2.0
    
    mask_dc = (R > dc_radius) & (R <= max_radius)
    mask_cardinal = ((theta < 21) | (theta > 159) | ((theta > 69) & (theta < 111))) & mask_dc
    mask_oblique = (((theta >= 24) & (theta <= 66)) | ((theta >= 114) & (theta <= 156))) & mask_dc
    
    # Calculate 1D Angular Energy Profile
    angles = np.arange(0, 180, 2)
    angular_energy = []
    for a in angles:
        bin_mask = (np.abs(theta - a) <= 2) & mask_dc
        angular_energy.append(np.sum(spectrum[bin_mask]))
    angular_energy = np.array(angular_energy)
    if np.max(angular_energy) > 0:
        angular_energy = angular_energy / np.max(angular_energy)
        
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Panel 1: Masked Spatial Image
    circular_mask = R <= max_radius
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
    axes[3].axvspan(0, 21, color='red', alpha=0.2, label='Cardinal (0°/180° ±21°)')
    axes[3].axvspan(69, 111, color='red', alpha=0.2, label='Cardinal (90° ±21°)')
    axes[3].axvspan(159, 180, color='red', alpha=0.2)
    
    axes[3].axvspan(24, 66, color='blue', alpha=0.2, label='Oblique (45° ±21°)')
    axes[3].axvspan(114, 156, color='blue', alpha=0.2, label='Oblique (135° ±21°)')
    
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

def analyze_layer_filters(filter_images, dc_radius=5, use_log=True):
    """Averages spectral transformations and NDI anisotropy across all channels in a target layer."""
    if len(filter_images) == 0:
        return np.zeros((80, 80)), 0.0

    sample_spec, _ = compute_rotational_anisotropy(filter_images[0], dc_radius=dc_radius, use_log=use_log)
    h, w = sample_spec.shape
    
    avg_spectrum = np.zeros((h, w), dtype=np.float64)
    anisotropy_scores = []
    
    for img in filter_images:
        spec, ndi = compute_rotational_anisotropy(img, dc_radius=dc_radius, use_log=use_log)
        avg_spectrum += spec
        anisotropy_scores.append(ndi)
        
    avg_spectrum /= len(filter_images)
    avg_anisotropy = np.mean(anisotropy_scores)
    
    return avg_spectrum, avg_anisotropy


def compute_layer_angular_distribution(filter_images, dc_radius=5, use_log=False):
    """
    Computes averaged 1D angular energy distribution profile and NDI anisotropy score
    across all filter images in a target layer.
    Optionally accumulates log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
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
    max_radius = min(h, w) / 2.0
    mask_dc = (R > dc_radius) & (R <= max_radius)

    total_angular_energy = np.zeros_like(angles, dtype=np.float64)
    anisotropy_scores = []

    for img in filter_images:
        image_gray = _step1_preprocess(img)
        magnitude, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, use_log=use_log)
        anisotropy_scores.append(ndi)

        spectrum = log_mag if use_log else magnitude

        img_energy = []
        for a in angles:
            bin_mask = (np.abs(theta - a) <= 2) & mask_dc
            img_energy.append(np.sum(spectrum[bin_mask]))
        total_angular_energy += np.array(img_energy)

    avg_angular_energy = total_angular_energy / len(filter_images)
    if np.max(avg_angular_energy) > 0:
        avg_angular_energy = avg_angular_energy / np.max(avg_angular_energy)

    avg_anisotropy = np.mean(anisotropy_scores)
    return angles, avg_angular_energy, avg_anisotropy


def plot_fourier_cmap_grid(filter_data, models, layers, dc_radius=5, log_scale=False, use_log=True, figsize=(14, 11), dpi=150):
    """
    Plots a grid (len(models) x len(layers)) of averaged 2D Fourier Magnitude Spectra (CMAP).
    Optionally toggles log-magnitude transformation via use_log=True (default True).
    Optionally applies log-scale color normalization via log_scale=True.
    Prints a summary text table of Anisotropy Index (NDI) for each model & layer.
    """
    anisotropy_results = {m: {} for m in models}

    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)
    title_prefix = "Log-Magnitude" if use_log else "Linear Magnitude"
    fig.suptitle(f"Layer-wise 2D Fourier {title_prefix} Spectra Analysis & Anisotropy Index Metrics", fontsize=16, y=0.96)

    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]

            # Process structural metrics
            avg_spectrum, anisotropy_index = analyze_layer_filters(images_list, dc_radius=dc_radius, use_log=use_log)
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


def plot_fourier_angular_distribution_grid(filter_data, models, layers, dc_radius=5, log_scale=False, use_log=False, plot_differences=True, figsize=(16, 11), dpi=150):
    """
    Plots a grid (len(models) x len(layers)) of averaged 1D Angular Energy Distributions.
    Highlights Cardinal (0°/180°, 90°) and Oblique (45°, 135°) angular sectors.
    Optionally sets Y-axis to logarithmic scale when log_scale=True.
    Optionally toggles log-magnitude energy summation via use_log=True (default False for linear magnitude energy).
    Optionally plots difference profiles (Fine-Tuned vs Natural, Screen vs Natural) when plot_differences=True.
    Prints a summary text table of Anisotropy Index (NDI) for each model & layer.
    """
    anisotropy_results = {m: {} for m in models}

    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)
    fig.suptitle("Layer-wise 1D Angular Energy Distribution & Anisotropy Index Metrics", fontsize=16, y=0.96)

    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]

            angles, avg_angular_energy, anisotropy_index = compute_layer_angular_distribution(images_list, dc_radius=dc_radius, use_log=use_log)
            anisotropy_results[model_name][layer_name] = anisotropy_index

            ax = axs[i, j] if len(models) > 1 and len(layers) > 1 else (axs[i] if len(models) > 1 else axs[j])

            # Plot 1D energy profile
            ax.plot(angles, avg_angular_energy, color='darkgreen', lw=2)

            # Highlight Cardinal (red) and Oblique (blue) sectors (±21° literature measurement windows)
            ax.axvspan(0, 21, color='red', alpha=0.2, label='Cardinal (0°/180°)' if (i == 0 and j == 0) else "")
            ax.axvspan(69, 111, color='red', alpha=0.2, label='Cardinal (90°)' if (i == 0 and j == 0) else "")
            ax.axvspan(159, 180, color='red', alpha=0.2)

            ax.axvspan(24, 66, color='blue', alpha=0.2, label='Oblique (45°)' if (i == 0 and j == 0) else "")
            ax.axvspan(114, 156, color='blue', alpha=0.2, label='Oblique (135°)' if (i == 0 and j == 0) else "")

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

    if plot_differences and "natural" in filter_data:
        plot_fourier_angular_difference_grid(filter_data, layers, dc_radius=dc_radius, use_log=use_log, dpi=dpi)

    return anisotropy_results


def plot_fourier_angular_difference_grid(filter_data, layers, dc_radius=5, use_log=False, figsize=(16, 8), dpi=150):
    """
    Plots a grid (2 x len(layers)) of 1D Angular Energy Difference Profiles:
    - Row 1: Fine-Tuned minus Natural (FT - NAT)
    - Row 2: Screen (Scratch) minus Natural (SCR - NAT)

    Highlights Cardinal (0°/180°, 90°) and Oblique (45°, 135°) angular sectors.
    Includes a reference zero baseline (y=0) and shaded gain/loss areas.
    Prints a summary text table of NDI scores and NDI shift (ΔNDI).
    """
    diff_pairs = [
        ("fine-tuned", "natural", "Diff: FINE-TUNED - NATURAL"),
        ("screen", "natural", "Diff: SCREEN - NATURAL")
    ]

    fig, axs = plt.subplots(len(diff_pairs), len(layers), figsize=figsize, dpi=dpi)
    fig.suptitle("Layer-wise 1D Angular Energy Difference Profiles (Domain Drift Shift)", fontsize=16, y=0.96)

    table_data = []

    for i, (model_b_name, model_a_name, pair_title) in enumerate(diff_pairs):
        for j, layer_name in enumerate(layers):
            images_a = filter_data[model_a_name][layer_name]
            images_b = filter_data[model_b_name][layer_name]

            angles, energy_a, ndi_a = compute_layer_angular_distribution(images_a, dc_radius=dc_radius, use_log=use_log)
            angles, energy_b, ndi_b = compute_layer_angular_distribution(images_b, dc_radius=dc_radius, use_log=use_log)

            diff_energy = energy_b - energy_a
            delta_ndi = ndi_b - ndi_a

            table_data.append((layer_name, model_b_name.upper(), model_a_name.upper(), ndi_b, ndi_a, delta_ndi))

            ax = axs[i, j] if len(diff_pairs) > 1 and len(layers) > 1 else (axs[i] if len(diff_pairs) > 1 else axs[j])

            color = 'purple' if i == 0 else 'crimson'
            ax.plot(angles, diff_energy, color=color, lw=2, label=f"Δ Energy ({model_b_name[:2].upper()} - {model_a_name[:3].upper()})")

            # Reference baseline at y = 0
            ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1)

            # Shaded positive (gain) and negative (loss) regions
            ax.fill_between(angles, diff_energy, 0, where=(diff_energy >= 0), color=color, alpha=0.15)
            ax.fill_between(angles, diff_energy, 0, where=(diff_energy < 0), color='gray', alpha=0.15)

            # Highlight Cardinal (red) and Oblique (blue) sectors (±21° literature measurement windows)
            ax.axvspan(0, 21, color='red', alpha=0.15, label='Cardinal (0°/180°)' if (i == 0 and j == 0) else "")
            ax.axvspan(69, 111, color='red', alpha=0.15, label='Cardinal (90°)' if (i == 0 and j == 0) else "")
            ax.axvspan(159, 180, color='red', alpha=0.15)

            ax.axvspan(24, 66, color='blue', alpha=0.15, label='Oblique (45°)' if (i == 0 and j == 0) else "")
            ax.axvspan(114, 156, color='blue', alpha=0.15, label='Oblique (135°)' if (i == 0 and j == 0) else "")

            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlim(0, 180)

            # Symmetrical y-axis limits around 0
            max_abs_diff = np.max(np.abs(diff_energy)) if len(diff_energy) > 0 else 0.5
            ylim_val = max(0.1, max_abs_diff * 1.2)
            ax.set_ylim(-ylim_val, +ylim_val)

            ax.set_title(f"{pair_title}\nFeatures.{layer_name}.Conv\n[ΔNDI: {delta_ndi:+.3f}]", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

            if j == 0:
                ax.set_ylabel("Δ Norm. Energy", fontsize=9)
            if i == len(diff_pairs) - 1:
                ax.set_xlabel("Angle θ (degrees)", fontsize=9)

    # Add single legend for the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Print summary text table
    print("\n" + "="*68)
    print(f"{'LAYER':<16} | {'COMPARISON (B vs A)':<24} | {'NDI (B)':<9} | {'NDI (A)':<9} | {'Δ NDI':<8}")
    print("="*68)
    for l_name, m_b, m_a, n_b, n_a, d_ndi in table_data:
        comp_str = f"{m_b} vs {m_a}"
        print(f"FEATURES.{l_name}.CONV | {comp_str:<24} | {n_b:9.4f} | {n_a:9.4f} | {d_ndi:+8.4f}")
        print("-" * 68)

    return table_data


# ==============================================================================
# ABSOLUTE DIFFERENCE MASK FOURIER ANALYSIS SEQUENCE
# ==============================================================================

def analyze_difference_masks_fourier(
    filter_data,
    layers,
    pairs=None,
    mode="all",
    dc_radius=5,
    log_scale=False,
    use_log=False,
    figsize=(16, 10),
    dpi=150
):
    """
    Applies the full 2D and 1D Fourier spectral analysis sequence and NDI Anisotropy Index metrics
    directly to spatial DIFFERENCE MASKS across model domain pairs.
    
    Parameters:
    -----------
    mode : str
        - "all" / "separate" : Analyzes APPEARED Features (ReLU(I_B - I_A)), DISAPPEARED Features (ReLU(I_A - I_B)),
                                and ABSOLUTE Total Differences (|I_B - I_A|) separately.
        - "appeared"        : Analyzes newly introduced features in model B.
        - "disappeared"     : Analyzes erased features from baseline model A.
        - "absolute"        : Analyzes unsigned total difference magnitude.
    """
    if pairs is None:
        pairs = [
            ("fine-tuned", "natural", "FT - NAT"),
            ("screen", "natural", "SCR - NAT"),
            ("fine-tuned", "screen", "FT - SCR")
        ]

    modes_to_run = ["appeared", "disappeared", "absolute"] if mode in ["all", "separate", "both"] else [mode]
    results = {}

    for current_mode in modes_to_run:
        mode_title = {
            "appeared": "✨ APPEARED / NEWLY LEARNED FEATURES (ReLU(I_B - I_A))",
            "disappeared": "🍂 DISAPPEARED / ERASED FEATURES (ReLU(I_A - I_B))",
            "absolute": "📊 ABSOLUTE TOTAL DIFFERENCE MASKS (|I_B - I_A|)"
        }.get(current_mode, f"FOURIER ANALYSIS ({current_mode.upper()})")

        diff_filter_data = {}
        diff_models = []

        for model_b, model_a, label in pairs:
            pair_key = f"{model_b}_minus_{model_a}"
            diff_models.append(pair_key)
            diff_filter_data[pair_key] = {}
            for layer_name in layers:
                p_a = filter_data[model_a][layer_name]
                p_b = filter_data[model_b][layer_name]
                diff_filter_data[pair_key][layer_name] = list(compute_difference_masks(p_a, p_b, mode=current_mode).values())

        print(f"\n=======================================================")
        print(f"📊 {mode_title}")
        print(f"=======================================================\n")

        anisotropy_angular = plot_fourier_angular_distribution_grid(
            filter_data=diff_filter_data,
            models=diff_models,
            layers=layers,
            dc_radius=dc_radius,
            log_scale=log_scale,
            use_log=use_log,
            plot_differences=False,
            figsize=figsize,
            dpi=dpi
        )
        results[current_mode] = diff_filter_data

    return results if len(modes_to_run) > 1 else results[modes_to_run[0]]


def plot_synthetic_fourier_difference_demo(dc_radius=5, use_log=False, figsize=(16, 4.5), dpi=150):
    """
    Generates a synthetic demonstration using two controlled 2D spatial gratings:
    - Image 1: 90° frequency energy (horizontal spatial grating)
    - Image 2: 45° frequency energy (135° spatial grating)

    Plots individual 1D Angular Energy distributions and calculates the resulting
    difference profile ΔEnergy = Energy_45° - Energy_90°.
    Optionally computes energy distributions with use_log=True or linear magnitude with use_log=False.
    """
    h, w = 120, 120
    y, x = np.indices((h, w))

    # Image 1: 90° frequency energy (horizontal spatial grating y)
    img_90 = np.sin(2 * np.pi * 0.15 * y)

    # Image 2: 45° frequency energy (diagonal spatial grating x+y)
    img_45 = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))

    mag_90, log_90, ndi_90 = compute_fourier_spectrum(img_90, dc_radius=dc_radius, use_log=use_log)
    mag_45, log_45, ndi_45 = compute_fourier_spectrum(img_45, dc_radius=dc_radius, use_log=use_log)

    angles, energy_90, _ = compute_layer_angular_distribution([img_90], dc_radius=dc_radius, use_log=use_log)
    angles, energy_45, _ = compute_layer_angular_distribution([img_45], dc_radius=dc_radius, use_log=use_log)
    diff_energy = energy_45 - energy_90

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.suptitle("Synthetic Demonstration: 90° vs 45° Spectral Energy Shift", fontsize=14, fontweight="bold")

    # Panel 1: Combined 2D Log Spectrum
    im0 = axes[0].imshow(log_90 + log_45, cmap="magma")
    axes[0].set_title("Combined 2D Fourier Spectra", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: 1D Angular Energy Distributions
    axes[1].plot(angles, energy_90, color="darkgreen", lw=2, label="90° Grating (Cardinal)")
    axes[1].plot(angles, energy_45, color="darkorange", lw=2, label="45° Grating (Oblique)")
    axes[1].axvspan(0, 10, color="red", alpha=0.15)
    axes[1].axvspan(80, 100, color="red", alpha=0.15, label="Cardinal Bins")
    axes[1].axvspan(170, 180, color="red", alpha=0.15)
    axes[1].axvspan(35, 55, color="blue", alpha=0.15, label="Oblique Bins")
    axes[1].axvspan(125, 145, color="blue", alpha=0.15)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_xlim(0, 180)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[1].set_ylabel("Norm. Energy", fontsize=9)
    axes[1].set_title("Individual 1D Energy Distributions", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Difference Profile
    axes[2].plot(angles, diff_energy, color="purple", lw=2, label="Δ Energy (45° - 90°)")
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.6, lw=1)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy >= 0), color="purple", alpha=0.2)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy < 0), color="gray", alpha=0.2)
    axes[2].axvspan(0, 10, color="red", alpha=0.15)
    axes[2].axvspan(80, 100, color="red", alpha=0.15)
    axes[2].axvspan(170, 180, color="red", alpha=0.15)
    axes[2].axvspan(35, 55, color="blue", alpha=0.15)
    axes[2].axvspan(125, 145, color="blue", alpha=0.15)
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].set_xlim(0, 180)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Δ Norm. Energy", fontsize=9)
    axes[2].set_title(f"Difference Profile (ΔNDI: {ndi_45 - ndi_90:+.3f})", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    return fig





