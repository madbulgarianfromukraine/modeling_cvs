# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
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
    
    print(f"\n======== STATISTICAL DRIFT ANALYSIS FOR {layer_name.upper()} ({pair_labels[1]} - {pair_labels[0]}) ========")
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


def compute_difference_masks(patterns_a, patterns_b, mode="appeared"):
    """
    Computes spatial difference masks for matching pattern dictionary keys or list indices:
    - mode="appeared" / "gained"   : ReLU(B - A) = max(0, B - A)
    - mode="disappeared" / "lost"  : ReLU(A - B) = max(0, A - B)
    - mode="absolute"              : |B - A|
    """
    if isinstance(patterns_a, list):
        patterns_a = {i: img for i, img in enumerate(patterns_a)}
    if isinstance(patterns_b, list):
        patterns_b = {i: img for i, img in enumerate(patterns_b)}

    # Normalize integer-like string keys (e.g. "0" -> 0)
    norm_a = {int(k) if str(k).isdigit() else k: v for k, v in patterns_a.items()}
    norm_b = {int(k) if str(k).isdigit() else k: v for k, v in patterns_b.items()}

    common_indices = sorted(list(set(norm_a.keys()).intersection(set(norm_b.keys()))))
    # Fallback to positional index matching if keys do not directly intersect
    if len(common_indices) == 0 and len(norm_a) > 0 and len(norm_b) > 0:
        list_a = list(norm_a.values())
        list_b = list(norm_b.values())
        min_len = min(len(list_a), len(list_b))
        norm_a = {i: list_a[i] for i in range(min_len)}
        norm_b = {i: list_b[i] for i in range(min_len)}
        common_indices = list(range(min_len))

    masks = {}
    for idx in common_indices:
        tensor_a = norm_a[idx]
        tensor_b = norm_b[idx]
        if torch.is_tensor(tensor_a):
            tensor_a = tensor_a.detach().cpu()
        else:
            tensor_a = torch.from_numpy(np.asarray(tensor_a))
        if torch.is_tensor(tensor_b):
            tensor_b = tensor_b.detach().cpu()
        else:
            tensor_b = torch.from_numpy(np.asarray(tensor_b))

        if tensor_a.ndim == 3 and tensor_a.shape[0] != 3 and tensor_a.shape[2] == 3:
            tensor_a = tensor_a.permute(2, 0, 1)
        if tensor_b.ndim == 3 and tensor_b.shape[0] != 3 and tensor_b.shape[2] == 3:
            tensor_b = tensor_b.permute(2, 0, 1)

        if tensor_a.shape != tensor_b.shape:
            tensor_b = F.interpolate(
                tensor_b.unsqueeze(0),
                size=(tensor_a.shape[-2], tensor_a.shape[-1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert to grayscale 1xHxW if 3 channels
        if tensor_a.ndim == 3 and tensor_a.shape[0] == 3:
            img_a = 0.2989 * tensor_a[0:1] + 0.5870 * tensor_a[1:2] + 0.1140 * tensor_a[2:3]
            img_b = 0.2989 * tensor_b[0:1] + 0.5870 * tensor_b[1:2] + 0.1140 * tensor_b[2:3]
        else:
            img_a = tensor_a
            img_b = tensor_b

        if mode in ["appeared", "gained", "new"]:
            mask = torch.relu(img_b - img_a)
        elif mode in ["disappeared", "lost", "gone"]:
            mask = torch.relu(img_a - img_b)
        else:
            mask = torch.abs(img_b - img_a)

        masks[idx] = mask.squeeze()

    return masks


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

        diff_tensors = []
        for mask in diff_masks.values():
            if torch.is_tensor(mask):
                t = mask.detach().cpu().float()
            else:
                t = torch.from_numpy(np.asarray(mask, dtype=np.float32))
            while t.ndim > 2 and (t.shape[0] == 1 or t.shape[-1] == 1):
                t = t.squeeze()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            diff_tensors.append(t)

        if len(diff_tensors) == 0:
            continue

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
    Prints a clear layer header banner once to stdout for all figures that follow.
    """
    clean_layer = f"FEATURES.{layer_name.upper()}.CONV" if not layer_name.upper().startswith("FEATURES") else layer_name.upper()
    print(f"\n=======================================================")
    print(f"📸 SPATIAL DIFFERENCE MASK ANALYSIS FOR LAYER: {clean_layer}")
    print(f"=======================================================\n")

    # Comparison 1: Fine-Tuned minus Natural (I_B = FT, I_A = NAT)
    plot_difference_grid_pairwise(
        nat_patterns,
        ft_patterns,
        title_suffix="(Fine-Tuned - Natural)",
        mode=mode
    )

    # Comparison 2: Screen-from-Scratch minus Natural (I_B = SCR, I_A = NAT)
    plot_difference_grid_pairwise(
        nat_patterns,
        scr_patterns,
        title_suffix="(Screen Scratch - Natural)",
        mode=mode
    )

    # Comparison 3: Screen-from-Scratch minus Fine-Tuned (I_B = SCR, I_A = FT)
    plot_difference_grid_pairwise(
        ft_patterns,
        scr_patterns,
        title_suffix="(Screen Scratch - Fine-Tuned)",
        mode=mode
    )


# ==============================================================================
# Fourier analysis & Normalized Difference Anisotropy Index (NDI)
# ==============================================================================

#  Fourier analysis & Normalized Difference Anisotropy Index (NDI)
def _step1_preprocess(image_data):
    """
    Step 1: Convert image to 2D grayscale array and crop to a central square.
    This guarantees equal grid dimensions for balanced 2D FFT frequency sampling.
    """
    if torch.is_tensor(image_data):
        image_data = image_data.detach().cpu().numpy()

    image_data = np.asarray(image_data, dtype=np.float64)
    # Squeeze extra leading or trailing singleton batch/channel dimensions (e.g. (1, H, W) or (1, 1, H, W))
    while image_data.ndim > 2 and (image_data.shape[0] == 1 or image_data.shape[-1] == 1):
        image_data = np.squeeze(image_data)
        
    if image_data.ndim == 3:
        if image_data.shape[0] == 3:    # CHW format (3, H, W)
            gray = 0.2989 * image_data[0] + 0.5870 * image_data[1] + 0.1140 * image_data[2]
        elif image_data.shape[2] == 3:  # HWC format (H, W, 3)
            gray = 0.2989 * image_data[:, :, 0] + 0.5870 * image_data[:, :, 1] + 0.1140 * image_data[:, :, 2]
        else:
            gray = np.mean(image_data, axis=0)
    elif image_data.ndim == 2:
        gray = image_data
    else:
        gray = np.squeeze(image_data)

    # Crop to central square of size min(h, w)
    h, w = gray.shape
    size = min(h, w)
    dy = (h - size) // 2
    dx = (w - size) // 2
    return gray[dy:dy+size, dx:dx+size]

def compute_fourier_spectrum(image_gray, dc_radius=5, delta=21, use_log=False, use_hann=True):
    """
    Step 2: Apply a circular spatial mask (and optional 2D Hann spatial window), compute 2D FFT, and calculate
    Normalized Difference Index (NDI) across cardinal and oblique sectors.
    Optionally computes NDI using log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
    Optionally applies 2D Hann windowing when use_hann=True (default True) to eliminate boundary spectral leakage.
    The angular sector window size is configurable via `delta` (default 21°).
    """
    h, w = image_gray.shape
    cy, cx = h // 2, w // 2
    
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) / 2.0
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # 1. Circular spatial crop & optional 2D Hann windowing
    circular_mask = dist_from_center <= radius
    if use_hann:
        window_2d = np.outer(np.hanning(h), np.hanning(w))
        masked_image = image_gray * circular_mask * window_2d
    else:
        masked_image = image_gray * circular_mask
    
    # 2. 2D Fast Fourier Transform
    f_transform = np.fft.ifftshift(masked_image)
    f_transform = np.fft.fft2(f_transform)
    f_shift = np.fft.fftshift(f_transform)
    
    magnitude = np.abs(f_shift)
    log_magnitude = np.log(1 + magnitude)
    
    # 3. Frequency sector masks (exclude DC center component and corner un-inscribed frequencies)
    # Measurement windows: ± delta degrees around target centers (default delta = 21°)
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = dist_from_center
    
    mask_dc = (R > dc_radius) & (R <= radius)
    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    mask_cardinal = ((theta < delta) | (theta > 180 - delta) | ((theta > c_low) & (theta < c_high))) & mask_dc
    mask_oblique = (((theta >= o1_low) & (theta <= o1_high)) | ((theta >= o2_low) & (theta <= o2_high))) & mask_dc
    
    spectrum = log_magnitude if use_log else magnitude
    
    cardinal_energy = np.mean(spectrum[mask_cardinal]) if np.count_nonzero(mask_cardinal) > 0 else 0.0
    oblique_energy = np.mean(spectrum[mask_oblique]) if np.count_nonzero(mask_oblique) > 0 else 0.0
    
    total_energy = cardinal_energy + oblique_energy
    if total_energy == 0:
        ndi = 0.0
    else:
        ndi = (cardinal_energy - oblique_energy) / total_energy
        
    return magnitude, log_magnitude, ndi

def compute_rotational_anisotropy(image_data, dc_radius=5, delta=21, use_log=True, use_hann=True):
    """
    Preprocesses image data (grayscale + square crop) and computes Fourier NDI metrics.
    Optionally returns log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
    Toggles 2D Hann spatial window preconditioning via use_hann=True (default True).
    """
    image_gray = _step1_preprocess(image_data)
    mag, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
    spectrum = log_mag if use_log else mag
    return spectrum, ndi

def _get_subplot_letter(index):
    """Generates subfigure lettering (a), (b), ..., (z), (aa), (ab)..."""
    if index < 26:
        return chr(ord('a') + index)
    else:
        first = chr(ord('a') + (index // 26) - 1)
        second = chr(ord('a') + (index % 26))
        return f"{first}{second}"


def plot_fourier_diagnostic(image_data, title="Fourier Diagnostic", dc_radius=5, delta=21, log_scale=False, use_log=False, use_hann=True, normalize=False, figsize=(18, 4.5)):
    """
    Generates a 4-panel visual diagnostic figure:
    (a) Spatial Image (Circular Masked)
    (b) Log Magnitude Fourier Spectrum
    (c) Cardinal (Red) vs Oblique (Blue) Sector Overlay
    (d) 1D Angular Energy Distribution Plot
    Prints a legend mapping table before displaying the plot.
    """
    image_gray = _step1_preprocess(image_data)
    magnitude, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
    spectrum = log_mag if use_log else magnitude
    
    h, w = image_gray.shape
    cy, cx = h // 2, w // 2
    
    Y_grid, X_grid = np.indices((h, w))
    theta = np.degrees(np.arctan2(cy - Y_grid, X_grid - cx)) % 180
    R = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
    max_radius = min(h, w) / 2.0
    
    mask_dc = (R > dc_radius) & (R <= max_radius)
    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    mask_cardinal = ((theta < delta) | (theta > 180 - delta) | ((theta > c_low) & (theta < c_high))) & mask_dc
    mask_oblique = (((theta >= o1_low) & (theta <= o1_high)) | ((theta >= o2_low) & (theta <= o2_high))) & mask_dc
    
    # Calculate 1D Angular Energy Profile
    angles = np.arange(0, 180, 1)
    angular_energy = []
    for a in angles:
        diff = np.abs(theta - a)
        diff = np.minimum(diff, 180 - diff)
        bin_mask = (diff <= 2) & mask_dc
        cnt = np.count_nonzero(bin_mask)
        angular_energy.append(np.mean(spectrum[bin_mask]) if cnt > 0 else 0.0)
    angular_energy = np.array(angular_energy)
    if normalize and np.max(angular_energy) > 0:
        angular_energy = angular_energy / np.max(angular_energy)
        
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Panel 1: Masked Spatial Image
    circular_mask = R <= max_radius
    masked_img = image_gray * circular_mask
    axes[0].imshow(masked_img, cmap='gray')
    axes[0].set_title("(a)", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Log Magnitude Spectrum
    im1 = axes[1].imshow(log_mag, cmap='magma')
    axes[1].set_title("(b)", fontsize=11, fontweight='bold')
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
    axes[2].set_title("(c)", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    # Panel 4: 1D Angular Energy Distribution
    axes[3].plot(angles, angular_energy, color='darkgreen', lw=2)
    axes[3].axvspan(0, delta, color='red', alpha=0.2, label=f'Cardinal (0°/180° ±{delta}°)')
    axes[3].axvspan(c_low, c_high, color='red', alpha=0.2, label=f'Cardinal (90° ±{delta}°)')
    axes[3].axvspan(180 - delta, 180, color='red', alpha=0.2)
    
    axes[3].axvspan(o1_low, o1_high, color='blue', alpha=0.2, label=f'Oblique (45° ±{delta}°)')
    axes[3].axvspan(o2_low, o2_high, color='blue', alpha=0.2, label=f'Oblique (135° ±{delta}°)')
    
    axes[3].set_xticks([0, 45, 90, 135, 180])
    axes[3].set_xlim(0, 180)
    
    if log_scale:
        axes[3].set_yscale('log')
        pos_vals = angular_energy[angular_energy > 0]
        min_pos = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
        axes[3].set_ylim(bottom=max(1e-4, min_pos * 0.5))
        axes[3].set_ylabel("Norm. Energy (Log Scale)" if normalize else "Spectral Energy (Log Scale)", fontsize=9)
    else:
        axes[3].set_ylabel("Norm. Energy" if normalize else "Spectral Energy", fontsize=9)

    axes[3].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[3].set_title("(d)", fontsize=11, fontweight='bold')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    # Print out subplot legend mapping table before plotting
    print("\n" + "="*70)
    print(f"📊 FOURIER DIAGNOSTIC SUBPLOT LEGEND MAPPING: ({title})")
    print("="*70)
    print("  (a) Spatial Image (Grayscale & Masked)")
    print("  (b) 2D Log Magnitude Fourier Spectrum")
    print(f"  (c) Cardinal & Oblique Sector Overlay (NDI = {ndi:.4f})")
    print("  (d) 1D Angular Energy Distribution Profile")
    print("="*70 + "\n")

    plt.tight_layout()
    plt.show()
    return fig

def analyze_layer_filters(filter_images, dc_radius=5, delta=21, use_log=True, use_hann=True):
    """Averages spectral transformations and NDI anisotropy across all channels in a target layer."""
    if len(filter_images) == 0:
        return np.zeros((80, 80)), 0.0

    sample_spec, _ = compute_rotational_anisotropy(filter_images[0], dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
    h, w = sample_spec.shape
    
    avg_spectrum = np.zeros((h, w), dtype=np.float64)
    anisotropy_scores = []
    
    for img in filter_images:
        spec, ndi = compute_rotational_anisotropy(img, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
        avg_spectrum += spec
        anisotropy_scores.append(ndi)
        
    avg_spectrum /= len(filter_images)
    avg_anisotropy = np.mean(anisotropy_scores)
    
    return avg_spectrum, avg_anisotropy


def compute_layer_angular_distribution(filter_images, dc_radius=5, delta=21, use_log=False, use_hann=True, normalize=False):
    """
    Computes averaged 1D angular energy distribution profile and NDI anisotropy score
    across all filter images in a target layer.
    Optionally accumulates log-magnitude spectrum when use_log=True or linear magnitude spectrum when use_log=False.
    Toggles 2D Hann spatial window preconditioning via use_hann=True (default True).
    Optionally normalizes peak energy to 1.0 when normalize=True (default False to preserve raw energy values for absolute comparison).
    """
    angles = np.arange(0, 180, 1)
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
        magnitude, log_mag, ndi = compute_fourier_spectrum(image_gray, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
        anisotropy_scores.append(ndi)

        spectrum = log_mag if use_log else magnitude

        img_energy = []
        for a in angles:
            diff = np.abs(theta - a)
            diff = np.minimum(diff, 180 - diff)
            bin_mask = (diff <= 2) & mask_dc
            cnt = np.count_nonzero(bin_mask)
            img_energy.append(np.mean(spectrum[bin_mask]) if cnt > 0 else 0.0)
        total_angular_energy += np.array(img_energy)

    avg_angular_energy = total_angular_energy / len(filter_images)
    if normalize and np.max(avg_angular_energy) > 0:
        avg_angular_energy = avg_angular_energy / np.max(avg_angular_energy)

    avg_anisotropy = np.mean(anisotropy_scores)
    return angles, avg_angular_energy, avg_anisotropy


def plot_fourier_cmap_grid(filter_data, models, layers, dc_radius=5, delta=21, log_scale=False, use_log=True, use_hann=True, figsize=(14, 11), dpi=150, save_path="fourier_cmap_grid.png"):
    """
    Plots a grid (len(models) x len(layers)) of averaged 2D Fourier Magnitude Spectra (CMAP).
    Optionally toggles log-magnitude transformation via use_log=True (default True).
    Optionally applies 2D Hann windowing via use_hann=True (default True).
    Optionally applies log-scale color normalization via log_scale=True.
    Prints a summary text table of Anisotropy Index (NDI) for each model & layer.
    Subfigure panels are titled (a), (b), (c)... with a mapping legend printed to terminal stdout.
    """
    anisotropy_results = {m: {} for m in models}

    title_prefix = "Log-Magnitude" if use_log else "Linear Magnitude"
    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)

    k = 0
    legend_entries = []
    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]

            # Process structural metrics
            avg_spectrum, anisotropy_index = analyze_layer_filters(images_list, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
            anisotropy_results[model_name][layer_name] = anisotropy_index

            letter = _get_subplot_letter(k)
            legend_entries.append((letter, model_name.upper(), f"FEATURES.{layer_name.upper()}.CONV", anisotropy_index))
            k += 1

            # Plot spectrum maps
            ax = axs[i, j] if len(models) > 1 and len(layers) > 1 else (axs[i] if len(models) > 1 else axs[j])
            
            if log_scale:
                pos_vals = avg_spectrum[avg_spectrum > 0]
                vmin = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
                norm = LogNorm(vmin=max(1e-4, vmin), vmax=max(np.max(avg_spectrum), vmin * 10))
                im = ax.imshow(avg_spectrum, cmap='magma', norm=norm)
            else:
                im = ax.imshow(avg_spectrum, cmap='magma')

            ax.set_title(f"({letter})", fontsize=11, fontweight='bold', pad=6)
            ax.axis('off')

            # Single colorbar anchor per row to prevent visual cluttering
            if j == len(layers) - 1:
                fig.colorbar(im, ax=ax, shrink=0.7, label='Log Spectral Intensity')

    # Print out subplot legend mapping table before plotting
    print("\n" + "="*76)
    print(f"📊 2D FOURIER SPECTRA ({title_prefix.upper()}) SUBPLOT LEGEND MAPPING:")
    print("="*76)
    print(f"{'SUBPLOT':<9} | {'MODEL':<14} | {'LAYER':<22} | {'ANISOTROPY INDEX (NDI)':<22}")
    print("-" * 76)
    for let, m_name, l_name, score in legend_entries:
        print(f"({let}){' ':<5} | {m_name:<14} | {l_name:<22} | {score:.4f}")
    print("="*76 + "\n")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.show()

    return anisotropy_results


def plot_fourier_angular_distribution_grid(filter_data, models, layers, dc_radius=5, delta=21, log_scale=False, use_log=False, use_hann=True, normalize=False, plot_differences=True, figsize=(16, 11), dpi=150, save_path="fourier_angular_distribution_grid.png", save_diff_path="fourier_angular_difference_grid.png"):
    """
    Plots a grid (len(models) x len(layers)) of averaged 1D Angular Energy Distributions.
    Highlights Cardinal (0°/180°, 90°) and Oblique (45°, 135°) angular sectors.
    Enforces shared y_min and y_max per layer column across all models for unnormalized direct comparisons.
    Subfigure panels are titled (a), (b), (c)... with a mapping legend printed to terminal stdout.
    """
    anisotropy_results = {m: {} for m in models}

    fig, axs = plt.subplots(len(models), len(layers), figsize=figsize, dpi=dpi)

    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    # Pass 1: Compute 1D angular energy distributions for all model & layer combinations
    energy_matrix = {}
    anisotropy_matrix = {}
    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            images_list = filter_data[model_name][layer_name]
            angles, avg_angular_energy, anisotropy_index = compute_layer_angular_distribution(
                images_list, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize
            )
            energy_matrix[(i, j)] = avg_angular_energy
            anisotropy_matrix[(i, j)] = anisotropy_index
            anisotropy_results[model_name][layer_name] = anisotropy_index

    # Pass 2: Calculate per-layer y_min and y_max across all models so each layer column shares unified y-limits
    layer_ylim = {}
    for j in range(len(layers)):
        layer_energies = [energy_matrix[(i, j)] for i in range(len(models))]
        all_vals = np.concatenate(layer_energies) if len(layer_energies) > 0 else np.array([0.0])

        if log_scale:
            pos_vals = all_vals[all_vals > 0]
            min_pos = np.min(pos_vals) if len(pos_vals) > 0 else 1e-4
            max_pos = np.max(all_vals) if len(all_vals) > 0 else 1.0
            layer_ylim[j] = (max(1e-4, min_pos * 0.5), max(1e-4, max_pos * 1.15))
        else:
            min_val = float(np.min(all_vals))
            max_val = float(np.max(all_vals))
            val_range = max_val - min_val
            margin = val_range * 0.06 if val_range > 0 else (abs(max_val) * 0.06 if max_val != 0 else 0.1)
            bottom = 0.0 if (min_val >= 0 and not log_scale) else (min_val - margin)
            top = max_val + margin
            layer_ylim[j] = (bottom, top)

    # Pass 3: Plot energy profiles and apply unified per-layer y-limits
    k = 0
    legend_entries = []
    for i, model_name in enumerate(models):
        for j, layer_name in enumerate(layers):
            avg_angular_energy = energy_matrix[(i, j)]
            anisotropy_index = anisotropy_matrix[(i, j)]

            letter = _get_subplot_letter(k)
            legend_entries.append((letter, model_name.upper(), f"FEATURES.{layer_name.upper()}.CONV", anisotropy_index))
            k += 1

            ax = axs[i, j] if len(models) > 1 and len(layers) > 1 else (axs[i] if len(models) > 1 else axs[j])

            # Plot 1D energy profile
            ax.plot(angles, avg_angular_energy, color='darkgreen', lw=2, label="Profile")

            # Highlight Cardinal (red) and Oblique (blue) sectors
            ax.axvspan(0, delta, color='red', alpha=0.2, label=f'Cardinal (±{delta}°)')
            ax.axvspan(c_low, c_high, color='red', alpha=0.2)
            ax.axvspan(180 - delta, 180, color='red', alpha=0.2)

            ax.axvspan(o1_low, o1_high, color='blue', alpha=0.2, label=f'Oblique (±{delta}°)')
            ax.axvspan(o2_low, o2_high, color='blue', alpha=0.2)

            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlim(0, 180)

            if log_scale:
                ax.set_yscale('log')

            # Enforce unified per-layer y-limits across models
            ax.set_ylim(layer_ylim[j])

            ax.set_title(f"({letter})", fontsize=11, fontweight='bold', pad=6)
            ax.grid(True, linestyle='--', alpha=0.5)

            if j == 0:
                ylabel = ("Norm. Energy" if normalize else "Spectral Energy") + (" (Log Scale)" if log_scale else "")
                ax.set_ylabel(ylabel, fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("Angle θ (degrees)", fontsize=9)

    # Print out subplot legend mapping table before plotting
    print("\n" + "="*76)
    print("📊 1D ANGULAR ENERGY DISTRIBUTION SUBPLOT LEGEND MAPPING:")
    print("="*76)
    print(f"{'SUBPLOT':<9} | {'MODEL':<14} | {'LAYER':<22} | {'ANISOTROPY INDEX (NDI)':<22}")
    print("-" * 76)
    for let, m_name, l_name, score in legend_entries:
        print(f"({let}){' ':<5} | {m_name:<14} | {l_name:<22} | {score:.4f}")
    print("="*76 + "\n")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.show()

    has_baseline = any(_find_model_key(filter_data, alias) is not None for alias in ["natural", "nonsc", "baseline"])
    if plot_differences and has_baseline:
        diff_save = save_diff_path if save_diff_path else ("fourier_angular_difference_grid.png" if save_path else None)
        plot_fourier_angular_difference_grid(filter_data, layers, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize, dpi=dpi, save_path=diff_save)

    return anisotropy_results


def _find_model_key(filter_data, target_name):
    """Robustly matches model keys in filter_data dictionary regardless of exact string key format."""
    if not isinstance(filter_data, dict):
        return None
    if target_name in filter_data:
        return target_name
    norm_target = target_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key in filter_data.keys():
        norm_key = str(key).lower().replace("-", "").replace("_", "").replace(" ", "")
        if norm_target == norm_key or norm_target in norm_key or norm_key in norm_target:
            return key
        if norm_target in ["natural", "nonsc", "nonscreen", "baseline"] and any(alias in norm_key for alias in ["natural", "nonsc", "nonscreen", "baseline"]):
            return key
        if norm_target in ["finetuned", "ft", "fine"] and any(alias in norm_key for alias in ["finetuned", "ft", "fine"]):
            return key
        if norm_target in ["screen", "sc", "scratch"] and any(alias in norm_key for alias in ["screen", "sc", "scratch"]):
            return key
    return None


def _get_model_acronym(model_name):
    """Maps model name strings to clean, standard acronyms: FT, NONSC, SC."""
    name_clean = str(model_name).lower().replace("-", "").replace("_", "").replace(" ", "")
    if "finetuned" in name_clean or name_clean == "ft" or "fine" in name_clean:
        return "FT"
    elif "natural" in name_clean or "nonsc" in name_clean or "baseline" in name_clean or "nonscreen" in name_clean or "nat" in name_clean:
        return "NONSC"
    elif "screen" in name_clean or "scratch" in name_clean or name_clean == "sc" or "scr" in name_clean:
        return "SC"
    return str(model_name).upper()[:5]


def plot_fourier_angular_difference_grid(filter_data, layers, pairs=None, dc_radius=5, delta=21, use_log=False, use_hann=True, normalize=False, figsize=(16, 11), dpi=150, save_path="fourier_angular_difference_grid.png"):
    """
    Plots a grid (len(pairs) x len(layers)) of 1D Angular Energy Difference Profiles:
    - Pair 1: Fine-Tuned minus Natural (FT - NONSC)
    - Pair 2: Screen (Scratch) minus Natural (SC - NONSC)
    - Pair 3: Fine-Tuned minus Screen (Scratch) (FT - SC)
    Enforces shared symmetrical y_min and y_max per layer column across all difference pairs.
    Subfigure panels are titled (a), (b), (c)... with a mapping legend printed to terminal stdout.
    """
    if pairs is None:
        pairs = [
            ("fine-tuned", "natural", "FINE-TUNED - NATURAL"),
            ("screen", "natural", "SCREEN SCRATCH - NATURAL"),
            ("fine-tuned", "screen", "FINE-TUNED - SCREEN SCRATCH")
        ]
    diff_pairs = pairs

    fig, axs = plt.subplots(len(diff_pairs), len(layers), figsize=figsize, dpi=dpi)

    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    # Pass 1: Compute difference profiles for all pairs and layers
    diff_matrix = {}
    table_data = []
    k = 0
    for i, (model_b_req, model_a_req, pair_title) in enumerate(diff_pairs):
        model_b_name = _find_model_key(filter_data, model_b_req)
        model_a_name = _find_model_key(filter_data, model_a_req)
        if not model_b_name or not model_a_name:
            print(f"⚠️ Warning: Model pair '{model_b_req}' ({model_b_name}) vs '{model_a_req}' ({model_a_name}) not found in filter_data keys ({list(filter_data.keys())}). Skipping.")
            continue

        for j, layer_name in enumerate(layers):
            images_a = filter_data[model_a_name][layer_name]
            images_b = filter_data[model_b_name][layer_name]

            angles, energy_a, ndi_a = compute_layer_angular_distribution(images_a, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize)
            angles, energy_b, ndi_b = compute_layer_angular_distribution(images_b, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize)

            diff_energy = energy_b - energy_a
            delta_ndi = ndi_b - ndi_a

            diff_matrix[(i, j)] = diff_energy
            letter = _get_subplot_letter(k)
            table_data.append((letter, layer_name, str(model_b_name).upper(), str(model_a_name).upper(), ndi_b, ndi_a, delta_ndi))
            k += 1

    if not diff_matrix:
        print("⚠️ Warning: No valid model pairs were found to compute difference profiles.")
        plt.close(fig)
        return []

    # Pass 2: Calculate per-layer y_min and y_max across all difference pairs so each layer column shares unified y-limits
    layer_diff_ylim = {}
    for j in range(len(layers)):
        column_diffs = [diff_matrix[(i, j)] for i in range(len(diff_pairs)) if (i, j) in diff_matrix]
        all_diffs = np.concatenate(column_diffs) if len(column_diffs) > 0 else np.array([0.0])
        max_abs_diff = float(np.max(np.abs(all_diffs))) if len(all_diffs) > 0 else 0.5
        margin = max_abs_diff * 0.08 if max_abs_diff > 0 else 0.1
        ylim_val = max_abs_diff + margin
        layer_diff_ylim[j] = (-ylim_val, +ylim_val)

    # Pass 3: Plot difference profiles and apply unified per-layer y-limits
    colors = ['purple', 'crimson', 'teal']
    k = 0
    for i, (model_b_req, model_a_req, pair_title) in enumerate(diff_pairs):
        model_b_name = _find_model_key(filter_data, model_b_req)
        model_a_name = _find_model_key(filter_data, model_a_req)
        for j, layer_name in enumerate(layers):
            if (i, j) not in diff_matrix:
                continue
            diff_energy = diff_matrix[(i, j)]
            letter = table_data[k][0]
            k += 1

            ax = axs[i, j] if len(diff_pairs) > 1 and len(layers) > 1 else (axs[i] if len(diff_pairs) > 1 else axs[j])

            color = colors[i % len(colors)]
            acronym_b = _get_model_acronym(model_b_name)
            acronym_a = _get_model_acronym(model_a_name)
            ax.plot(angles, diff_energy, color=color, lw=2, label=f"Δ Energy ({acronym_b} - {acronym_a})")

            # Reference baseline at y = 0
            ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1)

            # Shaded positive (gain) and negative (loss) regions
            ax.fill_between(angles, diff_energy, 0, where=(diff_energy >= 0), color=color, alpha=0.15)
            ax.fill_between(angles, diff_energy, 0, where=(diff_energy < 0), color='gray', alpha=0.15)

            # Highlight Cardinal (red) and Oblique (blue) sectors
            ax.axvspan(0, delta, color='red', alpha=0.15, label=f'Cardinal (±{delta}°)')
            ax.axvspan(c_low, c_high, color='red', alpha=0.15)
            ax.axvspan(180 - delta, 180, color='red', alpha=0.15)

            ax.axvspan(o1_low, o1_high, color='blue', alpha=0.15, label=f'Oblique (±{delta}°)')
            ax.axvspan(o2_low, o2_high, color='blue', alpha=0.15)

            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_xlim(0, 180)

            # Enforce unified per-layer y-limits across difference pairs
            ax.set_ylim(layer_diff_ylim[j])

            ax.set_title(f"({letter})", fontsize=11, fontweight='bold', pad=6)
            ax.grid(True, linestyle='--', alpha=0.5)

            if j == 0:
                ax.set_ylabel(f"Δ Energy ({acronym_b} - {acronym_a})", fontsize=9, fontweight='bold')
            if i == len(diff_pairs) - 1:
                ax.set_xlabel("Angle θ (degrees)", fontsize=9)

    # Print out subplot legend mapping table before plotting
    print("\n" + "="*85)
    print("📊 1D ANGULAR ENERGY DIFFERENCE PROFILES SUBPLOT LEGEND MAPPING:")
    print("="*85)
    print(f"{'SUBPLOT':<9} | {'SUBTRACTION PAIR (B - A)':<30} | {'LAYER':<18} | {'Δ NDI (B-A)':<12}")
    print("-" * 85)
    for let, l_name, m_b, m_a, n_b, n_a, d_ndi in table_data:
        comp_str = f"{m_b} - {m_a}"
        print(f"({let}){' ':<5} | {comp_str:<30} | FEATURES.{l_name.upper()}.CONV | {d_ndi:+10.4f}")
    print("="*85 + "\n")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.show()

    return table_data


def analyze_difference_masks_fourier(
    filter_data,
    layers,
    pairs=None,
    mode="all",
    dc_radius=5,
    delta=21,
    log_scale=False,
    use_log=False,
    use_hann=True,
    normalize=False,
    figsize=(16, 10),
    dpi=150,
    save_path_prefix="fourier_diff_masks"
):
    """
    Applies the full 2D and 1D Fourier spectral analysis sequence and NDI Anisotropy Index metrics
    directly to spatial DIFFERENCE MASKS across model domain pairs.
    Saves PNG figures if save_path_prefix is provided.
    """
    if pairs is None:
        pairs = [
            ("fine-tuned", "natural", "FINE-TUNED - NATURAL"),
            ("screen", "natural", "SCREEN SCRATCH - NATURAL"),
            ("fine-tuned", "screen", "FINE-TUNED - SCREEN SCRATCH")
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

        available_models = list(filter_data.keys())
        def _find_matching_model_key(target_name):
            norm_target = target_name.lower().replace("-", "").replace("_", "")
            for key in available_models:
                if key.lower().replace("-", "").replace("_", "") == norm_target:
                    return key
            return target_name

        for model_b_req, model_a_req, label in pairs:
            model_b = _find_matching_model_key(model_b_req)
            model_a = _find_matching_model_key(model_a_req)
            if model_b not in filter_data or model_a not in filter_data:
                print(f"⚠️ Warning: Model pair '{model_b_req}' vs '{model_a_req}' not found in filter_data keys ({available_models}). Skipping.")
                continue

            pair_key = f"{model_b}_minus_{model_a}"
            diff_models.append(pair_key)
            diff_filter_data[pair_key] = {}
            for layer_name in layers:
                if layer_name not in filter_data[model_a] or layer_name not in filter_data[model_b]:
                    print(f"⚠️ Warning: Layer '{layer_name}' not found for pair ({model_b}, {model_a}). Skipping.")
                    continue
                p_a = filter_data[model_a][layer_name]
                p_b = filter_data[model_b][layer_name]
                diff_filter_data[pair_key][layer_name] = list(compute_difference_masks(p_a, p_b, mode=current_mode).values())

        print(f"\n=======================================================")
        print(f"📊 {mode_title}")
        print(f"=======================================================\n")

        save_path = f"{save_path_prefix}_{current_mode}.png" if save_path_prefix else None

        anisotropy_angular = plot_fourier_angular_distribution_grid(
            filter_data=diff_filter_data,
            models=diff_models,
            layers=layers,
            dc_radius=dc_radius,
            delta=delta,
            log_scale=log_scale,
            use_log=use_log,
            use_hann=use_hann,
            normalize=normalize,
            plot_differences=False,
            figsize=figsize,
            dpi=dpi,
            save_path=save_path
        )
        results[current_mode] = diff_filter_data

    return results if len(modes_to_run) > 1 else results[modes_to_run[0]]


def plot_spatial_difference_masks_grid(
    filter_data,
    layers,
    pairs=None,
    mode="both",
    nrow=None,
    figsize=(16, 11),
    dpi=150,
    cmap="hot",
    save_path="spatial_difference_masks_grid.png"
):
    """
    Plots grids (len(pairs) x len(layers)) of Spatial Difference Masks (SDMs) between filter activation maximizations:
    - Pair 1: Fine-Tuned minus Natural (FT - NONSC)
    - Pair 2: Screen (Scratch) minus Natural (SC - NONSC)
    - Pair 3: Fine-Tuned minus Screen (Scratch) (FT - SC)

    Modes:
    - "both"        : Runs both "appeared" and "disappeared" modes (Default)
    - "appeared"    : ReLU(B - A) = newly learned/gained features
    - "disappeared" : ReLU(A - B) = erased/lost features
    - "absolute"    : |B - A|     = total magnitude change
    - "all"         : Runs "appeared", "disappeared", and "absolute" modes
    """
    if pairs is None:
        pairs = [
            ("fine-tuned", "natural", "FINE-TUNED - NATURAL"),
            ("screen", "natural", "SCREEN SCRATCH - NATURAL"),
            ("fine-tuned", "screen", "FINE-TUNED - SCREEN SCRATCH")
        ]

    if isinstance(mode, (list, tuple)):
        modes_to_run = mode
    elif mode == "all":
        modes_to_run = ["appeared", "disappeared", "absolute"]
    elif mode in ["both", "separate", "two"]:
        modes_to_run = ["appeared", "disappeared"]
    else:
        modes_to_run = [mode]

    results = {}

    for current_mode in modes_to_run:
        fig, axs = plt.subplots(len(pairs), len(layers), figsize=figsize, dpi=dpi)

        grid_matrix = {}
        table_entries = []
        k = 0

        for i, (model_b_req, model_a_req, pair_label) in enumerate(pairs):
            model_b = _find_model_key(filter_data, model_b_req)
            model_a = _find_model_key(filter_data, model_a_req)

            if not model_b or not model_a or model_b not in filter_data or model_a not in filter_data:
                print(f"⚠️ Warning: Pair '{model_b_req}' ({model_b}) vs '{model_a_req}' ({model_a}) not found in filter_data keys ({list(filter_data.keys())}). Skipping.")
                continue

            for j, layer_name in enumerate(layers):
                if layer_name not in filter_data[model_a] or layer_name not in filter_data[model_b]:
                    print(f"⚠️ Warning: Layer '{layer_name}' not found for pair ({model_b}, {model_a}). Skipping.")
                    continue

                p_a = filter_data[model_a][layer_name]
                p_b = filter_data[model_b][layer_name]

                diff_masks = compute_difference_masks(p_a, p_b, mode=current_mode)
                mask_list = list(diff_masks.values())

                # Convert to stacked tensor for PyTorch grid layout
                tensors = []
                for m in mask_list:
                    if torch.is_tensor(m):
                        t = m.detach().cpu().float()
                    else:
                        t = torch.from_numpy(np.asarray(m, dtype=np.float32))
                    while t.ndim > 2 and (t.shape[0] == 1 or t.shape[-1] == 1):
                        t = t.squeeze()
                    if t.ndim == 2:
                        t = t.unsqueeze(0)
                    tensors.append(t)

                if len(tensors) > 0:
                    batch = torch.stack(tensors)  # (N, 1, H, W)
                    n_filters = len(tensors)
                    grid_cols = nrow if nrow is not None else int(np.ceil(np.sqrt(n_filters)))
                    grid = vutils.make_grid(batch, nrow=grid_cols, padding=2, normalize=True)
                    grid_np = grid[0].numpy()  # Extract single channel (H, W) for colormapping
                    grid_matrix[(i, j)] = grid_np

                letter = _get_subplot_letter(k)
                table_entries.append((letter, layer_name, _get_model_acronym(model_b), _get_model_acronym(model_a), len(tensors)))
                k += 1

        if not grid_matrix:
            print(f"⚠️ Warning: No valid spatial difference mask grids were constructed for mode '{current_mode}'.")
            plt.close(fig)
            continue

        # Plot subplots
        k = 0
        for i, (model_b_req, model_a_req, pair_label) in enumerate(pairs):
            model_b = _find_model_key(filter_data, model_b_req)
            model_a = _find_model_key(filter_data, model_a_req)
            for j, layer_name in enumerate(layers):
                if (i, j) not in grid_matrix:
                    continue
                grid_img = grid_matrix[(i, j)]
                letter = table_entries[k][0]
                k += 1

                ax = axs[i, j] if len(pairs) > 1 and len(layers) > 1 else (axs[i] if len(pairs) > 1 else axs[j])
                im = ax.imshow(grid_img, cmap=cmap, vmin=0.0, vmax=1.0)
                ax.set_title(f"({letter})", fontsize=11, fontweight='bold', pad=6)
                ax.axis('off')

                if j == len(layers) - 1:
                    fig.colorbar(im, ax=ax, shrink=0.7, label=f"SDM Intensity ({current_mode.capitalize()})")

        print("\n" + "="*80)
        print(f"🖼️ SPATIAL DIFFERENCE MASKS (SDMs) GRID SUBPLOT MAPPING (MODE: {current_mode.upper()}):")
        print("="*80)
        print(f"{'SUBPLOT':<9} | {'LAYER':<20} | {'MODEL B':<12} | {'MODEL A':<12} | {'FILTERS':<10}")
        print("-" * 80)
        for let, l_name, m_b, m_a, n_f in table_entries:
            print(f"({let}){' ':<5} | {l_name:<20} | {m_b:<12} | {m_a:<12} | {n_f:<10}")
        print("="*80 + "\n")

        plt.tight_layout()
        if save_path:
            out_file = save_path if len(modes_to_run) == 1 else (
                save_path.replace(".png", f"_{current_mode}.png") if save_path.endswith(".png") else f"{save_path}_{current_mode}.png"
            )
            plt.savefig(out_file, format='png', bbox_inches='tight', dpi=dpi)
        plt.show()

        results[current_mode] = grid_matrix

    return results if len(modes_to_run) > 1 else (list(results.values())[0] if results else None)


plot_spatial_difference_grid = plot_spatial_difference_masks_grid
plot_difference_masks_grid = plot_spatial_difference_masks_grid







def plot_synthetic_fourier_difference_demo(dc_radius=5, delta=21, use_log=False, use_hann=True, normalize=False, figsize=(16, 4.5), dpi=150, save_path="fourier_synthetic_demo.png"):
    """
    Generates a synthetic demonstration using two controlled 2D spatial gratings:
    - Image 1: 90° frequency energy (horizontal spatial grating)
    - Image 2: 45° frequency energy (135° spatial grating)

    Plots individual 1D Angular Energy distributions and calculates the resulting
    difference profile ΔEnergy = Energy_45° - Energy_90°.
    """
    h, w = 120, 120
    y, x = np.indices((h, w))

    # Image 1: 90° frequency energy (horizontal spatial grating y)
    img_90 = np.sin(2 * np.pi * 0.15 * y)

    # Image 2: 45° frequency energy (diagonal spatial grating x+y)
    img_45 = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))

    mag_90, log_90, ndi_90 = compute_fourier_spectrum(img_90, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)
    mag_45, log_45, ndi_45 = compute_fourier_spectrum(img_45, dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann)

    angles, energy_90, _ = compute_layer_angular_distribution([img_90], dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize)
    angles, energy_45, _ = compute_layer_angular_distribution([img_45], dc_radius=dc_radius, delta=delta, use_log=use_log, use_hann=use_hann, normalize=normalize)
    diff_energy = energy_45 - energy_90

    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # Panel 1: Combined 2D Log Spectrum
    im0 = axes[0].imshow(log_90 + log_45, cmap="magma")
    axes[0].set_title("(a)", fontsize=11, fontweight="bold", pad=6)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: 1D Angular Energy Distributions
    axes[1].plot(angles, energy_90, color="darkgreen", lw=2, label="90° Grating (Cardinal)")
    axes[1].plot(angles, energy_45, color="darkorange", lw=2, label="45° Grating (Oblique)")
    axes[1].axvspan(0, delta, color="red", alpha=0.15)
    axes[1].axvspan(c_low, c_high, color="red", alpha=0.15, label=f"Cardinal (±{delta}°)")
    axes[1].axvspan(180 - delta, 180, color="red", alpha=0.15)
    axes[1].axvspan(o1_low, o1_high, color="blue", alpha=0.15, label=f"Oblique (±{delta}°)")
    axes[1].axvspan(o2_low, o2_high, color="blue", alpha=0.15)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_xlim(0, 180)
    axes[1].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[1].set_ylabel("Norm. Energy" if normalize else "Spectral Energy", fontsize=9)
    axes[1].set_title("(b)", fontsize=11, fontweight="bold", pad=6)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Difference Profile
    axes[2].plot(angles, diff_energy, color="purple", lw=2, label="Δ Energy (45° - 90°)")
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.6, lw=1)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy >= 0), color="purple", alpha=0.2)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy < 0), color="gray", alpha=0.2)
    axes[2].axvspan(0, delta, color="red", alpha=0.15)
    axes[2].axvspan(c_low, c_high, color="red", alpha=0.15)
    axes[2].axvspan(180 - delta, 180, color="red", alpha=0.15)
    axes[2].axvspan(o1_low, o1_high, color="blue", alpha=0.15)
    axes[2].axvspan(o2_low, o2_high, color="blue", alpha=0.15)
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].set_xlim(0, 180)
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Δ Norm. Energy" if normalize else "Δ Energy", fontsize=9)
    axes[2].set_title("(c)", fontsize=11, fontweight="bold", pad=6)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Print out subplot legend mapping table before plotting
    print("\n" + "="*70)
    print("📊 SYNTHETIC FOURIER DIFFERENCE DEMO SUBPLOT LEGEND MAPPING:")
    print("="*70)
    print("  (a) Combined 2D Fourier Spectra (90° + 45° Gratings)")
    print("  (b) Individual 1D Angular Energy Distributions")
    print(f"  (c) Angular Energy Difference Profile (ΔNDI = {ndi_45 - ndi_90:+.4f})")
    print("="*70 + "\n")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.show()
    return fig






