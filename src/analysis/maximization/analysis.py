# %% [code]
# %% [code]
# %% [code]
import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import torchvision.utils as vutils
import scipy.ndimage
from concurrent.futures import ThreadPoolExecutor


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
    f_transform = np.fft.fft2(masked_image)
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

def analyze_layer_filters(filter_images):
    """Averages spectral transformations and NDI anisotropy across all channels in a target layer."""
    if len(filter_images) == 0:
        return np.zeros((80, 80)), 0.0

    sample_log, _ = compute_rotational_anisotropy(filter_images[0])
    h, w = sample_log.shape
    
    avg_log_spectrum = np.zeros((h, w), dtype=np.float64)
    anisotropy_scores = []
    
    for img in filter_images:
        log_mag, ndi = compute_rotational_anisotropy(img)
        avg_log_spectrum += log_mag
        anisotropy_scores.append(ndi)
        
    avg_log_spectrum /= len(filter_images)
    avg_anisotropy = np.mean(anisotropy_scores)
    
    return avg_log_spectrum, avg_anisotropy
