import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.maximization.analysis import (
    compute_fourier_spectrum,
    _step1_preprocess
)


def run_log_magnitude_justification_demo(save_path=None, dpi=150):
    """
    Demonstrates the necessity of log-magnitude dynamic range compression M(u,v) = log(1 + |F(u,v)|).
    
    Creates a 3-panel figure:
    1. Spatial Grayscale Image (Circular Masked)
    2. Linear Raw Magnitude Spectrum |F_shift(u,v)| (Un-applied log: exp(M) - 1)
    3. Log-Compressed Magnitude Spectrum M(u,v) = log(1 + |F_shift(u,v)|)
    """
    h, w = 120, 120
    y, x = np.indices((h, w))

    # Synthetic multi-directional pattern with cardinal + oblique features
    sample_image = (
        np.sin(2 * np.pi * 0.12 * y) + 
        0.6 * np.cos(2 * np.pi * 0.18 * x) + 
        0.4 * np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))
    )

    image_gray = _step1_preprocess(sample_image)
    _, log_magnitude, ndi = compute_fourier_spectrum(image_gray, dc_radius=5)

    # Un-apply log transformation to recover linear magnitude spectrum: |F| = exp(M) - 1
    raw_linear_magnitude = np.exp(log_magnitude) - 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)
    fig.suptitle("Justification of Log-Magnitude Compression in Fourier Analysis", fontsize=14, fontweight="bold")

    # Panel 1: Spatial Image
    im0 = axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title(r"Spatial Preprocessed Image" + "\n" + r"$I(x, y)$", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Raw Linear Magnitude Spectrum (Un-applied Log)
    im1 = axes[1].imshow(raw_linear_magnitude, cmap="magma")
    axes[1].set_title(r"Linear Magnitude Spectrum (No Log)" + "\n" + r"$|F_{\text{shift}}(u, v)| = \exp(M) - 1$", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.set_yscale('linear')

    # Panel 3: Log-Compressed Magnitude Spectrum (As in Code)
    im2 = axes[2].imshow(log_magnitude, cmap="magma")
    axes[2].set_title(r"Log-Compressed Spectrum (In Code)" + "\n" + r"$M(u, v) = \log(1 + |F_{\text{shift}}(u, v)|)$", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Log-magnitude justification figure saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    demo_save_file = os.path.join(PROJECT_ROOT, "demo", "log_magnitude_justification.png")
    run_log_magnitude_justification_demo(save_path=demo_save_file)
