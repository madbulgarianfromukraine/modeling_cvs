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
    compute_layer_angular_distribution,
)


def run_synthetic_fourier_difference_demo(dc_radius=5, use_log=False, figsize=(16, 4.5), dpi=150, save_path=None):
    """
    Synthetic Demonstration:
    Generates two controlled spatial gratings:
    - Image 1: 90° frequency energy (horizontal spatial grating)
    - Image 2: 45° frequency energy (135° diagonal spatial grating)

    Computes 1D Angular Energy profiles for both synthetic patterns, and subtracts
    them to display the Difference Profile ΔEnergy = Energy_45° - Energy_90°.
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
    axes[1].axvspan(0, 21, color="red", alpha=0.15)
    axes[1].axvspan(69, 111, color="red", alpha=0.15, label="Cardinal (±21°)")
    axes[1].axvspan(159, 180, color="red", alpha=0.15)
    axes[1].axvspan(24, 66, color="blue", alpha=0.15, label="Oblique (±21°)")
    axes[1].axvspan(114, 156, color="blue", alpha=0.15)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_xlim(0, 180)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[1].set_ylabel("Norm. Energy", fontsize=9)
    axes[1].set_title("1D Energy Distributions", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Difference Profile (Δ Energy)
    axes[2].plot(angles, diff_energy, color="purple", lw=2, label="Δ Energy (45° - 90°)")
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.6, lw=1)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy >= 0), color="purple", alpha=0.2)
    axes[2].fill_between(angles, diff_energy, 0, where=(diff_energy < 0), color="gray", alpha=0.2)
    axes[2].axvspan(0, 21, color="red", alpha=0.15)
    axes[2].axvspan(69, 111, color="red", alpha=0.15)
    axes[2].axvspan(159, 180, color="red", alpha=0.15)
    axes[2].axvspan(24, 66, color="blue", alpha=0.15)
    axes[2].axvspan(114, 156, color="blue", alpha=0.15)
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].set_xlim(0, 180)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Δ Norm. Energy", fontsize=9)
    axes[2].set_title(f"Difference Profile (ΔNDI: {ndi_45 - ndi_90:+.3f})", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Demo figure saved to: {save_path}")

    plt.show()
    return fig


def run_appeared_vs_disappeared_mask_demo(dc_radius=5, figsize=(16, 4.5), dpi=150, save_path=None):
    """
    Demonstrates directional feature separation between Appeared (ReLU(B - A)) and Disappeared (ReLU(A - B)) features:
    - Baseline Pattern A (Natural): 45° Diagonal Grating
    - Target Pattern B (Screen)   : 90° Horizontal Grating
    """
    h, w = 120, 120
    y, x = np.indices((h, w))

    # Pattern A (Natural): Diagonal 45° grating
    img_a = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))

    # Pattern B (Screen): Horizontal 90° grating
    img_b = np.sin(2 * np.pi * 0.15 * y)

    # 1. Appeared mask (ReLU(B - A))
    mask_appeared = np.maximum(0, img_b - img_a)

    # 2. Disappeared mask (ReLU(A - B))
    mask_disappeared = np.maximum(0, img_a - img_b)

    # Compute 1D Fourier energy profiles
    angles, energy_app, ndi_app = compute_layer_angular_distribution([mask_appeared], dc_radius=dc_radius)
    angles, energy_dis, ndi_dis = compute_layer_angular_distribution([mask_disappeared], dc_radius=dc_radius)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.suptitle("Directional Feature Separation: Appeared (B - A) vs Disappeared (A - B) Features", fontsize=14, fontweight="bold")

    # Panel 1: Appeared Spatial Mask
    im0 = axes[0].imshow(mask_appeared, cmap="hot")
    axes[0].set_title(f"Appeared Features (B - A)\nReLU(I_B - I_A)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Disappeared Spatial Mask
    im1 = axes[1].imshow(mask_disappeared, cmap="hot")
    axes[1].set_title(f"Disappeared Features (A - B)\nReLU(I_A - I_B)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: 1D Fourier Energy Comparison
    axes[2].plot(angles, energy_app, color="darkred", lw=2, label=f"Appeared (NDI: {ndi_app:+.3f})")
    axes[2].plot(angles, energy_dis, color="navy", lw=2, linestyle="--", label=f"Disappeared (NDI: {ndi_dis:+.3f})")
    axes[2].axvspan(0, 21, color="red", alpha=0.15)
    axes[2].axvspan(69, 111, color="red", alpha=0.15, label="Cardinal (±21°)")
    axes[2].axvspan(159, 180, color="red", alpha=0.15)
    axes[2].axvspan(24, 66, color="blue", alpha=0.15, label="Oblique (±21°)")
    axes[2].axvspan(114, 156, color="blue", alpha=0.15)
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].set_xlim(0, 180)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Norm. Energy", fontsize=9)
    axes[2].set_title("Fourier Angular Energy Distribution", fontsize=11, fontweight="bold")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Appeared vs Disappeared figure saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    demo_save_file1 = os.path.join(PROJECT_ROOT, "demo", "synthetic_fourier_demo.png")
    demo_save_file2 = os.path.join(PROJECT_ROOT, "demo", "appeared_vs_disappeared_demo.png")
    run_synthetic_fourier_difference_demo(save_path=demo_save_file1)
    run_appeared_vs_disappeared_mask_demo(save_path=demo_save_file2)
