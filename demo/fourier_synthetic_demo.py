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
    axes[1].axvspan(69, 111, color="red", alpha=0.15)
    axes[1].axvspan(159, 180, color="red", alpha=0.15)
    axes[1].axvspan(24, 66, color="blue", alpha=0.15)
    axes[1].axvspan(114, 156, color="blue", alpha=0.15)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_xlim(0, 180)
    axes[1].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[1].set_ylabel("Spectral Energy", fontsize=9)
    axes[1].set_title("1D Energy Distributions", fontsize=11, fontweight="bold")
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
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Δ Energy", fontsize=9)
    axes[2].set_title(f"Difference Profile (ΔNDI: {ndi_45 - ndi_90:+.3f})", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Demo figure saved to: {save_path}")

    plt.show()
    return fig


def run_appeared_vs_disappeared_mask_demo(dc_radius=5, delta=21, figsize=(16, 4.5), dpi=150, save_path=None):
    """
    Demonstrates true Spectral Domain Feature Separation between Appeared (Spectral Gain) and Disappeared (Spectral Loss) features:
    - Model A (Natural)  : 45° Diagonal Grating (Peak at 135°/45°)
    - Model B (Screen)   : 90° Horizontal Grating (Peak at 90°)
    - Spectral Gain (B-A): ReLU(E_B - E_A) -> Clean peak at 90°
    - Spectral Loss (A-B): ReLU(E_A - E_B) -> Clean peak at 135°/45°
    """
    h, w = 120, 120
    y, x = np.indices((h, w))

    # Pattern A (Natural): Diagonal 45° grating
    img_a = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))

    # Pattern B (Screen): Horizontal 90° grating
    img_b = np.sin(2 * np.pi * 0.15 * y)

    # Compute 1D Fourier energy profiles directly for Model A and Model B
    angles, energy_a, ndi_a = compute_layer_angular_distribution([img_a], dc_radius=dc_radius, delta=delta, use_log=False)
    angles, energy_b, ndi_b = compute_layer_angular_distribution([img_b], dc_radius=dc_radius, delta=delta, use_log=False)

    # 1. Spectral Gain (Appeared Features in B relative to A)
    spectral_gain = np.maximum(0, energy_b - energy_a)

    # 2. Spectral Loss (Disappeared Features from A in B)
    spectral_loss = np.maximum(0, energy_a - energy_b)

    c_low, c_high = 90 - delta, 90 + delta
    o1_low, o1_high = 45 - delta, 45 + delta
    o2_low, o2_high = 135 - delta, 135 + delta

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.suptitle("Spectral Domain Feature Separation: Appeared (Gain) vs Disappeared (Loss)", fontsize=14, fontweight="bold")

    # Panel 1: Original Model A vs Model B Fourier Profiles
    axes[0].plot(angles, energy_a, color="navy", lw=2, label=f"Model A (45° Diagonal, NDI: {ndi_a:+.3f})")
    axes[0].plot(angles, energy_b, color="darkred", lw=2, label=f"Model B (90° Horizontal, NDI: {ndi_b:+.3f})")
    axes[0].set_xticks([0, 45, 90, 135, 180])
    axes[0].set_xlim(0, 180)
    axes[0].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[0].set_ylabel("Spectral Energy", fontsize=9)
    axes[0].set_title("Input Fourier Profiles (Model A vs B)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Truly Appeared Features (Spectral Gain: ReLU(E_B - E_A))
    axes[1].plot(angles, spectral_gain, color="darkred", lw=2.5, label="Spectral Gain: ReLU(E_B - E_A)")
    axes[1].fill_between(angles, spectral_gain, 0, color="darkred", alpha=0.2)
    axes[1].axvspan(0, delta, color="red", alpha=0.15)
    axes[1].axvspan(c_low, c_high, color="red", alpha=0.15)
    axes[1].axvspan(180 - delta, 180, color="red", alpha=0.15)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].set_xlim(0, 180)
    axes[1].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[1].set_ylabel("Gain Spectral Energy", fontsize=9)
    axes[1].set_title("✨ Appeared Features (Peak at 90° Cardinal)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Truly Disappeared Features (Spectral Loss: ReLU(E_A - E_B))
    axes[2].plot(angles, spectral_loss, color="navy", lw=2.5, linestyle="--", label="Spectral Loss: ReLU(E_A - E_B)")
    axes[2].fill_between(angles, spectral_loss, 0, color="navy", alpha=0.2)
    axes[2].axvspan(o1_low, o1_high, color="blue", alpha=0.15)
    axes[2].axvspan(o2_low, o2_high, color="blue", alpha=0.15)
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].set_xlim(0, 180)
    axes[2].set_xlabel("Angle θ (degrees)", fontsize=9)
    axes[2].set_ylabel("Loss Spectral Energy", fontsize=9)
    axes[2].set_title("🍂 Disappeared Features (Peak at 135° Oblique)", fontsize=11, fontweight="bold")
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
