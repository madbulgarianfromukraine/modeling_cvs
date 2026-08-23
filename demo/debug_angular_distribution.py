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
    plot_fourier_diagnostic,
    _step1_preprocess,
)


def generate_synthetic_gratings(h=120, w=120):
    """
    Generates two controlled lists of synthetic test images:
    1. canonical_gratings: Horizontal, vertical, and grid sine gratings (0°, 90° orientation).
    2. oblique_gratings: 45° and 135° diagonal sine gratings.
    """
    y, x = np.indices((h, w))

    # 1. Canonical Gratings (Orientations at 0°, 90°, and combined 0°+90°)
    c1 = np.sin(2 * np.pi * 0.15 * y)  # Horizontal spatial lines -> Fourier peak at 90°
    c2 = np.sin(2 * np.pi * 0.15 * x)  # Vertical spatial lines -> Fourier peak at 0° / 180°
    c3 = np.sin(2 * np.pi * 0.25 * y)  # Higher frequency horizontal lines -> Fourier peak at 90°
    c4 = np.sin(2 * np.pi * 0.15 * x) + np.sin(2 * np.pi * 0.15 * y)  # Grid -> Peaks at 0°, 90°, 180°

    canonical_gratings = [c1, c2, c3, c4]

    # 2. Oblique Gratings (Orientations at 45° and 135°)
    o1 = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))  # 135° spatial lines -> Fourier peak at 45°
    o2 = np.sin(2 * np.pi * 0.15 * (x - y) / np.sqrt(2))  # 45° spatial lines -> Fourier peak at 135°
    o3 = np.sin(2 * np.pi * 0.25 * (x + y) / np.sqrt(2))  # Higher frequency 135° lines -> Fourier peak at 45°
    o4 = np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2)) + np.sin(
        2 * np.pi * 0.15 * (x - y) / np.sqrt(2)
    )  # Diagonal cross hatch -> Peaks at 45°, 135°

    oblique_gratings = [o1, o2, o3, o4]

    return canonical_gratings, oblique_gratings


def debug_noise_floor_solutions(dc_radius=5, delta=21):
    """
    Step 3: Test whether background noise stacks up and compare 3 solutions:
    1. Original Raw Distribution
    2. Adaptive Baseline Subtraction (E_clean = ReLU(E - min(E)))
    3. 2D Hann Window Preconditioning
    """
    print("\n" + "=" * 80)
    print("🔬 STEP 3: NOISE FLOOR ACCUMULATION & SOLUTION COMPARISON")
    print("=" * 80)

    canonical_gratings, oblique_gratings = generate_synthetic_gratings()

    # Raw Distributions
    angles, energy_c_raw, _ = compute_layer_angular_distribution(
        canonical_gratings, dc_radius=dc_radius, delta=delta, use_log=False
    )
    _, energy_o_raw, _ = compute_layer_angular_distribution(
        oblique_gratings, dc_radius=dc_radius, delta=delta, use_log=False
    )

    # 1. Adaptive Baseline Subtraction: E_clean = ReLU(E - min(E)) / max(...)
    floor_c = np.min(energy_c_raw)
    floor_o = np.min(energy_o_raw)
    energy_c_sub = np.maximum(0, energy_c_raw - floor_c)
    energy_c_sub = energy_c_sub / np.max(energy_c_sub)

    energy_o_sub = np.maximum(0, energy_o_raw - floor_o)
    energy_o_sub = energy_o_sub / np.max(energy_o_sub)

    # 2. Hann Windowed Preconditioning
    hann_c = []
    for img in canonical_gratings:
        h, w = img.shape
        w2d = np.outer(np.hanning(h), np.hanning(w))
        hann_c.append(img * w2d)

    hann_o = []
    for img in oblique_gratings:
        h, w = img.shape
        w2d = np.outer(np.hanning(h), np.hanning(w))
        hann_o.append(img * w2d)

    _, energy_c_hann, _ = compute_layer_angular_distribution(
        hann_c, dc_radius=dc_radius, delta=delta, use_log=False
    )
    _, energy_o_hann, _ = compute_layer_angular_distribution(
        hann_o, dc_radius=dc_radius, delta=delta, use_log=False
    )

    print("\n📊 NOISE FLOOR LEVEL COMPARISON (Non-Peak Off-Axis Energy):")
    print(f"  • Raw Profile Baseline Noise Floor        : Canonical = {floor_c:.4f}, Oblique = {floor_o:.4f}")
    print(f"  • Adaptive Baseline Subtracted Noise Floor: Canonical = {np.min(energy_c_sub):.4f}, Oblique = {np.min(energy_o_sub):.4f}")
    print(f"  • Hann Windowed Profile Noise Floor       : Canonical = {np.min(energy_c_hann):.4f}, Oblique = {np.min(energy_o_hann):.4f}")

    # Plot Noise Floor Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), dpi=150)
    fig.suptitle("Noise Floor Suppression Techniques in 1D Angular Distribution", fontsize=13, fontweight="bold")

    # Panel 1: Original Raw Profile
    axes[0].plot(angles, energy_c_raw, color="red", lw=2, label="Canonical")
    axes[0].plot(angles, energy_o_raw, color="blue", lw=2, label="Oblique")
    axes[0].set_title(f"(a) Raw Profile (Baseline Noise ≈ {floor_c:.2f})", fontsize=10, fontweight="bold")
    axes[0].set_xticks([0, 45, 90, 135, 180])
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    # Panel 2: Adaptive Subtraction Profile
    axes[1].plot(angles, energy_c_sub, color="red", lw=2, label="Canonical")
    axes[1].plot(angles, energy_o_sub, color="blue", lw=2, label="Oblique")
    axes[1].set_title("(b) Adaptive Baseline Subtracted (E - min(E))", fontsize=10, fontweight="bold")
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    # Panel 3: Hann Windowed Profile
    axes[2].plot(angles, energy_c_hann, color="red", lw=2, label="Canonical")
    axes[2].plot(angles, energy_o_hann, color="blue", lw=2, label="Oblique")
    axes[2].set_title("(c) 2D Hann Window Preconditioning", fontsize=10, fontweight="bold")
    axes[2].set_xticks([0, 45, 90, 135, 180])
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "demo", "noise_suppression_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"🖼️ Saved noise suppression figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    debug_noise_floor_solutions()
