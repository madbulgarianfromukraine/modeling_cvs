import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Ensure repository root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.maximization.analysis import compute_difference_masks


def run_puppy_directional_difference_demo(image_path=None, figsize=(10, 9), dpi=200, save_path=None):
    """
    Visual Demonstration of Spatial Directional Difference Masks using puppy.jpeg in a 2x2 Grid Layout:
    - Image A (Baseline / Natural) : Puppy + Disappearing Diagonal Patch (Top-Left)
    - Image B (Target / Fine-Tuned): Puppy + Appearing Horizontal Patch (Bottom-Right)
    - Row 2: Appeared Features Mask (ReLU(B - A)) vs Disappeared Features Mask (ReLU(A - B))
    """
    if image_path is None:
        image_path = os.path.join(PROJECT_ROOT, "demo", "images", "puppy.jpeg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Puppy image not found at: {image_path}")

    pil_img = Image.open(image_path).convert('L').resize((240, 240))
    base_img = np.array(pil_img, dtype=np.float32) / 255.0
    h, w = base_img.shape
    y, x = np.indices((h, w))

    # Element 1: Disappearing Diagonal Patch (Added in Image A, top-left region)
    disappearing_patch = np.exp(-((x - 60)**2 + (y - 60)**2) / (2 * 25**2)) * np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))
    disappearing_element = np.maximum(0.0, disappearing_patch) * 0.7

    # Element 2: Appearing Horizontal Patch (Added in Image B, bottom-right region)
    appearing_patch = np.exp(-((x - 170)**2 + (y - 170)**2) / (2 * 25**2)) * np.sin(2 * np.pi * 0.15 * y)
    appearing_element = np.maximum(0.0, appearing_patch) * 0.7

    # Image A: Base + Disappearing Element
    img_a = np.clip(base_img + disappearing_element, 0.0, 1.0)

    # Image B: Base + Appearing Element
    img_b = np.clip(base_img + appearing_element, 0.0, 1.0)

    # Convert to torch RGB tensors for compute_difference_masks
    tensor_a = torch.from_numpy(img_a).float().unsqueeze(0).repeat(3, 1, 1)
    tensor_b = torch.from_numpy(img_b).float().unsqueeze(0).repeat(3, 1, 1)

    diff_appeared = compute_difference_masks({0: tensor_a}, {0: tensor_b}, mode="appeared")[0][0].numpy()
    diff_disappeared = compute_difference_masks({0: tensor_a}, {0: tensor_b}, mode="disappeared")[0][0].numpy()

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Spatial Directional Difference Mask Isolation (Puppy Demonstration)", fontsize=14, fontweight="bold", y=0.98)

    # Panel (0, 0): Modified Image A (Baseline)
    axes[0, 0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Image A (Baseline / Natural)\nPuppy + Diagonal Patch (Top-Left)", fontsize=10, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    # Panel (0, 1): Modified Image B (Target)
    axes[0, 1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Image B (Target / Fine-Tuned / Screen)\nPuppy + Horizontal Patch (Bottom-Right)", fontsize=10, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    # Panel (1, 0): Appeared Features Mask (+)
    im2 = axes[1, 0].imshow(diff_appeared, cmap="hot", vmin=0, vmax=0.7)
    axes[1, 0].set_title("Appeared Features Mask (+)\nReLU(I_B - I_A) [Newly Added Feature]", fontsize=10, fontweight="bold", pad=8)
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel (1, 1): Disappeared Features Mask (-)
    im3 = axes[1, 1].imshow(diff_disappeared, cmap="hot", vmin=0, vmax=0.7)
    axes[1, 1].set_title("Disappeared Features Mask (-)\nReLU(I_A - I_B) [Erased / Lost Feature]", fontsize=10, fontweight="bold", pad=8)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Puppy 2x2 spatial difference demo saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    puppy_img_path = os.path.join(PROJECT_ROOT, "demo", "images", "puppy.jpeg")
    demo_save_file = os.path.join(PROJECT_ROOT, "demo", "directional_difference_masks_demo.png")
    run_puppy_directional_difference_demo(image_path=puppy_img_path, save_path=demo_save_file)
