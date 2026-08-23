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


def run_puppy_directional_difference_demo(image_path=None, figsize=(10, 2.6), dpi=300, save_path=None):
    """
    Visual Demonstration of Spatial Directional Difference Masks using puppy.jpeg in a 1x4 Horizontal Grid Layout:
    - (a) Image A (Baseline / Natural) : Puppy + Disappearing Diagonal Patch (Top-Left)
    - (b) Image B (Target / Fine-Tuned): Puppy + Appearing Horizontal Patch (Bottom-Right)
    - (c) Appeared Features Mask       : ReLU(B - A)
    - (d) Disappeared Features Mask    : ReLU(A - B)
    """
    if image_path is None:
        image_path = os.path.join(PROJECT_ROOT, "demo", "images", "puppy.jpeg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Puppy image not found at: {image_path}")

    pil_img = Image.open(image_path).convert('L').resize((140, 140))
    base_img = np.array(pil_img, dtype=np.float32) / 255.0
    h, w = base_img.shape
    y, x = np.indices((h, w))

    # Element 1: Disappearing Diagonal Patch (Added in Image A, top-left region)
    disappearing_patch = np.exp(-((x - 35)**2 + (y - 35)**2) / (2 * 15**2)) * np.sin(2 * np.pi * 0.15 * (x + y) / np.sqrt(2))
    disappearing_element = np.maximum(0.0, disappearing_patch) * 0.7

    # Element 2: Appearing Horizontal Patch (Added in Image B, bottom-right region)
    appearing_patch = np.exp(-((x - 105)**2 + (y - 105)**2) / (2 * 15**2)) * np.sin(2 * np.pi * 0.15 * y)
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
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.0), dpi=300)

    # Panel 0: Image A
    axes[0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("(a) Image A\n($I_A$)", fontsize=9, fontweight="bold", pad=0)
    axes[0].axis("off")

    # Panel 1: Image B
    axes[1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) Image B\n($I_B$)", fontsize=9, fontweight="bold", pad=0)
    axes[1].axis("off")

    # Panel 2: Appeared Features Mask
    axes[2].imshow(diff_appeared, cmap="hot", vmin=0, vmax=0.7)
    axes[2].set_title("(c) Appeared\nFeatures", fontsize=9, fontweight="bold", pad=0)
    axes[2].axis("off")

    # Panel 3: Disappeared Features Mask
    axes[3].imshow(diff_disappeared, cmap="hot", vmin=0, vmax=0.7)
    axes[3].set_title("(d) Disappeared\nFeatures", fontsize=9, fontweight="bold", pad=0)
    axes[3].axis("off")

    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.05)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        print(f"Puppy 1x4 spatial difference demo saved to: {save_path}")



    plt.show()
    return fig




if __name__ == "__main__":
    puppy_img_path = os.path.join(PROJECT_ROOT, "demo", "images", "puppy.jpeg")
    demo_save_file = os.path.join(PROJECT_ROOT, "demo", "directional_difference_masks_demo.png")
    run_puppy_directional_difference_demo(image_path=puppy_img_path, save_path=demo_save_file)
