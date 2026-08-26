# %% [code]
# %% [code]
# %% [code]

import subprocess
subprocess.run(["pip", "install", "grad-cam"], check=True)

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import torchvision.transforms.functional as TF
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_one_sample_per_class(dataset, idx_to_class):
    
    samples_per_class = {}
    
    # Iterate through the dataset
    for i in range(len(dataset)):
        # Stop if we found a sample for every class
        if len(samples_per_class) == len(idx_to_class):
            break
            
        data, target = dataset[i]
        
        target_idx = int(target)
        
        # If we haven't found a sample for this class yet, save it
        if target_idx not in samples_per_class:
            samples_per_class[target_idx] = data
            print(f"✅ Found class: {idx_to_class[target_idx]}")
            
    return samples_per_class


def get_canny_edge(img, threshold1=30, threshold2=80):
    """
    Function to get the canny edge of an image
    """
    # Gray scale the image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray * 255
    gray = gray.astype(np.uint8)

    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get the edge
    edge = 255 - cv2.Canny(gray, threshold1, threshold2)
    edge = np.stack([edge] * 3, axis=-1) / 255

    return edge



def _get_class_label(class_names, target_idx):
    """Safely retrieves string class label from list or dict without raising IndexError or KeyError."""
    if isinstance(class_names, dict):
        if target_idx in class_names:
            return str(class_names[target_idx])
        if str(target_idx) in class_names:
            return str(class_names[str(target_idx)])
    elif isinstance(class_names, (list, tuple)):
        if 0 <= int(target_idx) < len(class_names):
            return str(class_names[int(target_idx)])
    return f"Class_{target_idx}"


def plot_random_gradcam_edges(model, target_layers, samples_dict, class_names, file_name="gradcam", num_samples=5, seed=317,
                              enrico_resize=False, caltech_resize=False, save_images=True, figsize=(6.5, 3.5), image_weight=0.7):
    """
    Executes GradCAM on a random subset of samples from samples_dict, 
    overlaying the resulting heatmaps directly onto their Canny edge maps.
    
    Parameters:
    - model: The trained PyTorch model.
    - target_layers: Target convolutional layer(s).
    - samples_dict: Dictionary containing {class_idx: image_tensor}.
    - class_names: List or dictionary mapping class indices to string names.
    - file_name: Base string path/name (e.g., 'gradcam_baseline').
    - num_samples: Number of random unique classes to visualize.
    - save_images: Whether to save the rendered figures to disk (default True).
    - figsize: Canvas dimensions tuple for the side-by-side comparison figure (default (6.5, 3.5)).
    - image_weight: Blending weight for background edges in show_cam_on_image (default 0.7 for normal brightness).
    """
    model.eval()
    
    available_targets = list(samples_dict.keys())
    k = min(num_samples, len(available_targets))
    if enrico_resize and caltech_resize:
        raise Exception("Cannot specify both resizes at the same time")
        
    random.seed(seed)
    selected_targets = random.sample(available_targets, k)
    
    if isinstance(target_layers, list):
        target_layers = target_layers
    else:
        target_layers = [target_layers]

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Use enumerate to track the loop count for naming the files sequentially
        for idx, target in enumerate(selected_targets, start=1):
            image = samples_dict[target]

            if enrico_resize:
                image = TF.resize(image, [320, 180])
            elif caltech_resize:
                image = TF.resize(image, [300, 200])

            input_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(input_tensor)
                num_classes = out.shape[-1]
                pred_class = int(out.argmax(dim=-1).item())

            target_idx = int(target)
            if target_idx < 0 or target_idx >= num_classes:
                print(f"⚠️ Warning: Target class index {target_idx} is out of bounds for model with {num_classes} output classes. Defaulting to top predicted class index {pred_class}.")
                target_for_cam = pred_class
            else:
                target_for_cam = target_idx

            class_label = _get_class_label(class_names, target_idx)
                
            heatmap_output = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_for_cam)])
            heatmap = heatmap_output[0]
            
            rgb_image = image.permute(1, 2, 0).cpu().numpy()
            min_v, max_v = rgb_image.min(), rgb_image.max()
            if max_v > min_v:
                rgb_image = (rgb_image - min_v) / (max_v - min_v)
            rgb_image = np.clip(rgb_image, 0.0, 1.0)
            
            edge = get_canny_edge(rgb_image)
            visualization = show_cam_on_image(edge, heatmap, use_rgb=True, image_weight=image_weight)

            
            # Render the 2-image comparison canvas side-by-side (compact size)
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
            
            # --- Panel 1: The Baseline Reference ---
            axes[0].imshow(rgb_image)
            axes[0].set_title(f"Original: {class_label}", fontsize=10, fontweight='bold', pad=4)
            axes[0].axis('off')
            
            # --- Panel 2: The Merged Structural Heatmap ---
            axes[1].imshow(visualization)
            axes[1].set_title("GradCAM Overlayed on Edges", fontsize=10, fontweight='bold', pad=4)
            axes[1].axis('off')
            
            plt.tight_layout(pad=0.5)
            
            # --- SAVE STEP ---
            if save_images:
                class_str = class_label.lower().strip().replace(" ", "_").replace("/", "_")
                output_path = f"{file_name}_{class_str}.png"
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                plt.savefig(output_path, bbox_inches='tight', dpi=300)

            # Display plot in the notebook
            plt.show()
            
            # Force close the figure to flush memory back to system
            plt.close(fig)