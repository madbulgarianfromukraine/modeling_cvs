# %% [code]

import subprocess
subprocess.run(["pip", "install", "grad-cam"], check=True)

import torch
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



def plot_random_gradcam_edges(model, target_layers, samples_dict, class_names, file_name, num_samples=5, seed=317,
                              enrico_resize=False, caltech_resize=False, save_images=False):
    """
    Executes GradCAM on a random subset of samples from samples_dict, 
    overlaying the resulting heatmaps directly onto their Canny edge maps.
    
    Parameters:
    - model: The trained PyTorch model.
    - target_layers: Target convolutional layer(s).
    - samples_dict: Dictionary containing {class_idx: image_tensor}.
    - class_names: List or dictionary mapping class indices to string names.
    - file_name: Base string path/name (e.g., 'results/gradcam_run').
    - num_samples: Number of random unique classes to visualize.
    - save_images: Whether to save the rendered figures to disk (default False).
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
        
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Use enumerate to track the loop count for naming the files sequentially
        for idx, target in enumerate(selected_targets, start=1):
            image = samples_dict[target]

            if enrico_resize:
                image = TF.resize(image, [320, 180])
            elif caltech_resize:
                image = TF.resize(image, [300, 200])
                
            heatmap_output = cam(input_tensor=image.unsqueeze(0), targets=[ClassifierOutputTarget(target)])
            heatmap = heatmap_output[0]
            
            rgb_image = image.permute(1, 2, 0).cpu().numpy()
            rgb_image = np.clip(rgb_image, 0, 1)
            
            edge = get_canny_edge(rgb_image)
            visualization = show_cam_on_image(edge, heatmap, use_rgb=True)
            
            # Render the 2-image comparison canvas side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # --- Panel 1: The Baseline Reference ---
            axes[0].imshow(rgb_image)
            axes[0].set_title(f"Original: {class_names[target]}", fontsize=13, fontweight='bold')
            axes[0].axis('off')
            
            # --- Panel 2: The Merged Structural Heatmap ---
            axes[1].imshow(visualization)
            axes[1].set_title("GradCAM Overlayed and Canny Edges", fontsize=13, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # --- SAVE STEP ---
            if save_images:
                output_path = f"{file_name}_sample_{idx}.png"
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
            
            # Display plot in the notebook
            plt.show()
            
            # Force close the figure to flush memory back to system
            plt.close(fig)