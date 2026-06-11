# %% [code]
# %% [code]
# %% [code]
import os
import cv2
import glob
import queue
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torchvision.utils as vutils
from gpu_utils import clean_all_gpu_memory
from model import PseudoAlexNet
from fine_tuning import reset_head

class FilterVisualizer:
    def __init__(self, model, layer_num):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = model.to(self.device).eval()
        
        for param in self.model.parameters():
            param.requires_grad_(False)
            
        self.layer_num = layer_num
        
        target_layer = self.model.features[self.layer_num].conv
        if hasattr(target_layer, 'out_channels'):
            self.total_filters = target_layer.out_channels
        elif hasattr(target_layer, 'num_features'): 
            self.total_filters = target_layer.num_features
        else:
            self.total_filters = None

    def visualize(self, filter_indices=None, iterations=30, lr=1.0, size=(30, 20)):
        if filter_indices is None:
            if self.total_filters is None:
                raise ValueError("Could not determine filter count. Please pass indices manually.")
            indices = list(range(self.total_filters))
        elif isinstance(filter_indices, int):
            indices = [filter_indices]
        else:
            indices = filter_indices
            
        num_to_viz = len(indices)
        
        img = torch.rand(num_to_viz, 3, size[0], size[1], device=self.device)
        img.requires_grad_()

        for i in range(iterations):
            if img.grad is not None:
                img.grad.zero_()
            
            x = img
            for idx, layer in enumerate(self.model.features):
                x = layer(x)
                if idx == self.layer_num:
                    break
            
            batch_idx = torch.arange(num_to_viz, device=self.device)
            filter_idx_tensor = torch.tensor(indices, device=self.device)
            
            target_activations = x[batch_idx, filter_idx_tensor, :, :]
            loss = torch.mean(target_activations)
            
            loss.backward()
            
            with torch.no_grad():
                grads = img.grad
                grad_norm = torch.sqrt(torch.mean(torch.square(grads), dim=[1, 2, 3], keepdim=True)) + 1e-5
                grads = grads / grad_norm
                
                img.data += lr * grads

        return self.deprocess_batch(img, indices)

    def deprocess_batch(self, img_batch, indices):
        img_batch = img_batch.detach().cpu()
        processed_dict = {}
        
        for i, idx in enumerate(indices):
            x = img_batch[i]
            
            x = x - x.mean()
            x = x / (x.std() + 1e-5)
            x = x * 0.1
            x = x + 0.5
            x = torch.clamp(x, 0.0, 1.0)
            
            processed_dict[idx] = x
            
        return processed_dict



def get_tensor_grid(batch_tensor, nrow=5, padding=2, brightness_offset=0.0):
    """
    Processes a batch of tensors into a single grid image array.
    
    Parameters:
        batch_tensor: A batch of image tensors.
        nrow: Number of images displayed in each row of the grid.
        padding: Amount of padding between images.
        brightness_offset: Float value to add to pixel intensities (e.g., 0.1 to 0.3).
        
    Returns:
        np.array: The grid image in (H, W, C) format, bounded between [0.0, 1.0].
    """
    # 1. Create the grid (C, H, W). normalize=True forces values to [0.0, 1.0]
    grid = vutils.make_grid(batch_tensor, nrow=nrow, padding=padding, normalize=True)
    
    # 2. Convert to (H, W, C) and move to CPU/NumPy
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # 3. Apply brightness offset and clip to valid image float bounds
    if brightness_offset != 0.0:
        grid_np = np.clip(grid_np + brightness_offset, 0.0, 1.0)
        
    return grid_np


def visualization_process(vis_queue, vis_finish_event, model_kwargs, layers_to_viz, output_dir, 
                          stage_name, input_features, output_features):
    os.makedirs(output_dir, exist_ok=True)
    
    device_id = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    device = torch.device(device_id)
    
    # model_constructor should be a function returning your uninitialized model architecture
    model = PseudoAlexNet(**model_kwargs).to(device)
    reset_head(model, input_features=input_features, output_features=output_features)

    model.eval()

    while not (vis_finish_event.is_set() and vis_queue.empty()):
        try:
            # timeout allows the loop to periodically check the finish_event if the queue is empty
            epoch, state_dict = vis_queue.get(timeout=3.0)
            
            model.load_state_dict(state_dict)
            
            for layer in layers_to_viz:
                vis = FilterVisualizer(model, layer_num=layer)
                vis.device = device
                vis.model.to(device)
                
                patterns = vis.visualize(filter_indices=None, iterations=100)
                grid_output = get_tensor_grid(list(patterns.values()))
                
                file_path = os.path.join(output_dir, f"{stage_name}_layer_{layer}_epoch_{epoch:03d}.png")
                plt.imsave(file_path, grid_output)
            
        except queue.Empty:
            continue
        finally:
            clean_all_gpu_memory()


def compile_frames_to_video(frames_dir: str, output_video_path: str, fps: int = 2, file_extension: str = "png"):
    """
    Compiles a directory of image frames into an MP4 video.
    
    Args:
        frames_dir (str): Directory containing the saved frame images.
        output_video_path (str): Desired output path and filename (e.g., 'output.mp4').
        fps (int): Frames per second. Default is 2.
        file_extension (str): The image file extension (e.g., 'png', 'jpg').
    """
    # 1. Grab all matching images in the directory
    search_pattern = os.path.join(frames_dir, f"*.{file_extension}")
    images = glob.glob(search_pattern)
    
    if not images:
        print(f"❌ No {file_extension} images found in {frames_dir}.")
        return

    # 2. Sort the images alphabetically/numerically 
    # (Since we used %03d like epoch_001.png, standard sort works perfectly)
    images.sort()
    
    print(f"Found {len(images)} frames. Compiling at {fps} FPS...")

    # 3. Read the first frame to determine the video resolution
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"❌ Failed to read the first image: {images[0]}")
        return
        
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # 4. Initialize the OpenCV Video Writer
    # 'mp4v' is the standard, widely compatible codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # 5. Loop through and write each frame
    for idx, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        
        # Verify frame loaded and matches dimensions
        if frame is not None:
            # OpenCV requires frames to perfectly match the initialized frame_size
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)
                
            video_writer.write(frame)
        else:
            print(f"⚠️ Warning: Could not read frame {image_path}. Skipping.")

    # 6. Release the writer to save the file properly
    video_writer.release()
    print(f"✅ Video successfully saved to: {output_video_path}")