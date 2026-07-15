# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Any, Callable, List, Union, Tuple

CATEGORIES = {'settings', 'calculator', 'other', 'terms', 'search', 'form', 'tutorial', 'gallery', 'mediaplayer', 'list', 'bare', 'chat', 'editor', 'modal', 'news', 'login', 'profile', 'camera', 'menu', 'maps'}
EXCLUDE_SMALL = {'calculator', 'camera', 'maps', 'chat', 'editor'}
EXCLUDE_HIGH_OVERLAP = {'profile', 'other', 'editor', 'mediaplayer', 'maps', 'search', 'chat', 'calculator'} #{"editor", "gallery", "mediaplayer", "modal", "news", "profile"} these are not pure classes nat images in images so to say

def get_allowed_classes():
    return CATEGORIES - EXCLUDE_SMALL - EXCLUDE_HIGH_OVERLAP

def make_screen_base_transform(resize: Tuple[int, int] = (300, 200)):
    return v2.Compose([
        v2.Resize(size=resize),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

class CustomEnricoDataset(Dataset):
    def __init__(self, 
                 root: str,
                 screen_ids: List[str],
                 labels_dict: Dict[str, str],
                 class_to_idx: Dict[str, int],
                 use_wireframes: bool = False,
                 transform: Optional[Callable] = None, 
                 transform_to_class: Optional[Dict[str, Union[Callable, List[Callable]]]] = None,
                 augment_for_each: Optional[List[Callable]] = None):
        """
        Internal constructor. Use CustomEnricoDataset.create_splits() to initialize 
        train and val datasets comfortably from a Kaggle path.
        """
        self.root = root
        
        if use_wireframes:
            self.file_ext = "png"
            self.img_dir = os.path.join(self.root, "wireframes/wireframes")
        else:
            self.file_ext = "jpg"
            self.img_dir = os.path.join(self.root, "screenshots/screenshots")
        self.transform = transform
        self.transform_to_class = transform_to_class or {}
        
        self.screen_ids = screen_ids
        self.labels_dict = labels_dict
        self.class_to_idx = class_to_idx

        self.samples = []
        for screen_id in self.screen_ids:
            label_str = self.labels_dict.get(screen_id)
            if label_str:
                self.samples.append((screen_id, None)) 

                for aug in augment_for_each or []:
                    self.samples.append((screen_id, aug))
                    
                if label_str in self.transform_to_class:
                    augs = self.transform_to_class[label_str]
                    augs = [augs] if not isinstance(augs, list) else augs
                    for aug in augs:
                        self.samples.append((screen_id, aug))

    @classmethod
    def create_splits(cls, 
                      root: str, 
                      val_size: float = 0.15, 
                      test_size: float = 0.15,
                      seed: int = 42,
                      use_wireframes: bool = False,
                      transform: Optional[Callable] = None,
                      train_transform: Optional[Callable] = None,
                      eval_transform: Optional[Callable] = None,
                      transform_to_class: Optional[Dict] = None,
                      augment_for_each: Optional[List] = None,
                      allowed_classes : Optional[Union[List[str], bool]] = True) -> Tuple['CustomEnricoDataset', 'CustomEnricoDataset', 'CustomEnricoDataset']:
        """
        Reads the Kaggle directory, performs a stratified train/val/test split,
        and returns all three datasets configured automatically.
        """
        from sklearn.model_selection import train_test_split
        
        csv_path = os.path.join(root, "design_topics.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}. Check your Kaggle input path.")

        df = pd.read_csv(csv_path)
        df['screen_id'] = df['screen_id'].astype(str)

        # --- NEW: Filter for large/specific classes first ---
        if allowed_classes:
            if isinstance(allowed_classes, bool):
                allowed_classes = get_allowed_classes()
            df = df[df['topic'].isin(allowed_classes)].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError("Filtered DataFrame is empty! Check your allowed_classes list.")
    
        
        # --- STEP 1: Split Train vs. (Val + Test) ---
        temp_size = val_size + test_size
        train_df, temp_df = train_test_split(
            df, 
            test_size=temp_size, 
            stratify=df['topic'], 
            random_state=seed
        )

        relative_test_size = test_size / temp_size
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=relative_test_size, 
            stratify=temp_df['topic'], 
            random_state=seed
        )

        # Build shared label mappings
        labels_dict = dict(zip(df['screen_id'], df['topic']))
        unique_labels = sorted(df['topic'].unique())
        class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        train_transform = train_transform if train_transform is not None else transform
        eval_transform = eval_transform if eval_transform is not None else transform

        def _build_dataset(split_df, split_transform, split_augment_for_each=None, split_transform_to_class=None):
            return cls(
                root=root,
                screen_ids=split_df['screen_id'].tolist(),
                labels_dict=labels_dict,
                class_to_idx=class_to_idx,
                use_wireframes=use_wireframes,
                transform=split_transform,
                transform_to_class=split_transform_to_class,
                augment_for_each=split_augment_for_each
            )

        return (
            _build_dataset(train_df, train_transform, augment_for_each, transform_to_class),
            _build_dataset(val_df, eval_transform, None, None),
            _build_dataset(test_df, eval_transform, None, None),
        )

    def __len__(self):
        return len(self.samples)
        
    @staticmethod
    def safe_rgba_to_rgb(image_path: str, bg_color=(255, 255, 255)) -> Image.Image:
        img = Image.open(image_path)
        
        # If it's already RGB or L, just return it converted to RGB
        if img.mode in ('RGB', 'L'):
            return img.convert('RGB')
            
        # If it has an alpha channel (RGBA or LA)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            
            # Create a solid background image
            background = Image.new("RGB", img.size, bg_color)
            
            # Paste the image using its own alpha channel as the mask
            background.paste(img, mask=img.split()[3])
            return background

        # Fallback for any other exotic modes
        return img.convert('RGB')
        
    def __getitem__(self, index):
        screen_id, specific_transform = self.samples[index]
        img_path = os.path.join(self.img_dir, f"{screen_id}.{self.file_ext}") 
        
        image = CustomEnricoDataset.safe_rgba_to_rgb(image_path=img_path)
        
        if specific_transform: 
            image = specific_transform(image)
            
        if self.transform: 
            image = self.transform(image)

        label_idx = self.class_to_idx[self.labels_dict[screen_id]]
        return image, label_idx

    def get_by_label(self, label: str, num_samples: int = 5):
        """Utility to retrieve images belonging to a specific label."""
        results = []
        for i, (screen_id, aug) in enumerate(self.samples):
            if self.labels_dict[screen_id] == label:
                img, _ = self.__getitem__(i)

                img = img.permute(1, 2, 0).detach().cpu().numpy()
                results.append(img)
                
                if len(results) > num_samples:
                    break
        return results
        


def plot_image_list(images, titles=None, cols=4, figsize_multiplier=3):
    """
    Plots a list of images in an organized grid.
    
    Args:
        images (list): A list of NumPy arrays representing images (H, W, C).
        titles (list, optional): A list of strings for the title of each image.
        cols (int): How many images to display per row.
        figsize_multiplier (int): Base size for each subplot to scale the overall figure.
    """
    num_images = len(images)
    if num_images == 0:
        print("The image list is empty.")
        return

    # Calculate exactly how many rows we need
    rows = math.ceil(num_images / cols)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_multiplier, rows * figsize_multiplier))
    
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each image
    for i in range(num_images):
        axes[i].imshow(images[i])
        
        # Add title if provided
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
            
        # Hide the X and Y axes ticks/lines for a cleaner look
        axes[i].axis('off')

    # Turn off the axes for any remaining empty subplots in the grid
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    # Tidy up the layout and display
    plt.tight_layout()
    plt.show()