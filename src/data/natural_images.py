# %% [code]
# %% [code]
# %% [code]
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import pathlib

from torchvision import datasets
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple

#Caltech 101 functions
def load_dataset(transforms_original=None, transforms_augmented_list=None):
    original_dataset = datasets.Caltech101(root='./data', download=True, transform=transforms_original)
    if transforms_augmented_list is not None and len(transforms_augmented_list) > 0:
        augmented_datasets = [
            datasets.Caltech101(root='./data', download=True, transform=
                                v2.Compose([transforms_original, t]))
            for t in transforms_augmented_list
        ]
        return torch.utils.data.ConcatDataset([original_dataset] + augmented_datasets)
    else:
        return original_dataset
        

def stratified_three_way_split(dataset: Dataset, train_ratio: float, test_ratio: float, 
    val_ratio: float, random_state: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Extracts a stratified train, validation, and test split from a PyTorch Dataset,
    ensuring exact class proportions are maintained across all three.
    """
    # Sanity check to ensure the mathematical ratios make sense
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0!"

    # 1. Safely extract targets regardless of dataset structure
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    elif hasattr(dataset, 'y'):
        targets = dataset.y
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]

    # 2. Format targets and cast to a NumPy array to allow easy advanced indexing later
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list) and torch.is_tensor(targets[0]):
        targets = np.array([t.item() for t in targets])
    else:
        targets = np.array(targets)

    # 3. Generate base dataset indices
    indices = np.arange(len(dataset))

    # --- FIRST SPLIT: Separate Train from the temporary pool (Val + test) ---
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=targets,
        random_state=random_state
    )

    # --- SECOND SPLIT: Separate the temporary pool into test and val ---
    # Calculate the relative size of the val split within the remaining data pool
    # e.g., if val=0.15 and test=0.15, relative_test_size is 0.15 / 0.30 = 0.50 (50% of the remainder)
    relative_test_size = test_ratio / (test_ratio + val_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_size,
        stratify=targets[temp_indices],  # Filter targets for the remaining indices
        random_state=random_state
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
    

def __is_valid_file(file_path: str) -> bool:
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True
    else:
        return False
#ImageNet functions
def load_imagenet_100(transforms_original=None, transforms_augmented_list=None) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    IMAGENET_100_ROOT_PATH = "/kaggle/input/datasets/ambityga/imagenet100"

    # code adapted from https://www.kaggle.com/code/goduguanilhimam/resnet-34-lmagenet100-21-8m?scriptVersionId=261568289&cellId=7 
    train_datasets = []
    val_dataset = None
    for parent_node in pathlib.Path(IMAGENET_100_ROOT_PATH).iterdir():

        # Avoiding the Label File
        if not parent_node.is_dir():
            continue
    
        # Retrieving the Split Name
        split_name = parent_node.stem.split(".")[0]
    
        # Executing the Copy based on the Split Name
        if split_name == "train" :
            #original dataset
            train_datasets.append(datasets.ImageFolder(root=parent_node, transform=transforms_original, is_valid_file=__is_valid_file))
            # augmented transforms
            if transforms_augmented_list is not None and len(transforms_augmented_list) > 0:
                train_datasets.extend(
                    [
                    datasets.ImageFolder(root=parent_node, transform=v2.Compose([transforms_original,t]), is_valid_file=__is_valid_file)
                    for t in transforms_augmented_list
                    ]
                )
        elif split_name == "val":
            val_dataset = datasets.ImageFolder(root=f'{IMAGENET_100_ROOT_PATH}/{parent_node}', transform=transforms_original, is_valid_file=__is_valid_file)
    
    return torch.utils.data.ConcatDataset(train_datasets), val_dataset
            
    
def visualize_dataset_distribution(train_ds, val_ds=[], test_ds=[], class_names=None):
    """
    Visualizes and compares the class distribution among train, validation, and test sets.
    """
    def __get_labels(dataset):
        # Optimized for ImageFolder; falls back to iteration for custom datasets
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif hasattr(dataset, 'labels'):
            return dataset.labels
        elif hasattr(dataset, 'y'):
            return dataset.y
        else:
            return [label for _, label in dataset]

    # 1. Extract labels for all three splits
    train_labels = __get_labels(train_ds)
    val_labels = __get_labels(val_ds)
    test_labels = __get_labels(test_ds)

    # 2. Create and merge DataFrames for Seaborn
    df_train = pd.DataFrame({'Label': train_labels, 'Split': 'Train'})
    df_val = pd.DataFrame({'Label': val_labels, 'Split': 'Validation'})
    df_test = pd.DataFrame({'Label': test_labels, 'Split': 'test'})
    df_all = pd.concat([df_train, df_val, df_test])

    # 3. Plotting
    plt.figure(figsize=(15, 8))
    # The 'viridis' palette scales smoothly to accommodate the third bar cluster
    ax = sns.countplot(data=df_all, x='Label', hue='Split', palette='viridis')
    
    if class_names:
        ax.set_xticklabels(class_names, rotation=90)
    
    plt.title('Class Distribution: Train vs Validation vs test Split', fontsize=16)
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Split')
    plt.tight_layout()
    plt.show()

    # 4. Calculate and print precise dataset proportions
    total_samples = len(train_labels) + len(val_labels) + len(test_labels)
    
    print(f"Total Training Samples:   {len(train_labels)} ({len(train_labels) / total_samples * 100:.1f}%)")
    print(f"Total Validation Samples: {len(val_labels)} ({len(val_labels) / total_samples * 100:.1f}%)")
    print(f"Total test Samples:       {len(test_labels)} ({len(test_labels) / total_samples * 100:.1f}%)")
    print(f"Total Combined Samples:   {total_samples}")


class StochasticCutMixDataLoader:
    def __init__(self, dataloader, num_classes=101, p=0.5):
        self.dataloader = dataloader
        self.cutmix = v2.CutMix(num_classes=num_classes)
        self.p = p

    def __iter__(self):
        for images, labels in self.dataloader:
            if random.random() < self.p:
                images, labels = self.cutmix(images, labels)
                labels = labels.argmax(dim=1)
                
            yield images, labels

    def __len__(self):
        return len(self.dataloader)