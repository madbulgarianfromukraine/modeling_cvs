# %% [code]
import numpy as np
import torch

def get_labels(dataset):
    """
    Retrieves the labels/targets from a dataset.
    Optimized for ImageFolder/Caltech/Subset/ConcatDataset/AugmentationWrapper/CustomEnricoDataset;
    falls back to iteration for custom datasets.
    """
    if dataset is None:
        return []
        
    # Handle empty list/tuple
    if isinstance(dataset, (list, tuple)) and len(dataset) == 0:
        return []

    # 1. Handle Subsets (with recursion and indices indexing)
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent_labels = get_labels(dataset.dataset)
        if torch.is_tensor(parent_labels):
            return parent_labels[dataset.indices]
        elif isinstance(parent_labels, np.ndarray):
            return parent_labels[dataset.indices]
        elif isinstance(parent_labels, list):
            return [parent_labels[i] for i in dataset.indices]
        else:
            try:
                return parent_labels[dataset.indices]
            except Exception:
                return [parent_labels[i] for i in dataset.indices]

    # 2. Handle ConcatDatasets (with recursion and concatenation)
    if hasattr(dataset, 'datasets'):
        child_labels = [get_labels(ds) for ds in dataset.datasets]
        if all(torch.is_tensor(cl) for cl in child_labels):
            return torch.cat(child_labels)
        elif all(isinstance(cl, np.ndarray) for cl in child_labels):
            return np.concatenate(child_labels)
        else:
            flat_list = []
            for cl in child_labels:
                if torch.is_tensor(cl):
                    flat_list.extend(cl.cpu().numpy().tolist())
                elif isinstance(cl, np.ndarray):
                    flat_list.extend(cl.tolist())
                elif isinstance(cl, (list, tuple)):
                    flat_list.extend(cl)
                else:
                    flat_list.append(cl)
            return flat_list

    # 3. Handle AugmentationWrapper or wrappers with base_dataset attribute
    if hasattr(dataset, 'base_dataset'):
        return get_labels(dataset.base_dataset)

    # 4. Handle CustomEnricoDataset specifically
    if hasattr(dataset, 'samples') and hasattr(dataset, 'labels_dict') and hasattr(dataset, 'class_to_idx'):
        try:
            return [dataset.class_to_idx[dataset.labels_dict[sample[0]]] for sample in dataset.samples]
        except Exception:
            pass

    # 5. Handle generic wrappers with dataset attribute (not Subset)
    if hasattr(dataset, 'dataset'):
        return get_labels(dataset.dataset)

    # 6. Extract targets/labels/y attributes if present
    if hasattr(dataset, 'targets'):
        return dataset.targets
    elif hasattr(dataset, 'labels'):
        return dataset.labels
    elif hasattr(dataset, 'y'):
        return dataset.y

    # 7. Fallback
    try:
        length = len(dataset)
        return [dataset[i][1] for i in range(length)]
    except Exception:
        return [label for _, label in dataset]

