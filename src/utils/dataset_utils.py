# %% [code]
# %% [code]
# %% [code]
# %% [code]
import numpy as np
import torch
import os
import json
import shutil
import glob
import sys
import re
import time
import subprocess
from datetime import datetime
from kaggle_secrets import UserSecretsClient
from kaggle.api.kaggle_api_extended import KaggleApi

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


def extract_kaggle_notebook_version_id() -> str | None:
    """
    Extracts scriptVersionId from notebook metadata if running inside a Kaggle run.
    """
    for nb_path in ["/kaggle/working/__notebook__.ipynb", "__notebook__.ipynb"]:
        if os.path.exists(nb_path):
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    match = re.search(r'scriptVersionId=(\d+)', content)
                    if match:
                        return f"v{match.group(1)}"
            except Exception:
                pass
    return None


def get_auto_version_notes(default_note: str = "Auto checkpoint update") -> str:
    """
    Constructs a concise Kaggle dataset version note (< 50 characters limit).
    E.g.: 'v337155846 checkpoints' or 'model-cvs-tiny-example checkpoints'
    """
    version_id = extract_kaggle_notebook_version_id()
    if version_id:
        note = f"{version_id} checkpoints"
    else:
        kernel_slug = os.environ.get("KAGGLE_SLUG")
        if kernel_slug:
            note = f"{kernel_slug} checkpoints"
        else:
            note = default_note

    return note.strip()[:50]


def push_checkpoints_to_kaggle_dataset(
    dataset_slug: str,
    checkpoint_paths: list[str] | str,
    version_notes: str | None = None,
    exclude_patterns: list[str] | None = None,
    staging_dir: str = "/kaggle/working/checkpoint_staging"
) -> None:
    if version_notes:
        version_notes = version_notes.strip()[:50]
    else:
        version_notes = get_auto_version_notes()


    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]

    # Resolve paths (expanding wildcards if provided)
    resolved_paths = []
    for path in checkpoint_paths:
        matches = glob.glob(path)
        if matches:
            resolved_paths.extend(sorted(matches))
        elif os.path.exists(path):
            resolved_paths.append(path)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Filter out excluded patterns (e.g. 'warmup')
    if exclude_patterns:
        resolved_paths = [
            p for p in resolved_paths
            if not any(pat in os.path.basename(p) for pat in exclude_patterns)
        ]

    if not resolved_paths:
        raise FileNotFoundError("No valid checkpoint files found matching the criteria.")

    user_secrets = UserSecretsClient()
    os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
    os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")

    api = KaggleApi()
    api.authenticate()

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    for path in resolved_paths:
        shutil.copy(path, staging_dir)

    dataset_title = dataset_slug.split("/")[-1].replace("-", " ").title()
    metadata = {
        "title": dataset_title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)

    api.dataset_create_version(
        staging_dir,
        version_notes=version_notes,
        delete_old_versions=False
    )

    shutil.rmtree(staging_dir)


def trigger_kaggle_notebook(
    notebook_slug: str,
    delay_seconds: int = 0,
    enable_gpu: bool = True,
    accelerator: str = "nvidiaTeslaT4",
    staging_dir: str = "/kaggle/working/nb_staging"
) -> None:
    user_secrets = UserSecretsClient()
    os.environ['KAGGLE_USERNAME'] = user_secrets.get_secret("KAGGLE_USERNAME")
    os.environ['KAGGLE_KEY'] = user_secrets.get_secret("KAGGLE_KEY")

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    meta_path = os.path.join(staging_dir, "kernel-metadata.json")

    # Try to pull existing notebook metadata
    try:
        subprocess.run([
            "kaggle", "kernels", "pull",
            notebook_slug,
            "-p", staging_dir,
            "-m"
        ], check=True, capture_output=True, text=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        # Fallback if kernel does not exist yet or slug uses hyphens/underscores
        print(f"Warning: Could not pull kernel metadata for {notebook_slug} ({e}). Creating new metadata...")
        kernel_title = notebook_slug.split("/")[-1].replace("-", " ").replace("_", " ").title()
        script_file = "script.py"
        script_path = os.path.join(staging_dir, script_file)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("# Triggered kernel run\nprint('Triggered')\n")
        meta = {
            "id": notebook_slug,
            "title": kernel_title,
            "code_file": script_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if enable_gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": []
        }

    meta["enable_gpu"] = "true" if enable_gpu else "false"
    if accelerator:
        meta["accelerator"] = accelerator

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    cmd = ["kaggle", "kernels", "push", "-p", staging_dir]
    if accelerator:
        cmd.extend(["--accelerator", accelerator])

    subprocess.run(cmd, check=True)

    shutil.rmtree(staging_dir)





