# %% [code]
# %% [code]
# %% [code]
# %% [code]
import os
import time
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import v2
import kornia.augmentation as K

from training import train_model_stages, plot_learning_curves, mean_and_std_for_normalization
from evaluation import evaluate_model
from fine_tuning import initial_freeze_unfreeze, second_unfreeze, reset_head_last_fc
from model import PseudoAlexNet, DEVICE
from screen_images import CustomEnricoDataset, get_allowed_classes, make_screen_base_transform
from natural_images import (
    make_caltech_base_transform,
    stratified_three_way_split,
    AugmentationWrapper,
    build_caltech_split_datasets
)
from dataset_utils import push_checkpoints_to_kaggle_dataset, trigger_kaggle_notebook, get_labels

try:
    from sam import SAM
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


# ==============================================================================
# 1. GRID SEARCH PARAMETER DEFINITIONS (3x3 Grid = 9 Possible Configurations)
# ==============================================================================

GRID_LR_NAT = [1e-3, 5e-3, 1e-2]       # Pretraining on Natural images (Caltech101)
GRID_LR_WARMUP = [1e-6, 1e-5, 1e-4]    # Stage 1: Warmup fine-tuning (classifier head)
GRID_LR_DRIFT = [1e-5, 5e-5, 1e-4]     # Stage 2: Full unfreeze / Drift fine-tuning

# Differential layer-wise learning rates (Early features, Late features, Head)
GRID_LR_EARLY = [1e-6, 1e-5, 5e-5]     # Features [0..1]
GRID_LR_LATE  = [1e-5, 5e-5, 1e-4]     # Features [3..4]
GRID_LR_HEAD  = [1e-4, 5e-4, 1e-3]     # Classifier Head


def get_grid_config(grid_i: int = 0, grid_j: int = 0) -> Dict[str, Any]:
    """
    Retrieves parameter dictionary for grid coordinate [grid_i, grid_j].
    Supports 3x3 grid (9 total configs): grid_i in [0, 1, 2], grid_j in [0, 1, 2].
    """
    i = max(0, min(2, int(grid_i)))
    j = max(0, min(2, int(grid_j)))
    
    return {
        "grid_i": i,
        "grid_j": j,
        "config_name": f"grid_{i}_{j}",
        "lr_nat": GRID_LR_NAT[i],
        "lr_warmup": GRID_LR_WARMUP[i],
        "lr_drift": GRID_LR_DRIFT[j],
        "lr_early": GRID_LR_EARLY[i],
        "lr_late": GRID_LR_LATE[j],
        "lr_head": GRID_LR_HEAD[(i + j) % 3],
    }


def get_all_grid_configs() -> List[Dict[str, Any]]:
    """
    Returns all 9 parameter configurations in the 3x3 grid matrix.
    """
    return [get_grid_config(i, j) for i in range(3) for j in range(3)]


def create_differential_optimizer(
    model: nn.Module,
    lr_early: float = 1e-5,
    lr_late: float = 5e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-3,
    eps: float = 1e-5,
    rho: float = 0.05,
    use_sam: bool = True
) -> optim.Optimizer:
    """
    Constructs an AdamW optimizer (optionally wrapped with SAM) applying 
    differential learning rates across early feature layers, late feature layers, 
    and the classifier head.
    """
    early_params, late_params, head_params = [], [], []

    # Unwrap DataParallel if present
    target_model = model.module if isinstance(model, nn.DataParallel) else model

    for name, param in target_model.named_parameters():
        if not param.requires_grad:
            continue
        if "features.0" in name or "features.1" in name:
            early_params.append(param)
        elif "features.3" in name or "features.4" in name:
            late_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": early_params, "lr": lr_early, "weight_decay": weight_decay},
        {"params": late_params, "lr": lr_late, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]

    base_optimizer = optim.AdamW(param_groups, eps=eps)
    if use_sam and HAS_SAM:
        return SAM(target_model.parameters(), base_optimizer, rho=rho)
    return base_optimizer


def resolve_dataset_root(candidates: List[str], default_path: str) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return default_path


def get_available_cuda_devices() -> List[torch.device]:
    """
    Returns a list of available CUDA devices (e.g. [cuda:0, cuda:1]) or [cpu].
    """
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(count)]
    return [torch.device("cpu")]


def setup_model_device(model: nn.Module, device: torch.device, use_data_parallel: bool = True) -> nn.Module:
    """
    Moves model to the designated device and optionally wraps with DataParallel if multi-GPU is available.
    """
    model = model.to(device)
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if device.type == "cuda" and (device.index is None or device.index == 0):
            model = nn.DataParallel(model)
    return model


# ==============================================================================
# 2. MODULAR DATA PREPARATION PIECES (CALTECH101 & ENRICO)
# ==============================================================================

import threading

CALTECH_LOCK = threading.Lock()
ENRICO_LOCK = threading.Lock()

_CALTECH_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
_ENRICO_CACHE: Dict[Tuple[str, bool, int], Dict[str, Any]] = {}


def prepare_caltech_dataloaders(
    caltech_root: str,
    batch_size: int = 128,
    num_workers: int = 0
) -> Dict[str, Any]:
    """
    Prepares DataLoaders and augmentation pipelines for Caltech101 (natural images pre-training).
    Includes thread locking and error handling to prevent multi-GPU download race conditions or corrupted archives.
    """
    cache_key = (caltech_root, batch_size, num_workers)
    with CALTECH_LOCK:
        if cache_key in _CALTECH_CACHE:
            return _CALTECH_CACHE[cache_key]

        os.makedirs(caltech_root, exist_ok=True)
        try:
            caltech_base_dataset = datasets.Caltech101(root=caltech_root, download=True)
        except Exception as e:
            print(f"⚠️ Caltech101 download error: {e}. Recovering by purging partial archives...")
            import shutil
            for name in ["101_ObjectCategories.tar.gz", "101_ObjectCategories"]:
                target_path = os.path.join(caltech_root, "caltech101", name)
                if os.path.isfile(target_path):
                    try:
                        os.remove(target_path)
                    except Exception:
                        pass
                elif os.path.isdir(target_path):
                    shutil.rmtree(target_path, ignore_errors=True)
            caltech_base_dataset = datasets.Caltech101(root=caltech_root, download=True)
        
        caltech_train_raw, caltech_val_raw, caltech_test_raw = stratified_three_way_split(
            dataset=caltech_base_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        caltech_preprocess = make_caltech_base_transform(resize=(300, 200))
        caltech_train_raw_loader = torch.utils.data.DataLoader(
            AugmentationWrapper(caltech_train_raw, caltech_preprocess),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        caltech_train_mean, caltech_train_std = mean_and_std_for_normalization(caltech_train_raw_loader)

        nat_images_train, nat_images_val, nat_images_test = build_caltech_split_datasets(
            train_subset=caltech_train_raw,
            val_subset=caltech_val_raw,
            test_subset=caltech_test_raw,
            mean=caltech_train_mean,
            std=caltech_train_std,
            transforms_augmented_list=None,
            resize=(300, 200),
        )

        train_loader = torch.utils.data.DataLoader(
            nat_images_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            nat_images_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            nat_images_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        gpu_augmentations = v2.Compose([
            v2.RandomResizedCrop(size=(300, 200), scale=(0.4, 1.0)),
            v2.ColorJitter(),
            v2.RandomErasing(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomElasticTransform(p=0.3, kernel_size=(63, 63), sigma=(32.0, 32.0), keepdim=True),
            K.RandomBoxBlur(p=0.2, keepdim=True),
        ])
        cutmix = v2.CutMix(num_classes=101)

        result = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "gpu_augmentations": gpu_augmentations,
            "cutmix": cutmix,
            "num_classes": 101
        }
        _CALTECH_CACHE[cache_key] = result
        return result


def prepare_enrico_dataloaders(
    enrico_root: str,
    use_wireframes: bool = False,
    batch_size: int = 128,
    num_workers: int = 0
) -> Dict[str, Any]:
    """
    Prepares normalized DataLoaders and data augmentation pipelines for the Enrico dataset.
    Includes thread locking and caching to prevent multi-GPU race conditions.
    """
    cache_key = (enrico_root, use_wireframes, batch_size, num_workers)
    with ENRICO_LOCK:
        if cache_key in _ENRICO_CACHE:
            return _ENRICO_CACHE[cache_key]

        screen_preprocess = make_screen_base_transform(resize=(300, 200))
        screen_train_raw, screen_val_raw, screen_test_raw = CustomEnricoDataset.create_splits(
            root=enrico_root,
            val_size=0.1,
            test_size=0.1,
            use_wireframes=use_wireframes,
            train_transform=screen_preprocess,
            eval_transform=screen_preprocess
        )

        enrico_classes = len(get_allowed_classes())
        screen_raw_loader = torch.utils.data.DataLoader(
            screen_train_raw, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers
        )
        screen_train_mean, screen_train_std = mean_and_std_for_normalization(screen_raw_loader)

        universal_base_screen = v2.Compose([
            screen_preprocess,
            v2.Normalize(mean=screen_train_mean, std=screen_train_std, inplace=False)
        ])

        train_ds, val_ds, test_ds = CustomEnricoDataset.create_splits(
            root=enrico_root,
            val_size=0.1,
            test_size=0.1,
            use_wireframes=use_wireframes,
            train_transform=universal_base_screen,
            eval_transform=universal_base_screen
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers
        )

        gpu_augmentations = v2.Compose([
            v2.RandomResizedCrop(size=(300, 200), scale=(0.4, 1.0)),
            v2.ColorJitter(),
            v2.RandomErasing(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomElasticTransform(p=0.3, kernel_size=(63, 63), sigma=(32.0, 32.0), keepdim=True),
            K.RandomBoxBlur(p=0.2, keepdim=True),
        ])
        cutmix = v2.CutMix(num_classes=enrico_classes)

        result = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "gpu_augmentations": gpu_augmentations,
            "cutmix": cutmix,
            "num_classes": enrico_classes
        }
        _ENRICO_CACHE[cache_key] = result
        return result


# ==============================================================================
# 3. MODULAR TRAINING STAGES (PRE-TRAINING, FINE-TUNING, SCREEN SCRATCH)
# ==============================================================================

def train_natural_pretraining_stage(
    model: nn.Module,
    cfg: Dict[str, Any],
    config_tag: str,
    caltech_loaders: Dict[str, Any],
    loss_fn: nn.Module,
    device: torch.device,
    max_nat_epochs: int,
    checkpoint_dir: str,
    patience: int = 5,
    save_intermediate_checkpoints: bool = False
) -> Dict[str, Any]:
    """
    Executes pre-training stage on Caltech101 natural images dataset.
    """
    train_loader = caltech_loaders["train_loader"]
    val_loader = caltech_loaders["val_loader"]
    test_loader = caltech_loaders["test_loader"]
    gpu_aug = caltech_loaders["gpu_augmentations"]
    cutmix = caltech_loaders["cutmix"]

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    base_optimizer = optim.AdamW(target_model.parameters(), lr=cfg["lr_nat"], eps=1e-5, weight_decay=1e-3)
    optimizer = SAM(target_model.parameters(), base_optimizer, rho=0.05) if HAS_SAM else base_optimizer

    ckpt_nat_name = f"finished_nat_images_{config_tag}"
    train_nat_loss, val_nat_loss = train_model_stages(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_epochs=max_nat_epochs,
        stage_name=f"nat_images_{config_tag}",
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        augmenter=gpu_aug,
        cutmix=cutmix,
        cutmix_prob=0.5,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        checkpoint_file_name=ckpt_nat_name
    )

    nat_test_loss, nat_test_acc = evaluate_model(model, test_loader, loss_fn, device)

    plot_learning_curves(
        train_losses=train_nat_loss,
        val_losses=val_nat_loss,
        log_scale=True,
        save_to_csv=True,
        stage=f"nat_images_{config_tag}"
    )

    return {
        "train_loss": train_nat_loss,
        "val_loss": val_nat_loss,
        "test_loss": nat_test_loss,
        "test_acc": nat_test_acc,
        "checkpoint": os.path.join(checkpoint_dir, f"{ckpt_nat_name}.pth")
    }


def train_fine_tuning_stages(
    model: nn.Module,
    cfg: Dict[str, Any],
    config_tag: str,
    enrico_loaders: Dict[str, Any],
    loss_fn: nn.Module,
    device: torch.device,
    max_warmup_epochs: int,
    max_drift_epochs: int,
    checkpoint_dir: str,
    patience: int = 5,
    save_intermediate_checkpoints: bool = False,
    use_differential_lr: bool = False
) -> Dict[str, Any]:
    """
    Executes fine-tuning across Stage 1 (Warmup on classifier head) and Stage 2 (Drift on full model).
    """
    num_classes = enrico_loaders["num_classes"]
    train_loader = enrico_loaders["train_loader"]
    val_loader = enrico_loaders["val_loader"]
    test_loader = enrico_loaders["test_loader"]
    gpu_aug = enrico_loaders["gpu_augmentations"]
    cutmix = enrico_loaders["cutmix"]

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    # --- Stage 1: Warmup (Classifier head only) ---
    reset_head_last_fc(target_model, output_features=num_classes)
    model = setup_model_device(model, device)
    initial_freeze_unfreeze(model=target_model)

    if use_differential_lr:
        opt_warmup = create_differential_optimizer(
            target_model, lr_early=cfg["lr_early"], lr_late=cfg["lr_late"], lr_head=cfg["lr_head"]
        )
    else:
        base_opt_warmup = optim.AdamW(target_model.parameters(), lr=cfg["lr_warmup"], eps=1e-5, weight_decay=1e-3)
        opt_warmup = SAM(target_model.parameters(), base_opt_warmup, rho=0.05) if HAS_SAM else base_opt_warmup

    ckpt_warmup_name = f"finished_warmup_2_{config_tag}"
    train_warmup_loss, val_warmup_loss = train_model_stages(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=opt_warmup,
        loss_fn=loss_fn,
        device=device,
        max_epochs=max_warmup_epochs,
        stage_name=f"warmup_2_{config_tag}",
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        augmenter=gpu_aug,
        cutmix=cutmix,
        cutmix_prob=0.5,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        checkpoint_file_name=ckpt_warmup_name
    )

    # --- Stage 2: Drift (Unfreeze full model) ---
    second_unfreeze(model=target_model)

    if use_differential_lr:
        opt_drift = create_differential_optimizer(
            target_model, lr_early=cfg["lr_early"], lr_late=cfg["lr_late"], lr_head=cfg["lr_head"]
        )
    else:
        base_opt_drift = optim.AdamW(target_model.parameters(), lr=cfg["lr_drift"], eps=1e-5, weight_decay=1e-3)
        opt_drift = SAM(target_model.parameters(), base_opt_drift, rho=0.05) if HAS_SAM else base_opt_drift

    ckpt_drift_name = f"finished_drift_2_{config_tag}"
    train_drift_loss, val_drift_loss = train_model_stages(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=opt_drift,
        loss_fn=loss_fn,
        device=device,
        max_epochs=max_drift_epochs,
        stage_name=f"drift_2_{config_tag}",
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        augmenter=gpu_aug,
        cutmix=cutmix,
        cutmix_prob=0.5,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        checkpoint_file_name=ckpt_drift_name
    )

    ft_test_loss, ft_test_acc = evaluate_model(model, test_loader, loss_fn, device)

    train_ft_loss = train_warmup_loss + train_drift_loss
    val_ft_loss = val_warmup_loss + val_drift_loss

    plot_learning_curves(
        train_losses=train_ft_loss,
        val_losses=val_ft_loss,
        log_scale=True,
        save_to_csv=True,
        stage=f"drift_2_{config_tag}"
    )

    return {
        "train_loss": train_ft_loss,
        "val_loss": val_ft_loss,
        "test_loss": ft_test_loss,
        "test_acc": ft_test_acc,
        "checkpoint": os.path.join(checkpoint_dir, f"{ckpt_drift_name}.pth")
    }


def train_screen_scratch_stage(
    cfg: Dict[str, Any],
    config_tag: str,
    enrico_loaders: Dict[str, Any],
    loss_fn: nn.Module,
    device: torch.device,
    tiny_factor: int,
    p_dropout: float,
    max_screen_epochs: int,
    checkpoint_dir: str,
    patience: int = 5,
    save_intermediate_checkpoints: bool = False
) -> Dict[str, Any]:
    """
    Trains PseudoAlexNet from scratch on the Enrico screen images dataset.
    """
    warnings.filterwarnings("ignore")
    num_classes = enrico_loaders["num_classes"]
    train_loader = enrico_loaders["train_loader"]
    val_loader = enrico_loaders["val_loader"]
    test_loader = enrico_loaders["test_loader"]
    gpu_aug = enrico_loaders["gpu_augmentations"]
    cutmix = enrico_loaders["cutmix"]

    model_scratch = PseudoAlexNet(tiny_factor=tiny_factor, p=p_dropout)
    reset_head_last_fc(model_scratch, output_features=num_classes)
    model_scratch = setup_model_device(model_scratch, device)
    target_scratch = model_scratch.module if isinstance(model_scratch, nn.DataParallel) else model_scratch

    base_opt_scratch = optim.AdamW(target_scratch.parameters(), lr=cfg["lr_drift"], eps=1e-5, weight_decay=1e-3)
    opt_scratch = SAM(target_scratch.parameters(), base_opt_scratch, rho=0.05) if HAS_SAM else base_opt_scratch

    ckpt_scratch_name = f"finished_screen_{config_tag}"
    train_scratch_loss, val_scratch_loss = train_model_stages(
        model=model_scratch,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=opt_scratch,
        loss_fn=loss_fn,
        device=device,
        max_epochs=max_screen_epochs,
        stage_name=f"screen_{config_tag}",
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        augmenter=gpu_aug,
        cutmix=cutmix,
        cutmix_prob=0.5,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        checkpoint_file_name=ckpt_scratch_name
    )

    scratch_test_loss, scratch_test_acc = evaluate_model(model_scratch, test_loader, loss_fn, device)

    plot_learning_curves(
        train_losses=train_scratch_loss,
        val_losses=val_scratch_loss,
        log_scale=True,
        save_to_csv=True,
        stage=f"screen_images_{config_tag}"
    )

    return {
        "train_loss": train_scratch_loss,
        "val_loss": val_scratch_loss,
        "test_loss": scratch_test_loss,
        "test_acc": scratch_test_acc,
        "checkpoint": os.path.join(checkpoint_dir, f"{ckpt_scratch_name}.pth")
    }


# ==============================================================================
# 4. MAIN EXPERIMENT ORCHESTRATOR & PARALLEL MULTI-GPU GRID SEARCH LOOP
# ==============================================================================

def run_fine_tuning_experiment(
    grid_i: int = 0,
    grid_j: int = 0,
    use_wireframes: bool = False,
    save_intermediate_checkpoints: bool = False,
    tiny_factor: int = 3,
    p_dropout: float = 0.25,
    batch_size: int = 128,
    max_nat_epochs: int = 50,
    max_warmup_epochs: int = 15,
    max_drift_epochs: int = 40,
    max_screen_epochs: int = 50,
    patience: int = 5,
    checkpoint_dir: str = "/kaggle/working/",
    enrico_root: Optional[str] = None,
    caltech_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_differential_lr: bool = False,
    run_screen_from_scratch: bool = True
) -> Dict[str, Any]:
    """
    Main experiment pipeline executing:
      1. Caltech101 Natural Images Pre-training
      2. Enrico Screen Images Fine-Tuning (Warmup + Drift)
      3. Enrico Screen Images Scratch Benchmark
    """
    if device is None:
        device = DEVICE

    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg = get_grid_config(grid_i, grid_j)
    config_tag = f"grid_{grid_i}_{grid_j}"

    print(f"\n=======================================================")
    print(f"🚀 RUNNING EXPERIMENT CONFIG: [{grid_i}, {grid_j}] ({config_tag})")
    print(f"   Target Device = {device}")
    print(f"   patience = {patience}")
    print(f"   use_wireframes = {use_wireframes}")
    print(f"   save_intermediate_checkpoints = {save_intermediate_checkpoints}")
    print(f"   lr_nat = {cfg['lr_nat']}, lr_warmup = {cfg['lr_warmup']}, lr_drift = {cfg['lr_drift']}")
    print(f"=======================================================\n")

    if enrico_root is None:
        enrico_root = resolve_dataset_root([
            "/kaggle/input/enricoscreenshotsandwireframes",
            "/kaggle/input/datasets/nazariyyuchnovskiy/enricoscreenshotsandwireframes",
            "./data/enricoscreenshotsandwireframes"
        ], "./data/enricoscreenshotsandwireframes")

    if caltech_root is None:
        caltech_root = resolve_dataset_root([
            "/kaggle/input/caltech101",
            "/kaggle/input/datasets/nazariyyuchnovskiy/caltech101",
            "./data"
        ], "./data")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Step 1: Caltech101 Natural Images Pre-training ---
    caltech_loaders = prepare_caltech_dataloaders(
        caltech_root=caltech_root,
        batch_size=batch_size
    )
    model = PseudoAlexNet(tiny_factor=tiny_factor, p=p_dropout)
    model = setup_model_device(model, device)

    nat_results = train_natural_pretraining_stage(
        model=model,
        cfg=cfg,
        config_tag=config_tag,
        caltech_loaders=caltech_loaders,
        loss_fn=loss_fn,
        device=device,
        max_nat_epochs=max_nat_epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        save_intermediate_checkpoints=save_intermediate_checkpoints
    )

    # --- Step 2: Enrico Screen Images Fine-Tuning (Warmup + Drift) ---
    enrico_loaders = prepare_enrico_dataloaders(
        enrico_root=enrico_root,
        use_wireframes=use_wireframes,
        batch_size=batch_size
    )

    ft_results = train_fine_tuning_stages(
        model=model,
        cfg=cfg,
        config_tag=config_tag,
        enrico_loaders=enrico_loaders,
        loss_fn=loss_fn,
        device=device,
        max_warmup_epochs=max_warmup_epochs,
        max_drift_epochs=max_drift_epochs,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        use_differential_lr=use_differential_lr
    )

    # --- Step 3: Enrico Screen Images Training from Scratch ---
    scratch_results = {}
    if run_screen_from_scratch:
        scratch_results = train_screen_scratch_stage(
            cfg=cfg,
            config_tag=config_tag,
            enrico_loaders=enrico_loaders,
            loss_fn=loss_fn,
            device=device,
            tiny_factor=tiny_factor,
            p_dropout=p_dropout,
            max_screen_epochs=max_screen_epochs,
            checkpoint_dir=checkpoint_dir,
            patience=patience,
            save_intermediate_checkpoints=save_intermediate_checkpoints
        )

    return {
        "config": cfg,
        "use_wireframes": use_wireframes,
        "natural_pretraining": nat_results,
        "fine_tuning": ft_results,
        "screen_scratch": scratch_results
    }


def _worker_run_config(args: Tuple[int, int, str, bool, bool, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    i, j, device_str, use_wireframes, save_intermediate_checkpoints, kwargs = args
    device = torch.device(device_str)
    tag = f"grid_{i}_{j}"
    
    if device.type == "cuda":
        torch.cuda.set_device(device)
        with torch.cuda.device(device):
            res = run_fine_tuning_experiment(
                grid_i=i,
                grid_j=j,
                use_wireframes=use_wireframes,
                save_intermediate_checkpoints=save_intermediate_checkpoints,
                device=device,
                **kwargs
            )
    else:
        res = run_fine_tuning_experiment(
            grid_i=i,
            grid_j=j,
            use_wireframes=use_wireframes,
            save_intermediate_checkpoints=save_intermediate_checkpoints,
            device=device,
            **kwargs
        )
    return tag, res


def run_grid_search(
    grid_indices: Optional[List[Tuple[int, int]]] = None,
    use_wireframes: bool = False,
    save_intermediate_checkpoints: bool = False,
    use_multi_gpu: bool = True,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Executes grid search across 9 configurations. Automatically parallelizes 
    experiments across available GPUs (e.g. cuda:0 and cuda:1) when use_multi_gpu=True.
    """
    if grid_indices is None:
        grid_indices = [(i, j) for i in range(3) for j in range(3)]

    cuda_devices = get_available_cuda_devices()
    num_gpus = len(cuda_devices)
    all_results = {}

    print(f"\n=======================================================")
    print(f"🌐 STARTING GRID SEARCH ACROSS {len(grid_indices)} CONFIGURATIONS")
    print(f"   Available Devices = {[str(d) for d in cuda_devices]} (Count: {num_gpus})")
    print(f"=======================================================\n")

    if use_multi_gpu and num_gpus >= 2:
        print(f"⚡ Parallelizing grid search across {num_gpus} GPUs simultaneously!\n")
        tasks = []
        for idx, (i, j) in enumerate(grid_indices):
            assigned_device = cuda_devices[idx % num_gpus]
            tasks.append((i, j, str(assigned_device), use_wireframes, save_intermediate_checkpoints, kwargs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_config = {
                executor.submit(_worker_run_config, task): f"grid_{task[0]}_{task[1]}"
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_config):
                tag = future_to_config[future]
                try:
                    res_tag, res = future.result()
                    all_results[res_tag] = res
                    print(f"✅ Completed parallel task: {res_tag}")
                except Exception as exc:
                    import traceback
                    print(f"❌ Config {tag} generated an exception:\n{traceback.format_exc()}")
    else:
        for idx, (i, j) in enumerate(grid_indices):
            tag = f"grid_{i}_{j}"
            res = run_fine_tuning_experiment(
                grid_i=i,
                grid_j=j,
                use_wireframes=use_wireframes,
                save_intermediate_checkpoints=save_intermediate_checkpoints,
                device=cuda_devices[0],
                **kwargs
            )
            all_results[tag] = res

    return all_results
