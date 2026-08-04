# %% [code]
import os
import time
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
import kornia.augmentation as K

from training import train_model_stages, plot_learning_curves, mean_and_std_for_normalization
from evaluation import evaluate_model
from fine_tuning import initial_freeze_unfreeze, second_unfreeze, reset_head_last_fc
from model import PseudoAlexNet, DEVICE
from screen_images import CustomEnricoDataset, get_allowed_classes, make_screen_base_transform
from natural_images import make_caltech_base_transform, stratified_three_way_split, AugmentationWrapper
from dataset_utils import push_checkpoints_to_kaggle_dataset, trigger_kaggle_notebook, get_labels

try:
    from sam import SAM
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


# ==============================================================================
# 1. GRID SEARCH PARAMETER DEFINITIONS (3x3 Grid = 9 Possible Configurations)
# ==============================================================================

GRID_LR_NAT = [1e-3, 5e-3, 1e-2]       # Pretraining on Natural images
GRID_LR_WARMUP = [1e-6, 1e-5, 1e-4]    # Stage 1: Warmup fine-tuning
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

    for name, param in model.named_parameters():
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
        return SAM(model.parameters(), base_optimizer, rho=rho)
    return base_optimizer


def resolve_dataset_root(candidates: List[str], default_path: str) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return default_path


# ==============================================================================
# 2. MODULAR DATA PREPARATION PIECES
# ==============================================================================

def prepare_enrico_dataloaders(
    enrico_root: str,
    use_wireframes: bool = False,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Prepares normalized DataLoaders and data augmentation pipelines for the Enrico dataset.
    """
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
        screen_train_raw, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
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
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3
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

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "gpu_augmentations": gpu_augmentations,
        "cutmix": cutmix,
        "num_classes": enrico_classes
    }


# ==============================================================================
# 3. MODULAR TRAINING & FINE-TUNING PIECES
# ==============================================================================

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
    save_intermediate_checkpoints: bool = False,
    use_differential_lr: bool = False
) -> Dict[str, Any]:
    """
    Executes fine-tuning across Stage 1 (Warmup on head) and Stage 2 (Drift on full model).
    """
    num_classes = enrico_loaders["num_classes"]
    train_loader = enrico_loaders["train_loader"]
    val_loader = enrico_loaders["val_loader"]
    test_loader = enrico_loaders["test_loader"]
    gpu_aug = enrico_loaders["gpu_augmentations"]
    cutmix = enrico_loaders["cutmix"]

    # --- Stage 1: Warmup (Classifier head only) ---
    reset_head_last_fc(model, output_features=num_classes)
    model = model.to(device)
    initial_freeze_unfreeze(model=model)

    if use_differential_lr:
        opt_warmup = create_differential_optimizer(
            model, lr_early=cfg["lr_early"], lr_late=cfg["lr_late"], lr_head=cfg["lr_head"]
        )
    else:
        base_opt_warmup = optim.AdamW(model.parameters(), lr=cfg["lr_warmup"], eps=1e-5, weight_decay=1e-3)
        opt_warmup = SAM(model.parameters(), base_opt_warmup, rho=0.05) if HAS_SAM else base_opt_warmup

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
        patience=max_warmup_epochs,
        checkpoint_dir=checkpoint_dir,
        augmenter=gpu_aug,
        cutmix=cutmix,
        cutmix_prob=0.5,
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        checkpoint_file_name=ckpt_warmup_name
    )

    # --- Stage 2: Drift (Unfreeze full model) ---
    second_unfreeze(model=model)

    if use_differential_lr:
        opt_drift = create_differential_optimizer(
            model, lr_early=cfg["lr_early"], lr_late=cfg["lr_late"], lr_head=cfg["lr_head"]
        )
    else:
        base_opt_drift = optim.AdamW(model.parameters(), lr=cfg["lr_drift"], eps=1e-5, weight_decay=1e-3)
        opt_drift = SAM(model.parameters(), base_opt_drift, rho=0.05) if HAS_SAM else base_opt_drift

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
        patience=max_drift_epochs,
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
    model_scratch = model_scratch.to(device)

    base_opt_scratch = optim.AdamW(model_scratch.parameters(), lr=cfg["lr_drift"], eps=1e-5, weight_decay=1e-3)
    opt_scratch = SAM(model_scratch.parameters(), base_opt_scratch, rho=0.05) if HAS_SAM else base_opt_scratch

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
        patience=max_screen_epochs,
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
# 4. MAIN EXPERIMENT ORCHESTRATOR & GRID SEARCH LOOP
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
    checkpoint_dir: str = "/kaggle/working/",
    enrico_root: Optional[str] = None,
    caltech_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_differential_lr: bool = False,
    run_screen_from_scratch: bool = True
) -> Dict[str, Any]:
    """
    Main experiment orchestrator breaking down data preparation, fine-tuning, and scratch training.
    """
    if device is None:
        device = DEVICE

    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg = get_grid_config(grid_i, grid_j)
    config_tag = f"grid_{grid_i}_{grid_j}"

    print(f"\n=======================================================")
    print(f"🚀 RUNNING EXPERIMENT CONFIG: [{grid_i}, {grid_j}] ({config_tag})")
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

    # Step 1: Data preparation piece
    enrico_loaders = prepare_enrico_dataloaders(
        enrico_root=enrico_root,
        use_wireframes=use_wireframes,
        batch_size=batch_size
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = PseudoAlexNet(tiny_factor=tiny_factor, p=p_dropout).to(device)

    # Step 2: Fine-tuning piece
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
        save_intermediate_checkpoints=save_intermediate_checkpoints,
        use_differential_lr=use_differential_lr
    )

    # Step 3: Screen scratch piece (Optional)
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
            save_intermediate_checkpoints=save_intermediate_checkpoints
        )

    return {
        "config": cfg,
        "use_wireframes": use_wireframes,
        "fine_tuning": ft_results,
        "screen_scratch": scratch_results
    }


def run_grid_search(
    grid_indices: Optional[List[Tuple[int, int]]] = None,
    use_wireframes: bool = False,
    save_intermediate_checkpoints: bool = False,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Executes a loop over selected (or all 9) grid index pairs (grid_i, grid_j).
    """
    if grid_indices is None:
        grid_indices = [(i, j) for i in range(3) for j in range(3)]

    all_results = {}
    print(f"\n=======================================================")
    print(f"🌐 STARTING GRID SEARCH ACROSS {len(grid_indices)} CONFIGURATIONS")
    print(f"=======================================================\n")

    for i, j in grid_indices:
        tag = f"grid_{i}_{j}"
        res = run_fine_tuning_experiment(
            grid_i=i,
            grid_j=j,
            use_wireframes=use_wireframes,
            save_intermediate_checkpoints=save_intermediate_checkpoints,
            **kwargs
        )
        all_results[tag] = res

    return all_results
