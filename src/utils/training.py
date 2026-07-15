# %% [code]
# %% [code]
# %% [code]
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
import time
import torch
import numpy as np
import random
import pandas as pd
import copy

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple, Optional

os.makedirs("/kaggle/working/", exist_ok=True)

def save_checkpoint(model, optimizer, epoch, loss, path="/kaggle/working/", file_name=None):
    cpu_state_dict = {k: v.to('cpu') for k, v in model.state_dict().items()}
    
    cpu_opt_dict = {}
    for k, v in optimizer.state_dict().items():
        if k == 'state':
            cpu_opt_dict[k] = {idx: {p_k: p_v.to('cpu') if torch.is_tensor(p_v) else p_v 
                                     for p_k, p_v in p_v.items()} 
                               for idx, p_v in v.items()}
        else:
            cpu_opt_dict[k] = v

    state = {
        'epoch': epoch,
        'state_dict': cpu_state_dict,
        'optimizer': cpu_opt_dict,
        'loss': loss,
    }
    
    full_path = f"{path}{file_name if file_name else f'after_{epoch}_checkpoint'}.pth"
    torch.save(state, full_path)
    print(f"✅ Neutral Checkpoint saved to {full_path}")


def load_checkpoint(model, optimizer, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        load_status = model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"🔄 Checkpoint loaded. Status: {load_status}")
        print(f"Resuming from Epoch {epoch} | Loss: {loss:.4f}")
        return epoch, loss
    else:
        print(f"⚠️ Path not found: {path}")
        return 0, float('inf')


def train_and_eval_epoch(epoch, model, train_loader, val_loader, 
                         optimizer, loss_fn, device, log_interval, 
                         checkpoint_dir, stage_name: str = "nat_images", bn_eval: bool = False, 
                         augmenter: Optional[nn.Module] = None) -> Tuple[float, Optional[float]]:
    epoch_start_time = time.time()

    model.train()
    if bn_eval:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
    local_train_loss_sum = 0.0
    local_train_steps = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if augmenter:
            data = augmenter(data)
            
        def __closure():
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
            
        loss = optimizer.step(__closure)
    
        local_train_loss_sum += loss.item()
        local_train_steps += 1

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader)*128} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    if local_train_steps > 0:
        avg_train_loss = local_train_loss_sum / local_train_steps
    else:
        avg_train_loss = float('inf')

    print(f'--- Epoch {epoch} Training Metrics ---')
    print(f'Average Train Loss: {avg_train_loss:.6f}')

    model.eval()
    local_val_loss_sum = 0.0
    local_val_steps = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            
            local_val_loss_sum += loss.item()
            local_val_steps += 1

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    
    if local_val_steps > 0:
        avg_val_loss = local_val_loss_sum / local_val_steps
    else:
        avg_val_loss = float('inf')
    
    val_accuracy = 100. * correct / total if total > 0 else 0.0
    
    print(f'\n--- Epoch {epoch} Evaluation ---')
    print(f'Validation set: Average Loss: {avg_val_loss:.6f}, Accuracy: {val_accuracy:.2f}%\n')

    if epoch % 10 == 0:
        file_name = f"after_{epoch}_{stage_name}"
        save_checkpoint(model, optimizer, epoch, avg_train_loss, path=checkpoint_dir, file_name=file_name)

    epoch_finish_time = time.time()
    print(f"Total time of epoch {epoch} is {epoch_finish_time - epoch_start_time:.2f}s\n")
    
    return avg_train_loss, avg_val_loss


def train_model_stages(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    max_epochs,
    patience=5,
    checkpoint_dir="/kaggle/working/",
    stage_name="nat_images",
    log_interval=3,
    vis_queue=None,
    vis_finish_event=None,
    bn_eval: bool = False,
    augmenter: Optional[nn.Module] = None
):
    print(f"=== Training on the {stage_name} ===")

    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    best_model_state = None

    print(f"Starting training of the {stage_name} in {max_epochs} epochs.")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        print(f" {stage_name} training epoch {epoch}/{max_epochs}...")

        train_loss, val_loss = train_and_eval_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir,
            stage_name=stage_name,
            bn_eval=bn_eval,
            augmenter=augmenter
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if vis_queue is not None:
            cpu_state = {k: v.to('cpu') for k, v in model.state_dict().items()}
            vis_queue.put((epoch, cpu_state))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n>>> EARLY STOPPING triggered at Epoch {epoch} <<<")
            print(f">>> Validation loss did not improve for {patience} consecutive epochs. Breaking cycle.")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(">>> Restored best model weights <<<")
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=best_val_loss,
                            path=checkpoint_dir, file_name=f"finished_{stage_name}")
            break

        if epoch == max_epochs:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(">>> Restored best model weights <<<")
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=best_val_loss,
                            path=checkpoint_dir, file_name=f"finished_{stage_name}_max_epochs")

    if vis_finish_event is not None:
        vis_finish_event.set()

    print(f"Finished {stage_name} training in {time.time() - start_time:.2f} seconds")
    return train_loss_history, val_loss_history
    

def plot_learning_curves(train_losses, val_losses, title="Model Loss Progression", log_scale: bool = False, save_to_csv: bool = False, stage: str = None):
    if save_to_csv:
        if not stage:
            raise ValueError("The 'stage' parameter is required when 'save_to_csv' is True.")
        
        df = pd.DataFrame({
            'Epoch': range(len(train_losses)),
            'Training Loss': train_losses,
            'Validation Loss': val_losses
        })
        
        #os.makedirs('/kaggle/working', exist_ok=True)
        csv_path = f'/kaggle/working/{stage}_loss.csv'
        df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))

    if log_scale:
        plt.yscale('log')
        
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2, linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
def mean_and_std_for_normalization(dataloader: torch.utils.data.DataLoader) -> Tuple[np.array, np.array]:
    # code from https://stackoverflow.com/questions/53735817/normalising-images-before-learning-in-pytorch access time at 21.06.2026 of 23:07
    data_mean = [] # Mean of the dataset
    data_std1 = [] # std with ddof = 1
    for _, (data,_) in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data.numpy()
    
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)
    
        data_mean.append(batch_mean)
        data_std1.append(batch_std1)

    return np.array(data_mean).mean(axis=0), np.array(data_std1).mean(axis=0)

def seed_everything(seed=42):
    # 1. Standard Python and OS replication
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch CPU and CUDA global seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    
    # 3. CuDNN back-end determinism (Crucial for CNNs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)