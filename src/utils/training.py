import os
import time
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from typing import List, Tuple, Optional

os.makedirs("/kaggle/working/", exist_ok=True)
def save_checkpoint(model, optimizer, epoch, loss, path="/kaggle/working/", file_name=None):
    """
    Saves a 'Neutral' checkpoint by manually migrating tensors to CPU.
    Bypasses torch_xla version issues.
    """
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
    """
    Loads checkpoint into CPU first to strip headers, then moves to current device.
    """
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
                         optimizer, loss_fn, device, log_interval, checkpoint_dir, stage_name: str = "nat_images") -> Tuple[float, Optional[float]]:
    epoch_start_time = time.time()

    model.train()
    local_train_loss_sum = 0.0
    local_train_steps = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
    
        local_train_loss_sum += loss.item()
        local_train_steps += 1

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader)*128} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        only_once = True
    
    
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


def plot_learning_curves(train_losses, val_losses, title="Model Loss Progression"):
    """
    Plots the training and testing loss curves.
    
    Args:
        train_losses (list or numpy array): A list of training loss values per epoch.
        test_losses (list or numpy array): A list of test/validation loss values per epoch.
        title (str): The title of the plot.
    """
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the lines
    # We use a solid line for training and a dashed line for testing for clear contrast
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2, linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    
    # Add labels and title
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # Add a grid for easier reading of values
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Add the legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Display the plot
    plt.show()