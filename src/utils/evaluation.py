# %% [code]
import torch
from sklearn.metrics import f1_score, matthews_corrcoef


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    local_test_loss_sum = 0.0
    local_test_steps = 0
    correct_predictions = 0
    total_samples = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            local_test_loss_sum += loss.item()
            local_test_steps += 1

            predicted_classes = output.argmax(dim=1)
            correct_predictions += (predicted_classes == target).sum().item()
            total_samples += target.size(0)

            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    if local_test_steps > 0:
        avg_test_loss = local_test_loss_sum / local_test_steps
        accuracy = correct_predictions / total_samples
        macro_f1 = f1_score(all_targets, all_preds, average="macro")
        mcc = matthews_corrcoef(all_targets, all_preds)
    else:
        avg_test_loss = float("inf")
        accuracy, macro_f1, mcc = 0.0, 0.0, 0.0

    print(
        f"=== Test set: Loss: {avg_test_loss:.6f} | Acc: {correct_predictions}/{total_samples} ({accuracy * 100:.2f}%) | "
        f"Macro F1: {macro_f1:.4f} | MCC: {mcc:.4f} ===\n"
    )

    return avg_test_loss, accuracy, macro_f1, mcc