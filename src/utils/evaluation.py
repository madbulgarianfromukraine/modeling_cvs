import torch


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    local_test_loss_sum = 0.0
    local_test_steps = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            local_test_loss_sum += loss.item()
            local_test_steps += 1

            predicted_classes = output.argmax(dim=1, keepdim=True)
            correct_predictions += (
                predicted_classes.eq(target.view_as(predicted_classes))
                .sum()
                .item()
            )
            total_samples += target.size(0)

    if local_test_steps > 0:
        avg_test_loss = local_test_loss_sum / local_test_steps
        accuracy = correct_predictions / total_samples
    else:
        avg_test_loss = float("inf")
        accuracy = 0.0

    print(
        f"=== Test set: Final Average Loss: {avg_test_loss:.6f} | Accuracy: {correct_predictions}/{total_samples} ({accuracy * 100:.2f}%) === \n"
    )

    return avg_test_loss, accuracy