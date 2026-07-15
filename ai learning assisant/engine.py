import torch
import torch.nn as nn


def calculate_accuracy(predictions, labels):
    """
    Calculate batch accuracy.
    """

    predicted = torch.argmax(predictions, dim=1)

    correct = (predicted == labels).float()

    accuracy = correct.sum() / len(correct)

    return accuracy


def train_epoch(model, iterator, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """

    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    total_batches = 0

    try:

        for batch in iterator:

            text = batch.text.to(device)
            text_lengths = batch.lengths.cpu()
            labels = batch.labels.to(device)

            optimizer.zero_grad()

            outputs = model(text, text_lengths)

            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)

            loss.backward()

            # Prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            total_batches += 1

        if total_batches == 0:
            return 0.0, 0.0

        return (
            epoch_loss / total_batches,
            epoch_acc / total_batches
        )

    except Exception as e:

        raise RuntimeError(
            f"Training failed during epoch.\nReason: {str(e)}"
        )