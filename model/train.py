try:
    from model.choose_architecture import choose_architecture
except ImportError:
    from choose_architecture import choose_architecture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from time import perf_counter
from tqdm.auto import tqdm
import numpy as np
from transformers import get_scheduler

# Zawsze takie same na danym urządzeniu
device = choose_architecture()


def get_multilabel_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-label accuracy across all outputs."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return correct, total


def get_sentiment_metrics(
    logits: torch.Tensor, labels: torch.Tensor, num_sentiments: int = 4
) -> dict:
    """
    Calculate metrics that account for class imbalance.

    For each aspect (8 aspects), we have 4 sentiment classes.
    This returns metrics broken down by sentiment type.

    Returns dict with:
    - sentiment_accuracy: dict mapping sentiment_idx -> accuracy for that sentiment
    - macro_f1: macro-averaged F1 across all sentiment classes
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    # Reshape to (batch, num_aspects, num_sentiments)
    batch_size = logits.shape[0]
    num_aspects = logits.shape[1] // num_sentiments

    preds_reshaped = preds.view(batch_size, num_aspects, num_sentiments)
    labels_reshaped = labels.view(batch_size, num_aspects, num_sentiments)

    metrics = {}

    # Calculate per-sentiment metrics (0=positive, 1=neutral, 2=negative, 3=notmentioned)
    for sent_idx in range(num_sentiments):
        pred_sent = preds_reshaped[:, :, sent_idx].flatten()
        label_sent = labels_reshaped[:, :, sent_idx].flatten()

        # True positives, false positives, false negatives
        tp = ((pred_sent == 1) & (label_sent == 1)).sum().float()
        fp = ((pred_sent == 1) & (label_sent == 0)).sum().float()
        fn = ((pred_sent == 0) & (label_sent == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f"sent_{sent_idx}_f1"] = f1.item()
        metrics[f"sent_{sent_idx}_precision"] = precision.item()
        metrics[f"sent_{sent_idx}_recall"] = recall.item()

    # Macro F1 (average across sentiments)
    macro_f1 = (
        sum(metrics[f"sent_{i}_f1"] for i in range(num_sentiments)) / num_sentiments
    )
    metrics["macro_f1"] = macro_f1

    # Weighted F1 excluding "notmentioned" (index 3) - this is more meaningful
    # because it shows how well we detect actual sentiments
    sentiment_f1 = sum(metrics[f"sent_{i}_f1"] for i in range(3)) / 3
    metrics["sentiment_f1"] = sentiment_f1

    return metrics


def calculate_pos_weights(vectors: np.ndarray, device) -> torch.Tensor:
    """
    Calculates the pos_weight tensor for BCEWithLogitsLoss.
    vectors: The (N, 32) numpy array of your training data.
    """
    total_samples = vectors.shape[0]

    # Count how many times '1' appears in each of the 32 columns
    positive_counts = vectors.sum(axis=0)

    # Hack: Prevent "division by zero" if a class literally NEVER appears in your dataset
    # We clip the minimum count to 1.
    positive_counts = np.clip(positive_counts, a_min=1, a_max=None)

    # Calculate negative counts
    negative_counts = total_samples - positive_counts

    # The magic formula
    pos_weights = negative_counts / positive_counts

    # Convert to a PyTorch tensor and send it to your Mac GPU
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    return pos_weights_tensor


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    learning_rate: float,
    epochs: int,
    pos_weights: torch.Tensor = None,
):
    start = perf_counter()
    history = []

    if pos_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        print(f"Using weighted loss with pos_weights: {pos_weights[:8]}...")
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))

    model = model.to(device)

    for epoch_num in range(epochs):
        epoch_start = perf_counter()

        # --- Training ---
        model.train()
        total_loss_train = 0.0
        train_correct = 0
        train_total = 0

        for inputs, label in train_dataloader:
            label = label.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            batch_loss = criterion(outputs, label)
            batch_loss.backward()

            total_loss_train += batch_loss.item()
            correct, total = get_multilabel_accuracy(outputs, label)
            train_correct += correct.item()
            train_total += total

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # --- Validation ---
        model.eval()
        total_loss_val = 0.0
        val_correct = 0
        val_total = 0

        # Collect all predictions and labels for better metrics
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                val_input = {k: v.to(device) for k, v in val_input.items()}

                outputs = model(**val_input)
                batch_loss = criterion(outputs, val_label)
                total_loss_val += batch_loss.item()

                correct, total = get_multilabel_accuracy(outputs, val_label)
                val_correct += correct.item()
                val_total += total

                all_val_logits.append(outputs)
                all_val_labels.append(val_label)

        # Calculate detailed metrics on validation set
        all_val_logits = torch.cat(all_val_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        val_metrics = get_sentiment_metrics(all_val_logits, all_val_labels)

        epoch_time = perf_counter() - epoch_start
        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0

        print(
            f"\nEpoch {epoch_num + 1}/{epochs} "
            f"| Train Loss: {total_loss_train / len(train_dataloader):.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {total_loss_val / max(len(val_dataloader), 1):.4f} "
            f"| Val Acc: {val_acc:.4f} "
            f"| Sentiment F1: {val_metrics['sentiment_f1']:.4f} "
            f"| Time: {epoch_time:.1f}s"
        )

        # Detailed per-sentiment F1 (helpful for debugging)
        sentiment_names = ["positive", "neutral", "negative", "notmentioned"]
        f1_details = " | ".join(
            [
                f"{s}: {val_metrics[f'sent_{i}_f1']:.3f}"
                for i, s in enumerate(sentiment_names)
            ]
        )
        print(f"  Per-sentiment F1: {f1_details}")

        history.append(
            {
                "epoch": epoch_num + 1,
                "train_loss": total_loss_train / len(train_dataloader),
                "train_acc": train_acc,
                "val_loss": total_loss_val / max(len(val_dataloader), 1),
                "val_acc": val_acc,
                "sentiment_f1": val_metrics["sentiment_f1"],
                "macro_f1": val_metrics["macro_f1"],
                "per_sentiment_f1": {
                    s: val_metrics[f"sent_{i}_f1"]
                    for i, s in enumerate(sentiment_names)
                },
                "epoch_time": epoch_time,
            }
        )

    time_taken = perf_counter() - start
    print(f"\nTotal training time: {time_taken:.1f}s")
    return history
