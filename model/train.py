import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments
import torch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1).flatten()
    labels_flat = labels.flatten()

    min_len = min(len(preds), len(labels_flat))
    preds = preds[:min_len]
    labels_flat = labels_flat[:min_len]

    report = classification_report(
        labels_flat, preds, output_dict=True, zero_division=0
    )

    sentiment_f1 = f1_score(
        labels_flat, preds, labels=[0, 1, 2], average="macro", zero_division=0
    )

    metrics = {
        "sentiment_f1": sentiment_f1,
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
    }
    for class_id in range(4):
        class_str = str(class_id)
        if class_str in report:
            metrics[f"sent_{class_id}_f1"] = report[class_str]["f1-score"]

    return metrics


def compute_class_weights(labels_array: np.ndarray) -> torch.Tensor:
    weights = compute_class_weight(
        "balanced", classes=np.unique(labels_array), y=labels_array.flatten()
    )
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./absa_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        weight_decay=0.05,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        remove_unused_columns=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
