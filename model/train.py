from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from model.prepare_dataset import ABSADataCollator, SENTIMENT_IGNORE_INDEX

LABEL_NAMES = ("mention_labels", "sentiment_labels")


def _balanced_weights(
    labels: np.ndarray, n_classes: int, ignore: int | None = None
) -> torch.Tensor | None:
    flat = labels.flatten()
    if ignore is not None:
        flat = flat[flat != ignore]
    if flat.size == 0:
        return None
    classes = np.arange(n_classes)
    w = compute_class_weight(
        "balanced", classes=classes, y=np.concatenate([flat, classes])
    )
    return torch.tensor(w, dtype=torch.float32)


def compute_sentiment_class_weights(labels: np.ndarray) -> torch.Tensor | None:
    return _balanced_weights(labels, 3, ignore=SENTIMENT_IGNORE_INDEX)


def compute_mention_class_weights(labels: np.ndarray) -> torch.Tensor | None:
    return _balanced_weights(labels, 2)


def _compute_metrics(eval_pred):
    """Compute sentiment F1 (mentioned only), detection F1, and overall 4-class F1."""
    (mention_logits, sentiment_logits), (mention_labels, sentiment_labels) = eval_pred

    m_pred = mention_logits.argmax(-1).flatten()
    s_pred = sentiment_logits.argmax(-1).flatten()
    m_true = mention_labels.flatten()
    s_true = sentiment_labels.flatten()

    mask = m_true == 1
    sent_f1 = (
        f1_score(
            s_true[mask],
            s_pred[mask],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
        if mask.sum()
        else 0.0
    )
    det_f1 = f1_score(m_true, m_pred, average="binary", zero_division=0)

    ov_true = np.where(m_true == 0, 0, s_true + 1)
    ov_pred = np.where(m_pred == 0, 0, s_pred + 1)
    ov_f1 = f1_score(
        ov_true, ov_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0
    )

    return {
        "sentiment_macro_f1_mentioned": float(sent_f1),
        "detection_f1": float(det_f1),
        "overall_macro_f1": float(ov_f1),
    }


class _ABSATrainer(Trainer):
    """Trainer that keeps dual labels and delegates loss to the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = list(LABEL_NAMES)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def train_model(
    model,
    train_dataset,
    val_dataset,
    *,
    tokenizer,
    output_dir="./absa_results",
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing_factor=0.05,
    metric_for_best_model="sentiment_macro_f1_mentioned",
    early_stopping_patience=2,
    seed=25,
):
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=2,
        label_smoothing_factor=label_smoothing_factor,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        seed=seed,
        report_to=["none"],
    )

    trainer = _ABSATrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        data_collator=ABSADataCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=1e-4,
            )
        ],
    )
    trainer.train()
    return trainer
