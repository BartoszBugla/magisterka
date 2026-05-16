from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

ARCHITECTURE_VERSION = "dual_head"
SENTIMENT_IGNORE_INDEX = -100


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha) if alpha is not None else None
        self.alpha = getattr(self, "alpha", None)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        valid = targets != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0.0
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss[valid].mean()


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


class ABSAModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_aspects: int,
        num_mention_classes: int = 2,
        num_sentiment_classes: int = 3,
        dropout_rate: float = 0.2,
        sentiment_class_weights: torch.Tensor | None = None,
        mention_pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        mention_loss_weight: float = 0.75,
        sentiment_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_mention_classes = num_mention_classes
        self.num_sentiment_classes = num_sentiment_classes
        self.mention_loss_weight = mention_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight
        self.architecture_version = ARCHITECTURE_VERSION

        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        h = self.encoder.config.hidden_size

        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(dropout_rate)
        self.mention_heads = nn.ModuleList(
            [nn.Linear(h, num_mention_classes) for _ in range(num_aspects)]
        )
        self.sentiment_heads = nn.ModuleList(
            [nn.Linear(h, num_sentiment_classes) for _ in range(num_aspects)]
        )

        self.mention_loss_fn = nn.CrossEntropyLoss(weight=mention_pos_weight)
        self.sentiment_loss_fn = FocalLoss(
            alpha=sentiment_class_weights,
            gamma=focal_gamma,
            ignore_index=SENTIMENT_IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        mention_labels: torch.Tensor | None = None,
        sentiment_labels: torch.Tensor | None = None,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        x = self.dropout(self.norm(_mean_pool(out.last_hidden_state, attention_mask)))

        mention_logits = torch.stack([h(x) for h in self.mention_heads], dim=1)
        sentiment_logits = torch.stack([h(x) for h in self.sentiment_heads], dim=1)

        if mention_labels is None and sentiment_labels is None:
            return {
                "mention_logits": mention_logits,
                "sentiment_logits": sentiment_logits,
            }

        loss = x.sum() * 0.0
        if mention_labels is not None:
            loss = loss + self.mention_loss_weight * self.mention_loss_fn(
                mention_logits.view(-1, self.num_mention_classes),
                mention_labels.view(-1),
            )
        if sentiment_labels is not None:
            loss = loss + self.sentiment_loss_weight * self.sentiment_loss_fn(
                sentiment_logits.view(-1, self.num_sentiment_classes),
                sentiment_labels.view(-1),
            )

        return {
            "loss": loss,
            "mention_logits": mention_logits,
            "sentiment_logits": sentiment_logits,
        }
