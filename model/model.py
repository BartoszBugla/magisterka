import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss
from transformers import AutoModel


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class ABSAModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_aspects: int,
        num_sentiments: int,
        dropout_rate: float = 0.2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments

        self.encoder = AutoModel.from_pretrained(model_name)
        # UWAGA: Usunięto zamrażanie warstw. Trenujemy całego BERTa.

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size, num_aspects * num_sentiments
        )
        self.loss_fn = (
            CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else CrossEntropyLoss()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(cls_repr).view(
            -1, self.num_aspects, self.num_sentiments
        )

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_sentiments), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return logits
