import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel

from config.global_config import SENTIMENT_LABELS, TRAIN_ASPECTS


def _encode_backbone(
    bert: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None,
    _accepts_token_type_ids: bool,
) -> torch.Tensor:
    kwargs: dict = dict(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )

    if token_type_ids is not None and _accepts_token_type_ids:
        kwargs["token_type_ids"] = token_type_ids

    outputs = bert(**kwargs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    return outputs.last_hidden_state[:, 0]


class ABSAModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_aspects: int = len(TRAIN_ASPECTS),
        num_sentiments: int = len(SENTIMENT_LABELS),
        dropout: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        self._accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(self.bert.forward).parameters
        )
        self.dropout = nn.Dropout(dropout)
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.class_weights = class_weights

        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.config.hidden_size, num_sentiments)
                for _ in range(num_aspects)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict | torch.Tensor:
        pooled = _encode_backbone(
            self.bert,
            input_ids,
            attention_mask,
            token_type_ids,
            self._accepts_token_type_ids,
        )
        pooled = self.dropout(pooled)

        logits = torch.stack([clf(pooled) for clf in self.classifiers], dim=1)

        if labels is not None:
            w = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(
                logits.view(-1, self.num_sentiments), labels.view(-1), weight=w,
            )
            return {"loss": loss, "logits": logits}

        return logits
