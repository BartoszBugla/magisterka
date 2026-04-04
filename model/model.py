import torch
import torch.nn as nn

from transformers import BertModel

from config.global_config import SENTIMENT_LABELS, TRAIN_ASPECTS


class BertForABSA(nn.Module):
    def __init__(self, bert: BertModel):

        super(BertForABSA, self).__init__()

        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            bert.config.hidden_size, len(TRAIN_ASPECTS) * len(SENTIMENT_LABELS)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids, return_dict=False
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
