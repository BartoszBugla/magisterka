from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from transformers import PreTrainedTokenizerBase

from config.global_config import TRAIN_ASPECTS, SENTIMENT_LABELS


def coerce_text_for_tokenizer(value) -> str:
    """Turn a dataframe cell into a string safe for ``tokenizer(...)``."""
    if value is None:
        return " "
    try:
        if pd.isna(value):
            return " "
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return " "
    return s


def _canonical_sentiment_label(raw) -> str:
    """Map CSV strings to keys in ``SENTIMENT_LABELS`` (case/spacing tolerant)."""
    if pd.isna(raw):
        return "notmentioned"
    key = str(raw).strip().lower().replace(" ", "").replace("_", "")
    if key in SENTIMENT_LABELS:
        return key
    return "notmentioned"


class ABSADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.n_examples = len(dataframe)

        print("Tokenizer step:")

        if "text" not in dataframe.columns:
            raise KeyError("DataFrame must contain a 'text' column for ABSADataset")

        texts = [coerce_text_for_tokenizer(v) for v in dataframe["text"].values]

        self.inputs = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        self.sequence_len = self.inputs["input_ids"].shape[-1]

        sentiment_to_idx = {s: i for i, s in enumerate(SENTIMENT_LABELS)}

        labels = []
        for _, row in dataframe.iterrows():
            row_labels = []
            for aspect in TRAIN_ASPECTS:
                sentiment = _canonical_sentiment_label(row[aspect])
                idx = sentiment_to_idx[sentiment]
                row_labels.append(idx)
            labels.append(row_labels)

        self.labels = torch.tensor(labels, dtype=torch.long)

        print(f"Finished! {self.n_examples} examples, label shape {self.labels.shape}")
        print(f"  Aspects: {len(TRAIN_ASPECTS)}, Sentiments: {len(SENTIMENT_LABELS)}\n")

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        item = {key: self.inputs[key][i] for key in self.inputs.keys()}
        item["labels"] = self.labels[i]
        return item

    def get_labels_numpy(self) -> np.ndarray:
        return self.labels.numpy()
