from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

from transformers import PreTrainedTokenizerBase

from config.global_config import TRAIN_ASPECTS, SENTIMENT_LABELS


class ABSADataset(Dataset):
    """
    Dataset dla architektury BertForABSAMultiHead (softmax per aspekt).
    Labels jako indeksy klas (0-3) dla każdego aspektu.
    
    Args:
        dataframe: DataFrame z kolumnami aspektów (wartości: 'positive', 'neutral', 'negative', 'notmentioned')
        tokenizer: PreTrainedTokenizerBase
        max_length: maksymalna długość sekwencji
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.n_examples = len(dataframe)
        
        print("Tokenizer step:")
        
        texts = list(dataframe["text"].values)
        
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
                sentiment = row[aspect]
                if pd.isna(sentiment):
                    sentiment = "notmentioned"
                idx = sentiment_to_idx.get(sentiment, sentiment_to_idx["notmentioned"])
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
        """Zwraca labels jako numpy array do obliczenia wag klas."""
        return self.labels.numpy()
