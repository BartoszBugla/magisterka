from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import BertTokenizer


class ABSADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: BertTokenizer,
        label_col: str = "multihot_vector",
        max_length: int = 128,
    ):
        self.n_examples = len(dataframe)

        print("Tokenizer step:")

        texts = list(dataframe["text"].values)

        self.inputs = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            # Maksymalny rozmiar "tokena"
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # index slownika dla każdego slowa
        self.sequence_len = self.inputs["input_ids"].shape[-1]

        labels = list(dataframe[label_col].values)

        self.labels = torch.tensor(labels, dtype=torch.float32)

        print(
            f"Finished! {self.n_examples} examples, label shape {self.labels.shape}\n"
        )

    # Calkowita ilość rzędów w datasecie
    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        inputs = {key: self.inputs[key][i] for key in self.inputs.keys()}
        return inputs, self.labels[i]
