import emoji
import re
import pandas as pd


def clean_translation_tags(text: str) -> str:
    text = str(text)

    if "(Translated by Google)" in text:
        # Usuwamy frazę początkową
        text = text.replace("(Translated by Google)", "").strip()
        # Odcinamy tekst oryginalny, zostawiając tylko tłumaczenie
        text = text.split("(Original)")[0].strip()

    return text


def clean_emojis(text: str) -> str:
    text = emoji.replace_emoji(text, replace="")
    # Remove extra whitespace left behind
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].str.replace("\n", "")
    df["text"] = df["text"].apply(clean_translation_tags)
    df = df.drop_duplicates(subset=["text"])
    df["text"] = df["text"].apply(clean_emojis)
    return df
