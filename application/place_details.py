from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from application.map_components import place_group_keys
from config.global_config import SENTIMENT_LABELS, SentimentLabel


METADATA_DIR = Path("statics/datasets")

t = {
    "place_details_avg_rating": "Średnia ocena",
}

_SENTIMENT_COLORS: dict[str, str] = {
    SentimentLabel.POSITIVE: "#4CAF50",
    SentimentLabel.NEUTRAL: "#FFC107",
    SentimentLabel.NEGATIVE: "#F44336",
    SentimentLabel.NOTMENTIONED: "#9E9E9E",
}

_SENTIMENT_DISPLAY: dict[str, str] = {
    SentimentLabel.POSITIVE: "Positive",
    SentimentLabel.NEUTRAL: "Neutral",
    SentimentLabel.NEGATIVE: "Negative",
    SentimentLabel.NOTMENTIONED: "Not mentioned",
}

_REVIEW_TEXT_PREVIEW_CHARS = 600


def list_metadata_files() -> list[Path]:
    if not METADATA_DIR.exists():
        return []
    return sorted(METADATA_DIR.glob("metadata_*.csv"))


@st.cache_data(show_spinner=False)
def load_metadata(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_metadata_for_place(
    metadata_df: pd.DataFrame,
    *,
    gmap_id: str | None = None,
    place_name: str | None = None,
) -> dict[str, Any] | None:
    if metadata_df.empty:
        return None

    if place_name and "name" in metadata_df.columns:
        match = metadata_df[metadata_df["name"].astype(str) == str(place_name)]
        if not match.empty:
            return match.iloc[0].to_dict()

    return None


def reviews_for_place(
    df: pd.DataFrame, place_keys: pd.Series, target_key: str
) -> pd.DataFrame:
    return df.loc[place_keys == target_key].copy()


def compute_place_keys(df: pd.DataFrame) -> pd.Series:
    return place_group_keys(df)


def sentiment_counts(reviews: pd.DataFrame, aspect: str) -> dict[str, int]:
    counts = {label: 0 for label in SENTIMENT_LABELS}
    if aspect not in reviews.columns:
        return counts
    for val in reviews[aspect]:
        key = str(val).strip().lower() if pd.notna(val) else "notmentioned"
        counts[key] = counts.get(key, 0) + 1
    return counts


def filter_reviews(
    reviews: pd.DataFrame,
    aspect: str,
    sentiment: str,
) -> pd.DataFrame:
    if aspect not in reviews.columns:
        return reviews
    return reviews[
        reviews[aspect].astype(str).str.strip().str.lower() == sentiment.lower()
    ]


def render_sentiment_bar(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    if total == 0:
        st.caption("No sentiment data for this aspect.")
        return

    for label in SENTIMENT_LABELS:
        count = counts.get(label, 0)
        pct = count / total * 100
        color = _SENTIMENT_COLORS.get(label, "#9E9E9E")
        display = _SENTIMENT_DISPLAY.get(label, label)

        st.markdown(
            f"<div style='display:flex;align-items:center;margin-bottom:4px;gap:8px'>"
            f"<span style='min-width:110px;font-size:0.85em'>{display}</span>"
            f"<div style='flex:1;background:#333;border-radius:4px;height:18px;overflow:hidden'>"
            f"<div style='width:{pct:.1f}%;background:{color};height:100%;"
            f"border-radius:4px;min-width:{'2px' if count else '0'}'></div>"
            f"</div>"
            f"<span style='min-width:32px;text-align:right;font-size:0.85em'>{count}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _aspect_tag_html(aspect: str, value: str) -> str:
    val = value.strip().lower()
    color = _SENTIMENT_COLORS.get(val, "#9E9E9E")

    if val not in _SENTIMENT_COLORS:
        return ""

    return (
        f"<span style='display:inline-flex;align-items:center;gap:3px;"
        f"margin-right:10px;font-size:0.8em'>"
        f"<span style='width:8px;height:8px;border-radius:50%;"
        f"background:{color};display:inline-block'></span>"
        f"{aspect}</span>"
    )


def _resolved_place_category(
    place_info: dict[str, Any], metadata: dict[str, Any] | None
) -> str:
    category = place_info.get("place_category", "—")
    if not metadata:
        return category
    meta_cat = metadata.get("category")
    if meta_cat and str(meta_cat) != "nan":
        return str(meta_cat).replace("||", " / ")
    return category


def _render_place_dialog_header(
    name: str, category: str, reviews: pd.DataFrame
) -> None:
    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    col1.metric("Kategoria", category)

    col2.metric("Opinie", len(reviews))
    if "rating" in reviews.columns:
        avg = pd.to_numeric(reviews["rating"], errors="coerce").mean()
        if pd.notna(avg):
            col3.metric(t["place_details_avg_rating"], f"{avg:.1f} / 5")


def _dialog_sentiment_profile_aspect(
    reviews: pd.DataFrame,
    available_aspects: list[str],
    selected_aspect: str,
) -> str:
    st.divider()
    profile_aspect = st.selectbox(
        "Profil sentymentu — aspekt",
        available_aspects,
        index=(
            available_aspects.index(selected_aspect)
            if selected_aspect in available_aspects
            else 0
        ),
        key="dlg_profile_aspect",
    )
    render_sentiment_bar(sentiment_counts(reviews, profile_aspect))
    return profile_aspect


def _dialog_review_filter_widgets(
    available_aspects: list[str], profile_aspect: str
) -> tuple[str, str]:
    st.divider()
    fc1, fc2 = st.columns(2)
    with fc1:
        sent_filter = st.selectbox(
            "Filtruj po sentymentach",
            ["All"] + [_SENTIMENT_DISPLAY[s] for s in SENTIMENT_LABELS],
            key="dlg_sent_filter",
        )
    with fc2:
        asp_filter = st.selectbox(
            "Aspekt dla filtru",
            available_aspects,
            index=(
                available_aspects.index(profile_aspect)
                if profile_aspect in available_aspects
                else 0
            ),
            key="dlg_asp_filter",
        )
    return sent_filter, asp_filter


def _dialog_filtered_reviews(
    reviews: pd.DataFrame, sent_filter: str, asp_filter: str
) -> pd.DataFrame:
    if sent_filter == "All":
        return reviews
    reverse_map = {v: k for k, v in _SENTIMENT_DISPLAY.items()}
    return filter_reviews(
        reviews, asp_filter, reverse_map.get(sent_filter, sent_filter)
    )


def _render_dialog_review_card(
    idx: int, row: pd.Series, available_aspects: list[str]
) -> None:
    text = str(row.get("text", "")) if pd.notna(row.get("text")) else ""
    rating_val = row.get("rating")
    rating_str = f"Ocena: {rating_val}" if pd.notna(rating_val) else ""
    header = f"**#{idx}**"
    if rating_str:
        header += f" &nbsp; {rating_str}"
    truncated = text[:_REVIEW_TEXT_PREVIEW_CHARS] + (
        "..." if len(text) > _REVIEW_TEXT_PREVIEW_CHARS else ""
    )
    with st.container(border=True):
        st.markdown(header)
        st.write(truncated)
        tags = "".join(
            _aspect_tag_html(asp, str(row[asp]))
            for asp in available_aspects
            if asp in row.index and pd.notna(row[asp])
        )
        if tags:
            st.markdown(tags, unsafe_allow_html=True)


def _render_dialog_review_list(
    display: pd.DataFrame, available_aspects: list[str], total: int
) -> None:
    st.caption(f"Pokazywanie {len(display)} z {total} opinii")
    if display.empty:
        st.info("Brak opinii pasujących do wybranych filtrów.")
        return
    for idx, (_, row) in enumerate(display.iterrows(), start=1):
        _render_dialog_review_card(idx, row, available_aspects)


def render_place_dialog(
    place_info: dict[str, Any],
    reviews: pd.DataFrame,
    metadata: dict[str, Any] | None,
    selected_aspect: str,
    available_aspects: list[str],
) -> None:
    name = place_info.get("place_name", "Unknown")
    category = _resolved_place_category(place_info, metadata)
    _render_place_dialog_header(name, category, reviews)

    profile_aspect = _dialog_sentiment_profile_aspect(
        reviews, available_aspects, selected_aspect
    )
    sent_filter, asp_filter = _dialog_review_filter_widgets(
        available_aspects, profile_aspect
    )
    display = _dialog_filtered_reviews(reviews, sent_filter, asp_filter)
    _render_dialog_review_list(display, available_aspects, len(reviews))
