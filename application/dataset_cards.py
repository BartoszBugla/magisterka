from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from application.dataset_types import DatasetType
from application.svg_icons import inline_icon_markup
from application.results_repository import EntryMetadata, repository as repo
from config.global_config import TRAIN_ASPECTS


@st.cache_data(show_spinner=False, ttl=300)
def get_dataset_stats(csv_filename: str) -> dict[str, Any]:
    """Load basic statistics for a dataset card preview."""
    try:
        df = pd.read_csv(repo.get_csv_path(csv_filename))
    except Exception:
        return {"rows": 0, "places": 0, "aspects": []}

    n_rows = len(df)
    n_places = 0
    if "gmap_id" in df.columns:
        n_places = df["gmap_id"].nunique()
    elif "name" in df.columns:
        n_places = df["name"].nunique()

    aspects = [a for a in TRAIN_ASPECTS if a in df.columns]

    return {"rows": n_rows, "places": n_places, "aspects": aspects}


def invalidate_stats_cache(csv_filename: str) -> None:
    get_dataset_stats.clear()


TYPE_ICON_KEYS: dict[DatasetType, str] = {
    DatasetType.RAW_REVIEWS: "file_text",
    DatasetType.CLEANED: "sparkles",
    DatasetType.LABELLED_AI: "bot",
    DatasetType.LABELLED_HUMAN: "user",
}

TYPE_COLORS: dict[DatasetType, str] = {
    DatasetType.RAW_REVIEWS: "#607D8B",
    DatasetType.CLEANED: "#2196F3",
    DatasetType.LABELLED_AI: "#9C27B0",
    DatasetType.LABELLED_HUMAN: "#4CAF50",
}


def format_date(iso_str: str) -> str:
    """Parse ISO date and return a short human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y")
    except Exception:
        return "—"


def render_dataset_card_info(entry: EntryMetadata, stats: dict[str, Any]) -> None:
    icon_key = TYPE_ICON_KEYS.get(entry.dataset_type, "folder")
    icon_html = inline_icon_markup(icon_key, size_em="1.3em")
    color = TYPE_COLORS.get(entry.dataset_type, "#9E9E9E")
    type_label = entry.dataset_type.label_pl

    n_rows = stats.get("rows", 0)
    n_places = stats.get("places", 0)
    aspects = stats.get("aspects", [])

    st.markdown(
        f"{icon_html}" f"<b style='font-size:1.1em'>{entry.csv_filename}</b>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.8em'>{type_label}</span>",
        unsafe_allow_html=True,
    )

    meta_parts = []
    if n_rows:
        meta_parts.append(f"**{n_rows:,}** reviews")
    if n_places:
        meta_parts.append(f"**{n_places:,}** places")
    if aspects:
        meta_parts.append(f"{len(aspects)} aspects")

    if meta_parts:
        st.caption(" · ".join(meta_parts))

    date_str = format_date(entry.created_at)
    st.caption(f"Created: {date_str}")

    if entry.notes:
        st.caption(f"_{entry.notes}_")


def render_dataset_card_readonly(
    entry: EntryMetadata,
    stats: dict[str, Any],
    button_label: str = "Open",
    button_key_prefix: str = "open",
) -> bool:
    with st.container(border=True):
        col_main, col_btn = st.columns([4, 1])

        with col_main:
            render_dataset_card_info(entry, stats)

        with col_btn:
            st.write("")
            clicked = st.button(
                button_label,
                key=f"{button_key_prefix}_{entry.csv_filename}",
                use_container_width=True,
            )

        return clicked


def render_dataset_card_editable(
    entry: EntryMetadata,
    stats: dict[str, Any],
) -> tuple[bool, bool, bool]:
    with st.container(border=True):
        render_dataset_card_info(entry, stats)
        st.write("")
        btn_cols = st.columns(3)

        with btn_cols[0]:
            edit_clicked = st.button(
                " ",
                key=f"edit_{entry.csv_filename}",
                help="Edit metadata",
                use_container_width=True,
                icon=":material/edit:",
            )

        with btn_cols[1]:
            csv_path = repo.get_csv_path(entry.csv_filename)
            st.download_button(
                " ",
                data=csv_path.read_bytes(),
                file_name=entry.csv_filename,
                mime="text/csv",
                key=f"dl_{entry.csv_filename}",
                help="Download CSV",
                use_container_width=True,
                icon=":material/download:",
            )
            download_clicked = False

        with btn_cols[2]:
            delete_clicked = st.button(
                " ",
                key=f"del_{entry.csv_filename}",
                help="Delete dataset",
                use_container_width=True,
                type="primary",
                icon=":material/delete:",
            )

    return edit_clicked, download_clicked, delete_clicked


def group_entries_by_type(
    entries: list[EntryMetadata],
) -> tuple[list[EntryMetadata], list[EntryMetadata]]:
    labelled = [
        e
        for e in entries
        if e.dataset_type in (DatasetType.LABELLED_AI, DatasetType.LABELLED_HUMAN)
    ]
    other = [e for e in entries if e not in labelled]
    return labelled, other
