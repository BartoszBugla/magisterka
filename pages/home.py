from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from application.dataset_types import DatasetType
from application.dataset_upload_validation import REPOSITORY_MAP_SYNTHETIC_ASPECT
from application.map_components import build_map
from application.results_repository import repository as repo
from config.global_config import NON_ASPECT_COLUMNS, TRAIN_ASPECTS


def prepare_dataframe_for_map_and_table(df: pd.DataFrame) -> pd.DataFrame:
    return df


def aspect_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        c
        for c in df.columns
        if c != REPOSITORY_MAP_SYNTHETIC_ASPECT and c.lower() not in NON_ASPECT_COLUMNS
    )


st.title("Map")

entries = repo.list_entries()
if not entries:
    st.info(
        "No datasets in the repository. Upload CSV files on the **Repository** page."
    )
    st.stop()

by_name = {e.csv_filename: e for e in entries}
choice = st.selectbox(
    "Dataset",
    options=list(by_name),
    format_func=lambda fn: f"{by_name[fn].dataset_type.label_pl} — {fn}",
)
meta = by_name[choice]

try:
    df_raw = pd.read_csv(repo.get_csv_path(choice))
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

if df_raw.empty:
    st.warning("This dataset has no rows.")
    st.stop()

for col in ("latitude", "longitude"):
    if col not in df_raw.columns:
        st.error(f"Missing required column: `{col}`.")
        st.stop()

prepared = prepare_dataframe_for_map_and_table(df_raw)

is_labelled = meta.dataset_type in (
    DatasetType.LABELLED_AI,
    DatasetType.LABELLED_HUMAN,
)

aspect = "name"
if is_labelled:
    aspect = st.selectbox(
        "Aspect (map colors follow this column)",
        TRAIN_ASPECTS,
        help=(
            "Per place: more negative than positive labels → red/orange; "
            "more positive than negative → green; "
            "no positive/negative mentions (only neutral / not mentioned) → gray."
        ),
    )
    st.caption(
        "**Colors:** green — dominant positive; dark red / orange — dominant negative "
        "(orange when mixed); **gray** — no pos./neg. signal for this aspect; "
        "**amber** — equal pos. and neg. counts."
    )

lat = pd.to_numeric(prepared["latitude"], errors="coerce")
lon = pd.to_numeric(prepared["longitude"], errors="coerce")

ok = lat.notna() & lon.notna()


prepared_map = prepared.loc[ok].copy()

if prepared_map.empty:
    st.warning("No rows with valid coordinates.")
    st.stop()

view = pdk.ViewState(
    latitude=float(lat[ok].median()),
    longitude=float(lon[ok].median()),
    zoom=10,
)

deck = build_map(
    prepared_map,
    aspect=aspect,
    viz_type="Points",
    view_state=view,
    point_size=1.0,
)
st.pydeck_chart(deck, use_container_width=True)
