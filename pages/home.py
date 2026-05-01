from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from application.dataset_cards import (
    get_dataset_stats,
    group_entries_by_type,
    render_dataset_card_readonly,
)
from application.dataset_types import DatasetType
from application.map_components import build_map
from application.notifications import notify
from application.place_details import (
    compute_place_keys,
    find_metadata_for_place,
    list_metadata_files,
    load_metadata,
    render_place_dialog,
    reviews_for_place,
)
from application.results_repository import EntryMetadata, repository as repo
from config.global_config import TRAIN_ASPECTS


def _render_dataset_gallery(entries: list[EntryMetadata]) -> str | None:
    st.subheader("Wybór zbioru danych")

    labelled, other = group_entries_by_type(entries)
    selected: str | None = None

    if labelled:
        st.markdown("##### Zbiory danych etykietowane")
        cols = st.columns(2)
        for i, entry in enumerate(labelled):
            with cols[i % 2]:
                stats = get_dataset_stats(entry.csv_filename)
                if render_dataset_card_readonly(entry, stats):
                    selected = entry.csv_filename

    if other:
        st.markdown("##### Zbiory danych nieetykowane")
        cols = st.columns(2)
        for i, entry in enumerate(other):
            with cols[i % 2]:
                stats = get_dataset_stats(entry.csv_filename)
                if render_dataset_card_readonly(entry, stats):
                    selected = entry.csv_filename

    return selected


def _available_aspects(df: pd.DataFrame) -> list[str]:
    return [a for a in TRAIN_ASPECTS if a in df.columns]


def _load_dataset(csv_filename: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(repo.get_csv_path(csv_filename))
    except Exception as e:
        notify.error("Nie udało się załadować CSV", exception=e)
        return None


def _validate_coords(df: pd.DataFrame) -> pd.DataFrame | None:
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            notify.error(f"Brak wymaganego kolumny: `{col}`")
            return None
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = lat.notna() & lon.notna()
    valid = df.loc[ok].copy()
    if valid.empty:
        notify.warning("Brak wierszy z poprawnymi współrzędnymi")
        return None
    return valid


@st.dialog("Szczegóły miejsca", width="large")
def _open_place_dialog() -> None:
    place_info = st.session_state.get("_sel_place")
    reviews = st.session_state.get("_sel_reviews")
    metadata = st.session_state.get("_sel_metadata")
    aspect = st.session_state.get("_sel_aspect", TRAIN_ASPECTS[0])
    aspects = st.session_state.get("_sel_aspects", TRAIN_ASPECTS)

    if place_info is None or reviews is None:
        notify.warning("Brak miejsca wybranego")
        return

    render_place_dialog(place_info, reviews, metadata, aspect, aspects)


def _handle_selection(
    event,
    df_raw: pd.DataFrame,
    place_keys: pd.Series,
    metadata_df: pd.DataFrame,
    aspect: str,
    aspects: list[str],
    layer_id: str = "absa-points",
) -> None:
    if event is None or event.selection is None:
        return

    objects = event.selection.get("objects", {})
    layer_hits = objects.get(layer_id, [])

    if not layer_hits:
        st.session_state.pop("_sel_opened_key", None)
        return

    clicked = layer_hits[0]
    place_key = clicked.get("_place_key")
    if not place_key:
        return

    if place_key == st.session_state.get("_sel_opened_key"):
        return
    st.session_state["_sel_opened_key"] = place_key

    reviews = reviews_for_place(df_raw, place_keys, place_key)
    meta = find_metadata_for_place(
        metadata_df,
        gmap_id=clicked.get("gmap_id_display"),
        place_name=clicked.get("place_name"),
    )

    st.session_state["_sel_place"] = clicked
    st.session_state["_sel_reviews"] = reviews
    st.session_state["_sel_metadata"] = meta
    st.session_state["_sel_aspect"] = aspect
    st.session_state["_sel_aspects"] = aspects

    _open_place_dialog()


def _render_map_view(csv_filename: str, entry_meta: EntryMetadata) -> None:
    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.subheader(f"Mapa: {csv_filename}")
    with col_back:
        if st.button("← Powrót", use_container_width=True):
            st.session_state.pop("selected_dataset", None)
            st.rerun()

    df_raw = _load_dataset(csv_filename)
    if df_raw is None or df_raw.empty:
        notify.warning("Ten zbiór danych nie ma wierszy")
        return

    prepared = _validate_coords(df_raw)
    if prepared is None:
        return

    metadata_df = pd.DataFrame()
    # meta_files = list_metadata_files()
    # if meta_files:
    #     with st.expander("Metadata file (optional — enriches place details)"):
    #         meta_choice = st.selectbox(
    #             "Metadata CSV",
    #             options=["None"] + [p.name for p in meta_files],
    #             key="meta_csv_select",
    #         )
    #         if meta_choice != "None":
    #             matched = [p for p in meta_files if p.name == meta_choice]
    #             if matched:
    #                 metadata_df = load_metadata(str(matched[0]))

    is_labelled = entry_meta.dataset_type in (
        DatasetType.LABELLED_AI,
        DatasetType.LABELLED_HUMAN,
    )

    aspect = "name"
    aspects = _available_aspects(prepared) or TRAIN_ASPECTS

    if is_labelled:
        aspect = st.selectbox(
            "Wybór aspektu",
            aspects,
        )

    place_keys = compute_place_keys(prepared)

    lat = pd.to_numeric(prepared["latitude"], errors="coerce")
    lon = pd.to_numeric(prepared["longitude"], errors="coerce")

    view = pdk.ViewState(
        latitude=float(lat.median()),
        longitude=float(lon.median()),
        zoom=10,
    )

    deck, scatter_df = build_map(
        prepared,
        aspect=aspect,
        view_state=view,
    )

    event = st.pydeck_chart(
        deck,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-object",
        key="map_chart",
    )

    _handle_selection(event, prepared, place_keys, metadata_df, aspect, aspects)


st.title("Eksplorator mapy aspektowanej analizy sentymentu (ABSA)")

entries = repo.list_entries()
if not entries:
    st.info(
        "Brak zbiorów danych w repozytorium. Wyślij pliki CSV na stronie **Repozytorium**."
    )
    st.stop()

by_name = {e.csv_filename: e for e in entries}

selected = st.session_state.get("selected_dataset")

if selected is None:
    clicked = _render_dataset_gallery(entries)
    if clicked:
        st.session_state["selected_dataset"] = clicked
        st.rerun()
else:
    if selected not in by_name:
        st.session_state.pop("selected_dataset", None)
        st.rerun()
    _render_map_view(selected, by_name[selected])
