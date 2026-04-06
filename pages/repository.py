from __future__ import annotations

import streamlit as st

from application.dataset_cards import (
    get_dataset_stats,
    group_entries_by_type,
    invalidate_stats_cache,
    render_dataset_card_editable,
    render_dataset_card_info,
)
from application.dataset_types import _SCHEMAS
from application.dataset_upload_validation import validate
from application.notifications import notify
from application.results_repository import EntryMetadata, repository as repo


@st.dialog("Upload new dataset", width="large")
def _upload_dialog() -> None:
    """Modal form for uploading a new dataset."""
    st.markdown("Upload a CSV file to add it to the repository.")

    dataset_type = st.selectbox(
        "Dataset type",
        options=[dt for dt in _SCHEMAS if _SCHEMAS[dt].uploadable],
        format_func=lambda dt: dt.label_pl,
        key="upload_type",
    )

    uploaded = st.file_uploader("CSV file", type=["csv"], key="upload_file")
    notes = st.text_input("Notes (optional)", key="upload_notes")

    col_cancel, col_submit = st.columns(2)

    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

    with col_submit:
        submit_disabled = uploaded is None
        if st.button(
            "Upload", use_container_width=True, disabled=submit_disabled, type="primary"
        ):
            raw = uploaded.getvalue()
            err = validate(raw, dataset_type)
            if err:
                notify.error("Validation failed", details=err)
            elif repo.exists(uploaded.name):
                notify.error(
                    "File already exists",
                    details=f"Entry '{uploaded.name}' already exists. Delete it first or rename the file.",
                )
            else:
                meta = EntryMetadata(
                    csv_filename=uploaded.name,
                    dataset_type=dataset_type,
                    notes=notes,
                )
                repo.save(uploaded.name, raw, meta)
                invalidate_stats_cache(uploaded.name)
                notify.success(f"Saved: {uploaded.name}")
                st.rerun()


@st.dialog("Edit dataset metadata", width="large")
def _edit_dialog() -> None:
    """Modal form for editing dataset metadata."""
    csv_filename = st.session_state.get("_edit_csv")
    if not csv_filename or not repo.exists(csv_filename):
        notify.warning("Dataset not found")
        return

    entry = repo.get_metadata(csv_filename)
    stats = get_dataset_stats(csv_filename)

    st.markdown("##### Current dataset")
    with st.container(border=True):
        render_dataset_card_info(entry, stats)

    st.divider()
    st.markdown("##### Edit metadata")

    new_type = st.selectbox(
        "Dataset type",
        options=[dt for dt in _SCHEMAS if _SCHEMAS[dt].uploadable],
        format_func=lambda dt: dt.label_pl,
        index=list(_SCHEMAS.keys()).index(entry.dataset_type),
        key="edit_type",
    )

    new_notes = st.text_input("Notes", value=entry.notes, key="edit_notes")

    col_cancel, col_save = st.columns(2)

    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("_edit_csv", None)
            st.rerun()

    with col_save:
        if st.button("Save changes", use_container_width=True, type="primary"):
            csv_path = repo.get_csv_path(csv_filename)
            csv_bytes = csv_path.read_bytes()

            updated_meta = EntryMetadata(
                csv_filename=csv_filename,
                dataset_type=new_type,
                created_at=entry.created_at,
                notes=new_notes,
                model_type=entry.model_type,
                extra=entry.extra,
            )

            repo.save(csv_filename, csv_bytes, updated_meta)
            invalidate_stats_cache(csv_filename)
            st.session_state.pop("_edit_csv", None)
            notify.success("Metadata updated")
            st.rerun()


# ---------------------------------------------------------------------------
# Delete confirmation dialog
# ---------------------------------------------------------------------------


@st.dialog("Confirm deletion")
def _delete_dialog() -> None:
    """Confirmation modal for deleting a dataset."""
    csv_filename = st.session_state.get("_delete_csv")
    if not csv_filename:
        notify.warning("No dataset selected")
        return

    st.warning(
        f"Are you sure you want to delete **{csv_filename}**? This cannot be undone."
    )

    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("_delete_csv", None)
            st.rerun()

    with col_confirm:
        if st.button("Delete", use_container_width=True, type="primary"):
            repo.delete(csv_filename)
            invalidate_stats_cache(csv_filename)
            st.session_state.pop("_delete_csv", None)
            st.rerun()


def _render_repository_gallery(entries: list[EntryMetadata]) -> None:
    """Render all datasets as editable cards in a 2-column grid."""
    labelled, other = group_entries_by_type(entries)

    if labelled:
        st.markdown("##### Labelled datasets")
        cols = st.columns(2)
        for i, entry in enumerate(labelled):
            with cols[i % 2]:
                stats = get_dataset_stats(entry.csv_filename)
                edit, _, delete = render_dataset_card_editable(entry, stats)
                if edit:
                    st.session_state["_edit_csv"] = entry.csv_filename
                    _edit_dialog()
                if delete:
                    st.session_state["_delete_csv"] = entry.csv_filename
                    _delete_dialog()

    if other:
        st.markdown("##### Other datasets")
        cols = st.columns(2)
        for i, entry in enumerate(other):
            with cols[i % 2]:
                stats = get_dataset_stats(entry.csv_filename)
                edit, _, delete = render_dataset_card_editable(entry, stats)
                if edit:
                    st.session_state["_edit_csv"] = entry.csv_filename
                    _edit_dialog()
                if delete:
                    st.session_state["_delete_csv"] = entry.csv_filename
                    _delete_dialog()


st.title("Repository")

col_title, col_upload = st.columns([4, 1])
with col_title:
    st.caption("Manage your datasets — upload, edit metadata, or delete entries.")
with col_upload:
    if st.button(
        "Upload",
        use_container_width=True,
        type="primary",
        icon=":material/add:",
    ):
        _upload_dialog()

st.divider()

entries = repo.list_entries()

if not entries:
    st.info(
        "No datasets in the repository yet. Click **Upload** to add your first dataset."
    )
    st.stop()

_render_repository_gallery(entries)
