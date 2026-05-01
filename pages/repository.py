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

t = {
    "upload_dialog_title": "Zuploaduj nowy zbiór danych",
    "upload_dialog_description": "Zuploaduj plik CSV aby dodać go do repozytorium.",
    "upload_dialog_dataset_type": "Typ zbioru danych",
    "upload_dialog_csv_file": "Plik CSV",
    "upload_dialog_notes": "Notatki (opcjonalne)",
    "upload_dialog_cancel": "Anuluj",
    "upload_dialog_save": "Zapisz zmiany",
    "upload_dialog_metadata_updated": "Metadane zaktualizowane",
    "upload_dialog_delete": "Usuń zbiór danych",
    "upload_dialog_delete_confirmation": "Czy na pewno chcesz usunąć **{csv_filename}**? Ta operacja nie może zostać cofnięta.",
    "upload_dialog_delete_confirmation_cancel": "Anuluj",
    "upload_dialog_delete_confirmation_delete": "Usuń",
    "upload_dialog_delete_confirmation_delete_button": "Usuń zbiór danych",
    "upload_dialog_file_already_exists": "Plik już istnieje",
    "upload_dialog_file_already_exists_details": lambda uploaded_name: f"Entry '{uploaded_name}' already exists. Delete it first or rename the file.",
    "upload_dialog_delete_confirmation_delete_button_text_confirm": "Usuń zbiór danych",
    "upload_dialog_delete_confirmation_delete_button_text_confirm_text": "Usuń zbiór danych",
    "upload_dialog_saved": lambda uploaded_name: f"Zapisano: {uploaded_name}",
    "repository_description": "Zarządzaj swoimi zbiorami danych — wyślij, edytuj metadane, lub usuń wpisy.",
    "repository_upload_button": "Zuploaduj",
    "repository_no_datasets_info": "Brak zbiorów danych w repozytorium. Kliknij **Wyślij** aby dodać swój pierwszy zbiór danych.",
    "repository_labelled_datasets": "Zbiory danych etykietowane",
    "repository_other_datasets": "Zbiory danych nieetykowane",
    "edit_dialog_title": "Edytuj metadane zbioru danych",
    "edit_dialog_description": "Edytuj metadane zbioru danych",
    "edit_dialog_current_dataset": "Aktualny zbiór danych",
    "edit_dialog_edit_metadata": "Edytuj metadane",
    "edit_dialog_dataset_type": "Typ zbioru danych",
    "edit_dialog_notes": "Notatki (opcjonalne)",
    "edit_dialog_cancel": "Anuluj",
    "edit_dialog_save": "Zapisz zmiany",
    "edit_dialog_metadata_updated": "Metadane zaktualizowane",
    "edit_dialog_delete": "Usuń zbiór danych",
    "edit_dialog_delete_confirmation": "Czy na pewno chcesz usunąć **{csv_filename}**? Ta operacja nie może zostać cofnięta.",
    "edit_dialog_delete_confirmation_cancel": "Anuluj",
    "edit_dialog_delete_confirmation_delete": "Usuń",
    "edit_dialog_delete_confirmation_delete_button": "Usuń zbiór danych",
    "edit_dialog_delete_confirmation_delete_button_text_confirm": "Usuń zbiór danych",
    "edit_dialog_delete_confirmation_delete_button_text_confirm_text": "Usuń zbiór danych",
    "edit_dialog_saved": lambda csv_filename: f"Zapisano: {csv_filename}",
    "delete_dialog_title": "Potwierdzenie usunięcia",
    "delete_dialog_description": "Potwierdzenie usunięcia",
    "delete_dialog_current_dataset": "Aktualny zbiór danych",
    "delete_dialog_edit_metadata": "Edytuj metadane",
    "delete_dialog_dataset_type": "Typ zbioru danych",
    "delete_dialog_notes": "Notatki (opcjonalne)",
    "delete_dialog_cancel": "Anuluj",
    "delete_dialog_save": "Zapisz zmiany",
    "delete_dialog_metadata_updated": "Metadane zaktualizowane",
    "delete_dialog_delete": "Usuń zbiór danych",
    "delete_dialog_delete_confirmation": lambda csv_filename: f"Czy na pewno chcesz usunąć **{csv_filename}**? Ta operacja nie może zostać cofnięta.",
    "delete_dialog_delete_confirmation_cancel": "Anuluj",
    "delete_dialog_delete_confirmation_delete": "Usuń",
    "delete_dialog_delete_confirmation_delete_button": "Usuń zbiór danych",
    "delete_dialog_delete_confirmation_delete_button_text_confirm": "Usuń zbiór danych",
    "delete_dialog_delete_confirmation_delete_button_text_confirm_text": "Usuń zbiór danych",
    "delete_dialog_saved": lambda csv_filename: f"Zapisano: {csv_filename}",
    "delete_dialog_deleted": lambda csv_filename: f"Usunięto: {csv_filename}",
    "delete_dialog_deleted_confirmation": "Zbiór danych został usunięty",
    "delete_dialog_deleted_confirmation_cancel": "Anuluj",
    "delete_dialog_deleted_confirmation_delete": "Usuń",
    "delete_dialog_deleted_confirmation_delete_button": "Usuń zbiór danych",
    "delete_dialog_deleted_confirmation_delete_button_text_confirm": "Usuń zbiór danych",
    "delete_dialog_deleted_confirmation_delete_button_text_confirm_text": "Usuń zbiór danych",
    "delete_dialog_deleted_confirmation_deleted": lambda csv_filename: f"Zbiór danych został usunięty: {csv_filename}",
}

# Metadata
st.title("Repozytorium")


@st.dialog("Upload new dataset", width="large")
def _upload_dialog() -> None:
    st.markdown(t["upload_dialog_description"])

    dataset_type = st.selectbox(
        t["upload_dialog_dataset_type"],
        options=[dt for dt in _SCHEMAS if _SCHEMAS[dt].uploadable],
        format_func=lambda dt: dt.label_pl,
        key="upload_type",
    )

    uploaded = st.file_uploader(
        t["upload_dialog_csv_file"], type=["csv"], key="upload_file"
    )
    notes = st.text_input(t["upload_dialog_notes"], key="upload_notes")

    col_cancel, col_submit = st.columns(2)

    with col_cancel:
        if st.button(t["upload_dialog_cancel"], use_container_width=True):
            st.rerun()

    with col_submit:
        submit_disabled = uploaded is None
        if st.button(
            t["upload_dialog_save"],
            use_container_width=True,
            disabled=submit_disabled,
            type="primary",
        ):
            raw = uploaded.getvalue()
            err = validate(raw, dataset_type)
            if err:
                notify.error(t["upload_dialog_validation_failed"], details=err)
            elif repo.exists(uploaded.name):
                notify.error(
                    t["upload_dialog_file_already_exists"],
                    details=t["upload_dialog_file_already_exists_details"](
                        uploaded.name
                    ),
                )
            else:
                meta = EntryMetadata(
                    csv_filename=uploaded.name,
                    dataset_type=dataset_type,
                    notes=notes,
                )
                repo.save(uploaded.name, raw, meta)
                invalidate_stats_cache(uploaded.name)
                notify.success(t["upload_dialog_saved"](uploaded.name))
                st.rerun()


@st.dialog(t["edit_dialog_title"], width="large")
def _edit_dialog() -> None:
    csv_filename = st.session_state.get("_edit_csv")
    if not csv_filename or not repo.exists(csv_filename):
        notify.warning(t["edit_dialog_dataset_not_found"])
        return

    entry = repo.get_metadata(csv_filename)
    stats = get_dataset_stats(csv_filename)

    st.markdown(t["edit_dialog_current_dataset"])
    with st.container(border=True):
        render_dataset_card_info(entry, stats)

    st.divider()
    st.markdown(f'#### {t["edit_dialog_edit_metadata"]} ')

    new_type = st.selectbox(
        t["edit_dialog_dataset_type"],
        options=[dt for dt in _SCHEMAS if _SCHEMAS[dt].uploadable],
        format_func=lambda dt: dt.label_pl,
        index=list(_SCHEMAS.keys()).index(entry.dataset_type),
        key="edit_type",
    )

    new_notes = st.text_input(
        t["edit_dialog_notes"], value=entry.notes, key="edit_notes"
    )

    col_cancel, col_save = st.columns(2)

    with col_cancel:
        if st.button(t["edit_dialog_cancel"], use_container_width=True):
            st.session_state.pop("_edit_csv", None)
            st.rerun()

    with col_save:
        if st.button(t["edit_dialog_save"], use_container_width=True, type="primary"):
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
            notify.success(t["edit_dialog_metadata_updated"])
            st.rerun()


@st.dialog("Potwierdzenie usunięcia")
def _delete_dialog() -> None:
    csv_filename = st.session_state.get("_delete_csv")
    if not csv_filename:
        notify.warning(t["delete_dialog_dataset_not_found"])
        return

    st.warning(t["delete_dialog_delete_confirmation"](csv_filename))

    col_cancel, col_confirm = st.columns(2)

    with col_cancel:
        if st.button(t["delete_dialog_cancel"], use_container_width=True):
            st.session_state.pop("_delete_csv", None)
            st.rerun()

    with col_confirm:
        if st.button(
            t["delete_dialog_delete"], use_container_width=True, type="primary"
        ):
            repo.delete(csv_filename)
            invalidate_stats_cache(csv_filename)
            st.session_state.pop("_delete_csv", None)
            st.rerun()


def _render_repository_gallery(entries: list[EntryMetadata]) -> None:
    labelled, other = group_entries_by_type(entries)

    if labelled:
        st.markdown(f'#### {t["repository_labelled_datasets"]}')
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
        st.markdown(f'#### {t["repository_other_datasets"]} ')
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


col_title, col_upload = st.columns([4, 1])
with col_title:
    st.caption(t["repository_description"])
with col_upload:
    if st.button(
        t["repository_upload_button"],
        use_container_width=True,
        type="primary",
        icon=":material/add:",
    ):
        _upload_dialog()

st.divider()

entries = repo.list_entries()

if not entries:
    st.info(t["repository_no_datasets_info"])
    st.stop()

_render_repository_gallery(entries)
