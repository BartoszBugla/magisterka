import pandas as pd
import streamlit as st
from config.global_config import ModelType
from predictions.predict_dataset import models, predict_dataset

from application.dataset_types import DatasetType
from application.dataset_upload_validation import validate
from application.label_dataset import (
    default_labelled_filename,
    list_labelable_entries,
    load_source_dataframe,
)
from application.notifications import notify
from application.results_repository import EntryMetadata, repository as repo

t = {
    "label_dataset_invalid_filename": "Nieprawidłowa nazwa pliku",
    "label_dataset_invalid_filename_details": "Nazwa pliku musi kończyć się na `.csv`.",
    "label_dataset_invalid_filename_path_separator": "Nazwa pliku nie może zawierać separatorów ścieżki.",
    "label_dataset_file_exists": "Plik już istnieje",
    "label_dataset_file_exists_details": lambda name: f"Plik `{name}` już istnieje w repozytorium.",
    "label_dataset_validation_failed": "Walidacja nie powiodła się",
    "label_dataset_validation_failed_details": lambda err: f"Walidacja nie powiodła się: {err}",
    "label_dataset_labelling_failed": "Etykietowanie nie powiodło się",
    "label_dataset_title": "Etykietowanie zbioru danych",
    "label_dataset_no_candidates_info": "Brak **oczyszczonych** lub **surowych** zbiorów danych opinii w repozytorium. ",
    "label_dataset_run_labelling": "Uruchom etykietowanie",
    "label_dataset_result": "Wynik etykietowania",
    "label_dataset_save_as": "Zapisz jako (nazwa pliku)",
    "label_dataset_save_to_repository": "Zapisz do repozytorium",
    "label_dataset_saved": lambda name: f"Zapisano `{name}` jako LABELLED_AI",
    "label_dataset_model": "Model",
    "label_dataset_source_csv": "Zbiór danych źródłowy",
    "label_dataset_num_rows": "Liczba wierszy do załadowania",
}

SS_SOURCE = "label_dataset_source_csv"
SS_RESULT = "label_dataset_result_df"
SS_MODEL = "label_dataset_model_type"

_MODEL_LABELS_PL: dict[ModelType, str] = {
    ModelType.FINE_TUNED_BERT: "BERT (fine-tuned)",
    ModelType.FINE_TUNED_DISTILBERT: "DistilBERT (fine-tuned)",
    ModelType.TFIDF_LSA: "TF-IDF + LSA + regresja logistyczna",
}


def _model_label(m: ModelType) -> str:
    return _MODEL_LABELS_PL.get(m, m.value)


def init_session_state():
    defaults = {
        SS_SOURCE: None,
        SS_RESULT: None,
        SS_MODEL: None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_session_state():
    st.session_state[SS_RESULT] = None
    st.session_state[SS_SOURCE] = None
    st.session_state[SS_MODEL] = None


def render_labelling_form(candidates: list, by_name: dict):
    model_type = st.selectbox(
        t["label_dataset_model"],
        options=list(models.keys()),
        format_func=_model_label,
    )

    source_csv = st.selectbox(
        t["label_dataset_source_csv"],
        options=list(by_name),
        format_func=lambda fn: f"{fn} — {by_name[fn].dataset_type.label_pl}",
    )

    max_rows = len(load_source_dataframe(repo, source_csv)) if source_csv else 1000
    num_rows = st.number_input(
        t["label_dataset_num_rows"],
        min_value=10,
        max_value=max_rows,
        value=min(500, max_rows),
        step=10,
    )

    return model_type, source_csv, num_rows


def run_labelling_process(source_csv: str, model_type, num_rows: int):
    df = load_source_dataframe(repo, source_csv).head(num_rows)
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total if total else 0.0)
        status.text(f"{done} / {total} wierszy")

    result = predict_dataset(df, model_type, on_progress=on_progress)
    progress.progress(1.0)
    status.text("Gotowe.")

    return result


def render_result_section(result_df: pd.DataFrame):
    st.divider()
    st.subheader(t["label_dataset_result"])

    if st.session_state[SS_MODEL] is not None:
        st.caption(f"Użyty model: **{_model_label(st.session_state[SS_MODEL])}**")

    st.dataframe(result_df.head(50), use_container_width=True)


def render_save_form(result_df: pd.DataFrame):
    src_name = st.session_state[SS_SOURCE] or ""
    default_name = (
        default_labelled_filename(src_name) if src_name else "dataset_labelled.csv"
    )

    out_name = st.text_input(t["label_dataset_save_as"], value=default_name)

    if st.button(t["label_dataset_save_to_repository"]):
        save_result(result_df, out_name, src_name)


def save_result(result_df: pd.DataFrame, out_name: str, src_name: str):
    name = (out_name or "").strip()

    if not name.endswith(".csv"):
        notify.error(
            t["label_dataset_invalid_filename"],
            details=t["label_dataset_invalid_filename_details"],
        )
        return

    if "/" in name or "\\" in name:
        notify.error(
            t["label_dataset_invalid_filename"],
            details=t["label_dataset_invalid_filename_path_separator"],
        )
        return

    if repo.exists(name):
        notify.error(
            t["label_dataset_file_exists"],
            details=t["label_dataset_file_exists_details"](name),
        )
        return

    raw = result_df.to_csv(index=False).encode("utf-8")
    err = validate(raw, DatasetType.LABELLED_AI)
    if err:
        notify.error(
            t["label_dataset_validation_failed"],
            details=t["label_dataset_validation_failed_details"](err),
        )
        return

    used_model = st.session_state[SS_MODEL]
    repo.save(
        name,
        raw,
        EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.LABELLED_AI,
            model_type=used_model,
            notes=f"Etykietowane z {src_name}",
        ),
    )
    notify.success(t["label_dataset_saved"](name))
    clear_session_state()
    st.rerun()


def main():
    st.title(t["label_dataset_title"])
    init_session_state()

    candidates = list_labelable_entries()
    if not candidates:
        st.info(t["label_dataset_no_candidates_info"])
        st.stop()

    by_name = {m.csv_filename: m for m in candidates}

    model_type, source_csv, num_rows = render_labelling_form(candidates, by_name)

    if st.button(t["label_dataset_run_labelling"], type="primary"):
        try:
            result = run_labelling_process(source_csv, model_type, num_rows)
            st.session_state[SS_SOURCE] = source_csv
            st.session_state[SS_RESULT] = result
            st.session_state[SS_MODEL] = model_type
            st.rerun()
        except Exception as e:
            notify.error(t["label_dataset_labelling_failed"], exception=e)

    result_df: pd.DataFrame | None = st.session_state[SS_RESULT]
    if result_df is None:
        st.stop()

    render_result_section(result_df)
    render_save_form(result_df)


main()
