import pandas as pd
import streamlit as st
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

SS_SOURCE = "label_dataset_source_csv"
SS_RESULT = "label_dataset_result_df"
SS_MODEL = "label_dataset_model_type"


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
    """Render the labelling configuration form and return selected values."""
    model_type = st.selectbox(
        "Model",
        options=list(models.keys()),
        # format_func=lambda m: models.keys()[m],
    )

    source_csv = st.selectbox(
        "Source dataset",
        options=list(by_name),
        format_func=lambda fn: f"{fn} — {by_name[fn].dataset_type.label_pl}",
    )

    max_rows = len(load_source_dataframe(repo, source_csv)) if source_csv else 1000
    num_rows = st.number_input(
        "Number of rows to load",
        min_value=10,
        max_value=max_rows,
        value=min(500, max_rows),
        step=10,
    )

    return model_type, source_csv, num_rows


def run_labelling_process(source_csv: str, model_type, num_rows: int):
    """Execute the labelling process with progress tracking."""
    df = load_source_dataframe(repo, source_csv).head(num_rows)
    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total if total else 0.0)
        status.text(f"{done} / {total} rows")

    result = predict_dataset(df, model_type, on_progress=on_progress)
    progress.progress(1.0)
    status.text("Done.")

    return result


def render_result_section(result_df: pd.DataFrame):
    """Display labelling results and model info."""
    st.divider()
    st.subheader("Result")

    if st.session_state[SS_MODEL] is not None:
        st.caption(f"Model used: **{models[st.session_state[SS_MODEL]]}**")

    st.dataframe(result_df.head(50), use_container_width=True)


def render_save_form(result_df: pd.DataFrame):
    """Render the save-to-repository form."""
    src_name = st.session_state[SS_SOURCE] or ""
    default_name = (
        default_labelled_filename(src_name) if src_name else "dataset_labelled.csv"
    )

    out_name = st.text_input("Save as (filename)", value=default_name)

    if st.button("Save to repository"):
        save_result(result_df, out_name, src_name)


def save_result(result_df: pd.DataFrame, out_name: str, src_name: str):
    """Validate and save the labelled dataset to repository."""
    name = (out_name or "").strip()

    if not name.endswith(".csv"):
        notify.error("Invalid filename", details="Filename must end with `.csv`.")
        return

    if "/" in name or "\\" in name:
        notify.error(
            "Invalid filename", details="Filename cannot contain path separators."
        )
        return

    if repo.exists(name):
        notify.error(
            "File exists", details=f"`{name}` already exists in the repository."
        )
        return

    raw = result_df.to_csv(index=False).encode("utf-8")
    err = validate(raw, DatasetType.LABELLED_AI)
    if err:
        notify.error("Validation failed", details=err)
        return

    used_model = st.session_state[SS_MODEL]
    repo.save(
        name,
        raw,
        EntryMetadata(
            csv_filename=name,
            dataset_type=DatasetType.LABELLED_AI,
            model_type=used_model,
            notes=f"Labelled from {src_name}",
        ),
    )
    notify.success(f"Saved `{name}` as LABELLED_AI")
    clear_session_state()
    st.rerun()


def main():
    st.title("Label dataset")
    init_session_state()

    candidates = list_labelable_entries()
    if not candidates:
        st.info(
            "No **cleaned** or **raw reviews** datasets in the repository. "
            "Upload one on the **Repository** page."
        )
        st.stop()

    by_name = {m.csv_filename: m for m in candidates}

    model_type, source_csv, num_rows = render_labelling_form(candidates, by_name)

    if st.button("Run labelling", type="primary"):
        try:
            result = run_labelling_process(source_csv, model_type, num_rows)
            st.session_state[SS_SOURCE] = source_csv
            st.session_state[SS_RESULT] = result
            st.session_state[SS_MODEL] = model_type
            st.rerun()
        except Exception as e:
            notify.error("Labelling failed", exception=e)

    result_df: pd.DataFrame | None = st.session_state[SS_RESULT]
    if result_df is None:
        st.stop()

    render_result_section(result_df)
    render_save_form(result_df)


main()
