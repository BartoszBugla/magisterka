import pandas as pd
import streamlit as st

from application.dataset_types import DatasetType
from application.dataset_upload_validation import validate
from application.label_dataset import (
    MODEL_TYPE_LABELS,
    default_labelled_filename,
    list_labelable_entries,
    load_source_dataframe,
    run_labelling,
)
from application.results_repository import EntryMetadata, repository as repo

SS_SOURCE = "label_dataset_source_csv"
SS_RESULT = "label_dataset_result_df"
SS_MODEL = "label_dataset_model_type"

st.title("Label dataset")

for key, default in (
    (SS_SOURCE, None),
    (SS_RESULT, None),
    (SS_MODEL, None),
):
    if key not in st.session_state:
        st.session_state[key] = default

candidates = list_labelable_entries()
if not candidates:
    st.info(
        "No **cleaned** or **raw reviews** datasets in the repository. "
        "Upload one on the **Repository** page."
    )
    st.stop()

by_name = {m.csv_filename: m for m in candidates}

model_type = st.selectbox(
    "Model",
    options=list(MODEL_TYPE_LABELS),
    format_func=lambda m: MODEL_TYPE_LABELS[m],
)

source_csv = st.selectbox(
    "Source dataset",
    options=list(by_name),
    format_func=lambda fn: f"{fn} — {by_name[fn].dataset_type.label_pl}",
)

if st.button("Run labelling", type="primary"):
    try:
        df = load_source_dataframe(repo, source_csv)
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total if total else 0.0)
            status.text(f"{done} / {total} rows")

        result = run_labelling(df, model_type, on_progress=on_progress)
        progress.progress(1.0)
        status.text("Done.")
        st.session_state[SS_SOURCE] = source_csv
        st.session_state[SS_RESULT] = result
        st.session_state[SS_MODEL] = model_type
        st.rerun()
    except Exception as e:
        st.exception(e)

result_df: pd.DataFrame | None = st.session_state[SS_RESULT]
if result_df is None:
    st.stop()

st.divider()
st.subheader("Result")
if st.session_state[SS_MODEL] is not None:
    st.caption(f"Model used: **{MODEL_TYPE_LABELS[st.session_state[SS_MODEL]]}**")

st.dataframe(result_df.head(50), use_container_width=True)

src_name = st.session_state[SS_SOURCE] or ""
out_name = st.text_input(
    "Save as (filename)",
    value=default_labelled_filename(src_name) if src_name else "dataset_labelled.csv",
)

if st.button("Save to repository"):
    name = (out_name or "").strip()
    if not name.endswith(".csv"):
        st.error("Filename must end with `.csv`.")
    elif "/" in name or "\\" in name:
        st.error("Filename cannot contain path separators.")
    elif repo.exists(name):
        st.error(f"`{name}` already exists.")
    else:
        raw = result_df.to_csv(index=False).encode("utf-8")
        err = validate(raw, DatasetType.LABELLED_AI)
        if err:
            st.error(err)
        else:
            used = st.session_state[SS_MODEL]
            repo.save(
                name,
                raw,
                EntryMetadata(
                    csv_filename=name,
                    dataset_type=DatasetType.LABELLED_AI,
                    model_type=used,
                    notes=f"Labelled from {src_name}",
                ),
            )
            st.success(f"Saved `{name}` as **LABELLED_AI**.")
            st.session_state[SS_RESULT] = None
            st.session_state[SS_SOURCE] = None
            st.session_state[SS_MODEL] = None
            st.rerun()
