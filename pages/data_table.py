import json

import pandas as pd
import streamlit as st

from application.notifications import notify
from application.results_repository import repository as repo

st.title("Data table")

entries = repo.list_entries()
if not entries:
    st.info(
        "No datasets in the repository. Upload CSV files on the **Repository** page."
    )
    st.stop()

by_name = {e.csv_filename: e for e in entries}
selected = st.selectbox(
    "Dataset",
    options=list(by_name),
    format_func=lambda fn: f"{by_name[fn].dataset_type.label_pl} — {fn}",
)

meta = repo.get_metadata(selected)

st.subheader("Metadata")
st.json(json.loads(meta.to_json()))

st.subheader("CSV preview")
max_rows = st.number_input(
    "Max rows to load", min_value=10, max_value=100_000, value=500, step=10
)
try:
    df = pd.read_csv(repo.get_csv_path(selected), nrows=max_rows)
    st.caption(f"Loaded {len(df)} row(s); preview limit is {max_rows}.")
    st.dataframe(df, use_container_width=True, height=480)
except Exception as e:
    notify.error("Could not read CSV", exception=e)
