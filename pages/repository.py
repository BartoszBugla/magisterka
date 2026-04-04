import pandas as pd
import streamlit as st

from application.dataset_types import _SCHEMAS
from application.dataset_upload_validation import validate
from application.results_repository import EntryMetadata, repository as repo


st.title("Repository")

# --- Upload Section ---
st.subheader("Upload CSV")

dataset_type = st.selectbox(
    "Dataset type",
    options=[dt for dt in _SCHEMAS if _SCHEMAS[dt].uploadable],
    format_func=lambda dt: dt.label_pl,
)

uploaded = st.file_uploader("CSV file", type=["csv"])
notes = st.text_input("Notes (optional)")

if st.button("Upload", disabled=uploaded is None):
    raw = uploaded.getvalue()
    err = validate(raw, dataset_type)
    if err:
        st.error(err)
    elif repo.exists(uploaded.name):
        st.error(f"Entry '{uploaded.name}' already exists.")
    else:
        meta = EntryMetadata(
            csv_filename=uploaded.name,
            dataset_type=dataset_type,
            notes=notes,
        )
        repo.save(uploaded.name, raw, meta)
        st.success(f"Saved: {uploaded.name}")
        st.rerun()

# --- Entries List ---
st.divider()
st.subheader("Entries")

entries = repo.list_entries()
if not entries:
    st.info("No entries yet.")
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "File": m.csv_filename,
                "Type": m.dataset_type.label_pl,
                "Created": m.created_at[:19],
                "Notes": m.notes,
            }
            for m in entries
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

for meta in entries:
    with st.expander(meta.csv_filename):
        st.markdown(f"**Type:** {meta.dataset_type.label_pl}")
        st.markdown(f"**Created:** {meta.created_at}")
        if meta.notes:
            st.markdown(f"**Notes:** {meta.notes}")

        csv_path = repo.get_csv_path(meta.csv_filename)
        st.download_button(
            "Download CSV",
            data=csv_path.read_bytes(),
            file_name=meta.csv_filename,
            mime="text/csv",
            key=f"dl_{meta.csv_filename}",
        )

        if st.button("Delete", key=f"del_{meta.csv_filename}", type="primary"):
            repo.delete(meta.csv_filename)
            st.rerun()
