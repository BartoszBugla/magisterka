from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Projekt Magisterski - aspektowana analiza sentymentu - wyniki na mapie",
    page_icon=str(_ROOT / "statics" / "icons" / "app-favicon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(
        str(_ROOT / "pages" / "home.py"), title="Strona główna", icon=":material/map:"
    ),
    st.Page(
        str(_ROOT / "pages" / "data_table.py"),
        title="Tabela danych",
        icon=":material/table_chart:",
    ),
    st.Page(
        str(_ROOT / "pages" / "label_dataset.py"),
        title="Etykietowanie danych",
        icon=":material/label:",
    ),
    st.Page(
        str(_ROOT / "pages" / "repository.py"),
        title="Repozytorium",
        icon=":material/inventory_2:",
    ),
]

st.navigation(pages).run()
