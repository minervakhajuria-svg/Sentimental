"""Streamlit entry point: page navigation and the persistent disclaimer.

    streamlit run src/app/streamlit_app.py
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's folder on sys.path, not src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import views  # noqa: E402

st.set_page_config(page_title="Sentiment screen", layout="wide")
rankings_page = st.Page(views.rankings, title="Weekly rankings", default=True)
views.drilldown_page = st.Page(views.drilldown, title="Ticker drill-down", url_path="ticker")
nav = st.navigation([rankings_page, views.drilldown_page])
st.sidebar.caption(views.DISCLAIMER)
nav.run()
st.divider()
st.caption(views.DISCLAIMER)
