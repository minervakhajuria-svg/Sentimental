"""Sentimental: Streamlit entry point, navigation and the persistent disclaimer.

    streamlit run src/app/streamlit_app.py
"""

import sys
from pathlib import Path

# `streamlit run` puts this file's folder on sys.path, not src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import theme, views  # noqa: E402

st.set_page_config(page_title=views.APP_NAME, page_icon=":material/monitoring:", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

rankings_page = st.Page(views.rankings, title="Weekly rankings", icon=":material/leaderboard:",
                        default=True)
views.drilldown_page = st.Page(views.drilldown, title="Ticker drill-down",
                               icon=":material/query_stats:", url_path="ticker")
# Navigation is drawn by hand so the logo sits above the links.
nav = st.navigation([rankings_page, views.drilldown_page], position="hidden")
with st.sidebar:
    st.markdown(theme.logo(), unsafe_allow_html=True)
    for key, page in (("nav-rankings", rankings_page), ("nav-ticker", views.drilldown_page)):
        with st.container(key=key):
            st.page_link(page)
    st.divider()
# Green tint on the current page's link (Streamlit has no stable "active" hook).
current = "nav-ticker" if nav.url_path == views.drilldown_page.url_path else "nav-rankings"
st.markdown(theme.active_nav_css(current), unsafe_allow_html=True)

nav.run()

st.sidebar.markdown(f'<div class="sm-footer" style="margin-top:24px">{views.DISCLAIMER}</div>',
                    unsafe_allow_html=True)
st.divider()
st.markdown(f'<div class="sm-footer">{views.APP_NAME} · {views.DISCLAIMER}</div>',
            unsafe_allow_html=True)
