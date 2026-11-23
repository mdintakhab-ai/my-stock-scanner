# =============================================================================
# FALCON NEXT-DAY BEHAVIOUR PREDICTOR V5.5 — COMPACT SINGLE-TABLE EDITION
# GitHub + Streamlit Community Cloud
# Calculation engine: falcon_core_v55.py (kept unchanged)
# =============================================================================

from __future__ import annotations

import contextlib
import io
import runpy
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
CORE_FILE = APP_DIR / "falcon_core_v55.py"

st.set_page_config(
    page_title="FALCON V5.5 | Next-Day Predictor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# One compact table is used for every screen size. The mobile rules wrap cells
# and shrink typography so the complete table stays within the viewport.
_CSS = "\n".join([
    "<style>",
    ".stApp{background:#f4f7fb;color:#172334}",
    "[data-testid='stHeader']{background:rgba(244,247,251,.96)}",
    ".block-container{max-width:100%;padding:.65rem .8rem 1rem .8rem}",
    ".falcon-hero{background:linear-gradient(125deg,#102544,#1e4777);color:#fff;border-radius:13px;padding:12px 15px;margin:0 0 8px 0}",
    ".falcon-kicker{font-size:10px;letter-spacing:1.5px;opacity:.82;font-weight:800}",
    ".falcon-title{font-size:clamp(19px,3vw,29px);font-weight:900;line-height:1.15;margin-top:4px}",
    ".falcon-subtitle{font-size:10px;opacity:.86;margin-top:4px}",
    ".falcon-note{background:#fff8e8;border:1px solid #f1dfb8;color:#73551b;border-radius:9px;padding:7px 9px;font-size:10px;margin:7px 0}",
    ".falcon-table-wrap{width:100%;max-width:100%;overflow:visible;margin:0;padding:0}",
    "table.falcon-single-table{width:100%!important;max-width:100%!important;table-layout:fixed!important;border-collapse:collapse!important;border-spacing:0!important}",
    "table.falcon-single-table th{white-space:normal!important;overflow-wrap:anywhere!important;word-break:break-word!important;padding:4px 2px!important;font-size:10px!important;line-height:1.08!important}",
    "table.falcon-single-table td{white-space:normal!important;overflow-wrap:anywhere!important;word-break:break-word!important;padding:4px 2px!important;font-size:10px!important;line-height:1.12!important}",
    "table.falcon-single-table tbody tr:nth-child(even) td{background-color:#f5f8fc}",
    "table.falcon-single-table tbody tr:hover td{border-top:1px solid #8094b0!important;border-bottom:1px solid #8094b0!important}",
    "div[data-testid='stDownloadButton'] button{width:100%;border-radius:9px}",
    "@media(max-width:900px){.block-container{padding:.45rem .35rem .8rem}.falcon-hero{padding:10px}.falcon-title{font-size:18px}.falcon-subtitle{font-size:9px}table.falcon-single-table th{font-size:7px!important;padding:2px 1px!important}table.falcon-single-table td{font-size:7px!important;padding:2px 1px!important;line-height:1.05!important}}",
    "@media(max-width:480px){.block-container{padding:.25rem .15rem .6rem}.falcon-title{font-size:16px}table.falcon-single-table th{font-size:5.5px!important;padding:1px 0!important;line-height:1.0!important}table.falcon-single-table td{font-size:5.5px!important;padding:1px 0!important;line-height:1.0!important}table.falcon-single-table tbody td.col0{font-size:6px!important;font-weight:800!important}}",
    "</style>",
])
st.markdown(_CSS, unsafe_allow_html=True)

_HERO_HTML = (
    '<div class="falcon-hero">'
    '<div class="falcon-kicker">FALCON TRADING INTELLIGENCE</div>'
    '<div class="falcon-title">🦅 NEXT-DAY BEHAVIOUR PREDICTOR</div>'
    '<div class="falcon-subtitle">V5.5 · SINGLE CONSOLIDATED TABLE · NSE EQUITY</div>'
    '</div>'
)
st.markdown(_HERO_HTML, unsafe_allow_html=True)
st.markdown(
    '<div class="falcon-note"><b>Risk notice:</b> Behaviour shares are heuristic score distributions, not calibrated probabilities. '
    'No guaranteed accuracy is claimed. Numeric price/level values are displayed to exactly 2 decimal places; quantity is shown as a whole number.</div>',
    unsafe_allow_html=True,
)

if not CORE_FILE.is_file():
    st.error("falcon_core_v55.py is missing. Keep app.py, falcon_core_v55.py and requirements.txt in the same GitHub repository folder.")
    st.stop()

button_col, info_col = st.columns([1, 2])
with button_col:
    run_clicked = st.button("🚀 RUN FALCON SCANNER", type="primary", use_container_width=True)
with info_col:
    if st.button("↻ CLEAR PREVIOUS RESULT", use_container_width=True):
        st.session_state.pop("falcon_bundle", None)
        st.session_state.pop("falcon_run_error", None)
        st.rerun()

if run_clicked:
    try:
        captured = io.StringIO()
        with st.spinner("FALCON is scanning the NSE universe and calculating multi-timeframe profiles. Please wait..."):
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                namespace = runpy.run_path(str(CORE_FILE), run_name="__falcon_v55_scan__")

        results = namespace.get("FALCON_RESULTS", pd.DataFrame())
        dashboard = namespace.get("dashboard", pd.DataFrame())
        rendered_html = namespace.get("rendered_html", "")

        st.session_state["falcon_bundle"] = {
            "results": results if isinstance(results, pd.DataFrame) else pd.DataFrame(),
            "dashboard": dashboard if isinstance(dashboard, pd.DataFrame) else pd.DataFrame(),
            "rendered_html": rendered_html if isinstance(rendered_html, str) else "",
            "errors": namespace.get("FALCON_ERRORS", []),
            "selected_count": len(namespace.get("selected_rows", [])),
            "target_count": namespace.get("TARGET_FINAL_STOCKS", 7),
            "run_time": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
            "console": captured.getvalue(),
        }
        st.session_state.pop("falcon_run_error", None)
        st.success("Scanner execution completed.")
    except Exception as exc:
        st.session_state["falcon_run_error"] = f"{type(exc).__name__}: {exc}"
        st.exception(exc)

if "falcon_run_error" in st.session_state:
    st.error(st.session_state.pop("falcon_run_error"))

bundle = st.session_state.get("falcon_bundle")
if bundle:
    results = bundle["results"]
    dashboard = bundle["dashboard"]
    st.caption(
        f"Last run: {bundle['run_time']}  |  Selected: {bundle['selected_count']} / {bundle['target_count']}"
    )

    if results.empty or dashboard.empty:
        st.markdown(
            '<div style="background:#fff;border:1px dashed #c7d1df;padding:18px;border-radius:10px;text-align:center;color:#59687b">'
            '<b>NO QUALIFIED STOCKS</b><br/>No stock passed the scanner\'s mandatory filters in this run. '
            'This is not automatically a software failure.</div>',
            unsafe_allow_html=True,
        )
    else:
        # rendered_html is generated by the original V5.5 core. It contains
        # the original U/D colours, nearest-pivot yellow glow, lower-rejection
        # green glow and upper-rejection red glow. Do not rebuild/replace it
        # with st.dataframe: that would discard the highlighting logic.
        table_html = bundle.get("rendered_html", "")
        if table_html.strip():
            st.markdown('<div class="falcon-table-wrap">' + table_html + '</div>', unsafe_allow_html=True)
        else:
            # Defensive fallback if the core ever stops returning styled HTML.
            fallback = dashboard.style.hide(axis="index").format(precision=2, na_rep="—").to_html()
            st.markdown('<div class="falcon-table-wrap">' + fallback + '</div>', unsafe_allow_html=True)

        csv_bytes = results.to_csv(index=False, float_format="%.2f").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download full scanner data (CSV)",
            data=csv_bytes,
            file_name="FALCON_V5_5_FULL_RESULTS.csv",
            mime="text/csv",
            use_container_width=True,
        )

    errors = bundle.get("errors", [])
    if errors:
        with st.expander(f"Data / analysis warnings ({len(errors)})"):
            for message in errors[:100]:
                st.write(f"• {message}")
            if len(errors) > 100:
                st.caption("Showing first 100 warnings.")

    if bundle.get("console"):
        with st.expander("Scanner execution log"):
            st.code(bundle["console"][-12000:] or "No console output.")
else:
    st.markdown(
        '<div style="background:#fff;border:1px dashed #c7d1df;padding:18px;border-radius:10px;text-align:center;color:#59687b">'
        '<b>Scanner is ready.</b><br/>Tap RUN FALCON SCANNER to start a fresh scan.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "FALCON V5.5 · Single consolidated dashboard · Calculation engine: falcon_core_v55.py · "
    "Market-data availability depends on NSE/Yahoo Finance and network access."
    f" · Python {sys.version.split()[0]}"
)
