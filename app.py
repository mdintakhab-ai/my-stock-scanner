# =============================================================================
# FALCON NEXT-DAY BEHAVIOUR PREDICTOR V5.5 — MOBILE STREAMLIT EDITION
# GitHub + Streamlit Community Cloud
#
# The calculation engine is kept in falcon_core_v55.py.
# This file only provides the mobile-first Streamlit user interface.
# =============================================================================

from __future__ import annotations

import contextlib
import html
import io
import runpy
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

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #f4f7fb; }
    .block-container {
        max-width: 1180px;
        padding-top: 1.0rem;
        padding-bottom: 2.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .falcon-hero {
        background: linear-gradient(125deg,#102544,#1e4777);
        color: #fff;
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(16,37,68,.14);
    }
    .falcon-kicker { font-size: 11px; letter-spacing: 2px; opacity: .78; font-weight: 800; }
    .falcon-title { font-size: clamp(22px,4vw,32px); font-weight: 900; line-height: 1.15; margin-top: 5px; }
    .falcon-subtitle { font-size: 12px; opacity: .82; margin-top: 7px; }
    .falcon-card-grid {
        display: grid;
        grid-template-columns: repeat(2,minmax(0,1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .falcon-card {
        background: #fff;
        border: 1px solid #dce4ef;
        border-radius: 15px;
        padding: 14px;
        box-shadow: 0 3px 12px rgba(25,45,75,.055);
        min-width: 0;
        color: #172334;
    }
    .falcon-card-head { display:flex; align-items:center; justify-content:space-between; gap:8px; padding-bottom:10px; border-bottom:1px solid #edf1f6; }
    .falcon-stock { font-size:17px; font-weight:900; letter-spacing:.2px; overflow-wrap:anywhere; }
    .falcon-bias { font-size:12px; font-weight:900; padding:5px 9px; border-radius:999px; white-space:nowrap; }
    .falcon-bull { color:#08743d; background:#e8f7ee; }
    .falcon-bear { color:#a51e2b; background:#fff0f0; }
    .falcon-neutral { color:#5b6573; background:#eef1f5; }
    .falcon-price { font-size:19px; font-weight:900; color:#142d4e; margin-top:10px; }
    .falcon-muted { color:#687587; font-size:11px; font-weight:600; }
    .falcon-metrics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:12px; }
    .falcon-metric { background:#f6f8fb; border-radius:10px; padding:8px 9px; min-width:0; }
    .falcon-metric-label { color:#6a7788; font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.45px; }
    .falcon-metric-value { color:#1d2c40; font-size:14px; font-weight:850; margin-top:3px; overflow-wrap:anywhere; }
    .falcon-levels { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; margin-top:10px; }
    .falcon-level { background:#f2f5fa; border-radius:9px; padding:7px 5px; text-align:center; min-width:0; }
    .falcon-level-label { font-size:10px; color:#667386; font-weight:800; }
    .falcon-level-value { font-size:12px; color:#172b46; font-weight:900; margin-top:2px; overflow-wrap:anywhere; }
    .falcon-behaviour { margin-top:11px; padding-top:9px; border-top:1px solid #edf1f6; }
    .falcon-behaviour-title { font-size:11px; color:#59687b; font-weight:900; margin-bottom:6px; }
    .falcon-bar-row { display:grid; grid-template-columns:minmax(90px,1fr) 3fr 42px; gap:7px; align-items:center; margin:5px 0; font-size:10px; color:#566477; }
    .falcon-bar-bg { height:6px; background:#e9eef5; border-radius:20px; overflow:hidden; }
    .falcon-bar { height:6px; background:#547da9; border-radius:20px; }
    .falcon-disclaimer { background:#fff8e8; border:1px solid #f1dfb8; color:#73551b; border-radius:12px; padding:10px 12px; font-size:12px; margin:12px 0; }
    .falcon-empty { background:#fff; border:1px dashed #c7d1df; padding:22px; border-radius:14px; text-align:center; color:#59687b; }
    div[data-testid="stDownloadButton"] button { width:100%; border-radius:10px; }
    @media (max-width: 700px) {
        .block-container { padding-left:.65rem; padding-right:.65rem; padding-top:.65rem; }
        .falcon-card-grid { grid-template-columns:1fr; gap:9px; }
        .falcon-card { padding:12px; border-radius:13px; }
        .falcon-stock { font-size:16px; }
        .falcon-price { font-size:18px; }
        .falcon-metric-value { font-size:13px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="falcon-hero">
      <div class="falcon-kicker">FALCON TRADING INTELLIGENCE</div>
      <div class="falcon-title">🦅 NEXT-DAY BEHAVIOUR PREDICTOR</div>
      <div class="falcon-subtitle">V5.5 · VALIDATION-FIRST · MOBILE DASHBOARD · NSE EQUITY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="falcon-disclaimer"><b>Risk notice:</b> Behaviour shares are heuristic score distributions, not calibrated probabilities. '
    'No 70% or guaranteed accuracy claim is made. Validate with dated walk-forward results before trading.</div>',
    unsafe_allow_html=True,
)

if not CORE_FILE.exists():
    st.error("falcon_core_v55.py is missing. Keep app.py and falcon_core_v55.py in the same GitHub repository folder.")
    st.stop()

left, right = st.columns([1.25, 1])
with left:
    run_clicked = st.button("🚀 RUN FALCON SCANNER", type="primary", use_container_width=True)
with right:
    if st.button("↻ CLEAR PREVIOUS RESULT", use_container_width=True):
        st.session_state.pop("falcon_bundle", None)
        st.rerun()

if run_clicked:
    try:
        captured = io.StringIO()
        with st.spinner("FALCON is loading the official NSE universe, market data and multi-timeframe profiles. This may take several minutes depending on data providers."):
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                namespace = runpy.run_path(str(CORE_FILE), run_name="__falcon_v55_scan__")

        st.session_state["falcon_bundle"] = {
            "results": namespace.get("FALCON_RESULTS", pd.DataFrame()),
            "dashboard": namespace.get("dashboard", pd.DataFrame()),
            "errors": namespace.get("FALCON_ERRORS", []),
            "selected_count": len(namespace.get("selected_rows", [])),
            "target_count": namespace.get("TARGET_FINAL_STOCKS", 7),
            "run_time": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
            "console": captured.getvalue(),
            "direction_map": namespace.get("DIRECTION_BY_STOCK", {}),
        }
        st.success("Scanner execution completed.")
    except Exception as exc:
        st.session_state["falcon_run_error"] = f"{type(exc).__name__}: {exc}"
        st.exception(exc)

if "falcon_run_error" in st.session_state:
    st.error(st.session_state.pop("falcon_run_error"))

bundle = st.session_state.get("falcon_bundle")

def fmt_price(value):
    try:
        number = float(value)
        if pd.notna(number):
            return f"₹{number:,.2f}"
    except Exception:
        pass
    return "—"

def fmt_qty(value):
    try:
        number = float(value)
        if pd.notna(number):
            return f"{number:,.0f}"
    except Exception:
        pass
    return "—"

def metric_html(label, value):
    return (
        '<div class="falcon-metric">'
        f'<div class="falcon-metric-label">{html.escape(str(label))}</div>'
        f'<div class="falcon-metric-value">{html.escape(str(value))}</div>'
        '</div>'
    )

if bundle:
    results = bundle["results"]
    dashboard = bundle["dashboard"]
    errors = bundle["errors"]

    st.caption(f"Last run: {bundle['run_time']} · Selected: {bundle['selected_count']} / {bundle['target_count']}")

    if not isinstance(results, pd.DataFrame) or results.empty:
        st.markdown(
            '<div class="falcon-empty"><b>NO QUALIFIED STOCKS</b><br/>No stock passed all mandatory data-integrity, profile and rejection-zone conditions in this run. This is not a scanner failure by itself.</div>',
            unsafe_allow_html=True,
        )
    else:
        strict_count = int(results.get("_is_strict", pd.Series(dtype=bool)).fillna(False).sum()) if "_is_strict" in results else 0
        bullish_count = int((results.get("_ud_bias", pd.Series(dtype=str)) == "BULLISH").sum()) if "_ud_bias" in results else 0
        bearish_count = int((results.get("_ud_bias", pd.Series(dtype=str)) == "BEARISH").sum()) if "_ud_bias" in results else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PICKS", len(results))
        m2.metric("BULLISH", bullish_count)
        m3.metric("BEARISH", bearish_count)
        m4.metric("STRICT", strict_count)

        cards = []
        for _, row in results.iterrows():
            stock = html.escape(str(row.get("Stock", "N/A")))
            bias = str(row.get("_ud_bias", "NEUTRAL")).upper()
            if bias == "BULLISH":
                bias_class, bias_text = "falcon-bull", "▲ BULLISH"
            elif bias == "BEARISH":
                bias_class, bias_text = "falcon-bear", "▼ BEARISH"
            else:
                bias_class, bias_text = "falcon-neutral", "— NEUTRAL"

            open_label = "Open / Prev" if isinstance(dashboard, pd.DataFrame) and "Open / Prev" in dashboard.columns else "Prev Close"
            open_value = fmt_price(row.get("Display_Prev_Or_Open"))
            level_pairs = [("R1", "R1"), ("R2", "R2"), ("S1", "S1"), ("S2", "S2")]
            level_html = "".join(
                '<div class="falcon-level">'
                f'<div class="falcon-level-label">{label}</div>'
                f'<div class="falcon-level-value">{fmt_price(row.get(key))}</div>'
                '</div>'
                for label, key in level_pairs
            )

            metrics = [
                ("Lower Rejection", fmt_price(row.get("Lower Rejection"))),
                ("Upper Rejection", fmt_price(row.get("Upper Rejection"))),
                ("Max Single-Order Qty", fmt_qty(row.get("MAX SINGLE-ORDER QTY"))),
                ("Target Open", fmt_price(row.get("Target Open"))),
                ("Support / Down", fmt_price(row.get("Support/Down"))),
                ("ST Avg", fmt_price(row.get("ST Avg"))),
                ("ST VAH", fmt_price(row.get("ST VAH"))),
                ("LT VAL", fmt_price(row.get("LT VAL"))),
                ("LT VAH Avg", fmt_price(row.get("LT VAH Avg"))),
            ]
            metrics_html = "".join(metric_html(k, v) for k, v in metrics)

            behaviour = row.get("All Behaviour Heuristic Shares", {})
            behaviour_html = ""
            if isinstance(behaviour, dict) and behaviour:
                bars = []
                for name, pct in sorted(behaviour.items(), key=lambda item: float(item[1] or 0), reverse=True):
                    try:
                        pct_num = max(0.0, min(100.0, float(pct)))
                    except Exception:
                        pct_num = 0.0
                    bars.append(
                        '<div class="falcon-bar-row">'
                        f'<span>{html.escape(str(name))}</span>'
                        f'<div class="falcon-bar-bg"><div class="falcon-bar" style="width:{pct_num:.1f}%"></div></div>'
                        f'<b>{pct_num:.1f}%</b></div>'
                    )
                behaviour_html = (
                    '<div class="falcon-behaviour"><div class="falcon-behaviour-title">'
                    'BEHAVIOUR HEURISTIC SHARE · NOT CALIBRATED PROBABILITY</div>'
                    + "".join(bars) + '</div>'
                )

            cards.append(
                '<div class="falcon-card">'
                '<div class="falcon-card-head">'
                f'<div class="falcon-stock">{stock}</div>'
                f'<div class="falcon-bias {bias_class}">{bias_text}</div>'
                '</div>'
                f'<div class="falcon-price">{open_value}</div>'
                f'<div class="falcon-muted">{open_label}</div>'
                f'<div class="falcon-levels">{level_html}</div>'
                f'<div class="falcon-metrics">{metrics_html}</div>'
                f'{behaviour_html}'
                '</div>'
            )

        st.markdown('<div class="falcon-card-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)

        st.markdown("### Detailed dashboard")
        if isinstance(dashboard, pd.DataFrame) and not dashboard.empty:
            st.dataframe(dashboard, use_container_width=True, hide_index=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download complete scanner results (CSV)",
            data=csv_bytes,
            file_name="FALCON_V5_5_FULL_RESULTS.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if errors:
        with st.expander(f"Data / analysis warnings ({len(errors)})"):
            for message in errors[:100]:
                st.write(f"• {message}")
            if len(errors) > 100:
                st.caption(f"Showing first 100 of {len(errors)} messages.")

    if bundle.get("console"):
        with st.expander("Scanner execution log"):
            st.code(bundle["console"][-12000:] or "No console output.")
else:
    st.markdown(
        '<div class="falcon-empty"><b>Scanner is ready.</b><br/>Tap RUN FALCON SCANNER to start a fresh scan. '
        'The scan does not run automatically when the page is opened or refreshed.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("FALCON V5.5 Mobile UI · Calculation engine: falcon_core_v55.py · Market data availability and NSE/Yahoo Finance access are external dependencies.")
