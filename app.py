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
