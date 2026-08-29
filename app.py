import io
import logging
import os
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Suppress logs and warnings
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Streamlit Page Configuration for Mobile / Responsive UI
st.set_page_config(
    page_title="Falcon Trinity Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Mobile-Optimized CSS
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    .stDataFrame {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 1. CORE MATHEMATICAL & TECHNICAL INDICATORS
# =====================================================================
def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if np.isnan(value):
        return 0.0
    return max(min_val, min(float(value), max_val))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calculate_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    if df_temp.index.tz is None:
        df_temp.index = df_temp.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    else:
        df_temp.index = df_temp.index.tz_convert("Asia/Kolkata")

    df_temp["Session_Date"] = df_temp.index.date
    typical_price = (df_temp["High"] + df_temp["Low"] + df_temp["Close"]) / 3.0
    df_temp["TP_Vol"] = typical_price * df_temp["Volume"]

    cum_tp_vol = df_temp.groupby("Session_Date")["TP_Vol"].cumsum()
    cum_vol = df_temp.groupby("Session_Date")["Volume"].cumsum()
    return (cum_tp_vol / (cum_vol + 1e-9)).fillna(df_temp["Close"])


# =====================================================================
# 2. OPTIMIZED SMC STATE & DEMAND/SUPPLY MATRIX ENGINE
# =====================================================================
def process_smc_matrix(
    df: pd.DataFrame,
    pivot_matrix: int = 2,
    pivot_smc: int = 3,
    state_life: int = 40,
    fvg_min_atr: float = 0.08,
    merge_thresh: float = 0.3,
    max_tests: int = 4,
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n < 15:
        return df

    # Ribbon EMAs (8, 13, 21)
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ATR"] = calculate_atr(df, 14).bfill().ffill()
    df["RSI"] = calculate_rsi(df["Close"], 14)
    df["VWAP"] = calculate_intraday_vwap(df)
    df["Vol_SMA14"] = df["Volume"].rolling(14, min_periods=1).mean()

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)
    vols = df["Volume"].to_numpy(dtype=float)
    vol_sma = df["Vol_SMA14"].to_numpy(dtype=float)
    ema8 = df["EMA8"].to_numpy(dtype=float)
    ema13 = df["EMA13"].to_numpy(dtype=float)
    ema21 = df["EMA21"].to_numpy(dtype=float)
    rsi = df["RSI"].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float)

    # 1. SMC Ribbon & Dynamic EMA13 Logic
    trend_smc = np.zeros(n, dtype=int)
    for i in range(1, n):
        ema_up = ema13[i] >= ema13[i - 1]
        ema_dn = ema13[i] <= ema13[i - 1]
        
        # Bullish: Ribbon Stack ya Strong Momentum over EMA13
        bull_cond = (closes[i] >= ema8[i] and ema8[i] >= ema13[i] and ema13[i] >= ema21[i]) or (
            closes[i] >= ema13[i] and ema_up and rsi[i] >= 50
        )
        
        # Bearish: Ribbon Stack ya Strong Drag under EMA13
        bear_cond = (closes[i] <= ema8[i] and ema8[i] <= ema13[i] and ema13[i] <= ema21[i]) or (
            closes[i] <= ema13[i] and ema_dn and rsi[i] <= 50
        )

        if bull_cond:
            trend_smc[i] = 1
        elif bear_cond:
            trend_smc[i] = -1
        else:
            trend_smc[i] = 0

    df["Trend_SMC"] = trend_smc

    # 2. Demand & Supply Matrix Processing
    has_active_demand = np.zeros(n, dtype=bool)
    has_active_supply = np.zeros(n, dtype=bool)
    supply_zones, demand_zones = [], []

    for i in range(pivot_matrix * 2, n):
        p_idx = i - pivot_matrix
        w_start = p_idx - pivot_matrix
        w_end = p_idx + pivot_matrix + 1

        is_phi = highs[p_idx] == np.max(highs[w_start:w_end])
        is_plo = lows[p_idx] == np.min(lows[w_start:w_end])

        if is_phi:
            top_lvl = highs[p_idx]
            bot_lvl = max(opens[p_idx], closes[p_idx])
            is_dup = any(s["active"] and abs(s["top"] - top_lvl) < (atr[i] * merge_thresh) for s in supply_zones)
            if not is_dup:
                supply_zones.append({"top": top_lvl, "bot": bot_lvl, "active": True, "tests": 0})

        if is_plo:
            top_lvl = min(opens[p_idx], closes[p_idx])
            bot_lvl = lows[p_idx]
            is_dup = any(d["active"] and abs(d["bot"] - bot_lvl) < (atr[i] * merge_thresh) for d in demand_zones)
            if not is_dup:
                demand_zones.append({"top": top_lvl, "bot": bot_lvl, "active": True, "tests": 0})

        cur_h, cur_l = highs[i], lows[i]

        for s in supply_zones:
            if s["active"]:
                if cur_h > s["top"]:
                    s["active"] = False
                elif cur_h >= s["bot"]:
                    s["tests"] += 1
                    if s["tests"] >= max_tests:
                        s["active"] = False

        for d in demand_zones:
            if d["active"]:
                if cur_l < d["bot"]:
                    d["active"] = False
                elif cur_l <= d["top"]:
                    d["tests"] += 1
                    if d["tests"] >= max_tests:
                        d["active"] = False

        in_supply = any(s["active"] and (cur_h >= s["bot"] and cur_l <= s["top"] * 1.002) for s in supply_zones)
        in_demand = any(d["active"] and (cur_l <= d["top"] and cur_h >= d["bot"] * 0.998) for d in demand_zones)

        if in_supply:
            in_demand = False

        has_active_supply[i] = in_supply
        has_active_demand[i] = in_demand

    df["Active_Demand_Box"] = has_active_demand
    df["Active_Supply_Box"] = has_active_supply
    return df


# =====================================================================
# 3. PRE-MARKET & LIVE COBI ENGINE
# =====================================================================
def compute_live_candle_scores(df: pd.DataFrame):
    if len(df) < 11:
        return 0.0, 0.0

    last_open = float(df["Open"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    last_atr = float(df["ATR"].iloc[-1])
    gap = (last_open - prev_close) / (last_atr + 1e-9)

    last_vol = float(df["Volume"].iloc[-1])
    avg_vol = float(df["Volume"].rolling(10, min_periods=1).mean().iloc[-1])
    vol_ratio = (last_vol / (avg_vol + 1e-9)) - 1.0

    pms = round(0.5 * clamp(gap, -1.0, 1.0) + 0.5 * clamp(vol_ratio, -1.0, 1.0), 4)

    last_close = float(df["Close"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])
    last_high = float(df["High"].iloc[-1])

    vol_weight = (last_close - last_low) - (last_high - last_close)
    denom = (last_high - last_low) + 1e-9
    cobi_val = (vol_weight / denom) * (last_vol / (avg_vol + 1e-9))
    cobi = round(clamp(cobi_val, -1.0, 1.0), 4)

    return pms, cobi


# =====================================================================
# 4. DATA ENGINE & EXPANDED SYMBOLS UNIVERSE
# =====================================================================
@st.cache_data(ttl=60)
def load_intraday_data(symbol: str):
    try:
        t = yf.Ticker(symbol)
        df_1m = t.history(period="5d", interval="1m", auto_adjust=False)

        if df_1m.empty or len(df_1m) < 30:
            df_5m = t.history(period="5d", interval="5m", auto_adjust=False)
            if df_5m.empty or len(df_5m) < 15:
                return pd.DataFrame(), pd.DataFrame()
            return df_5m, df_5m

        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df_1m.index = df_1m.index.tz_convert("Asia/Kolkata")

        df_1m = df_1m.between_time("09:15", "15:30")

        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        df_3m = df_1m.resample("3min").agg(agg_dict).dropna()
        df_5m = df_1m.resample("5min").agg(agg_dict).dropna()

        df_3m = df_3m[df_3m["Volume"] > 0]
        df_5m = df_5m[df_5m["Volume"] > 0]

        return df_3m, df_5m
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def get_expanded_symbols():
    fallback = [
        "VTL.NS", "ARVIND.NS", "ITC.NS", "VEDL.NS", "COALINDIA.NS", "NTPC.NS", "TATAPOWER.NS",
        "BPCL.NS", "HINDPETRO.NS", "WIPRO.NS", "DABUR.NS", "BEL.NS",
        "APOLLOTYRE.NS", "AMBUJACEM.NS", "EXIDEIND.NS", "BHEL.NS",
        "NATIONALUM.NS", "GNFC.NS", "CHAMBLFERT.NS", "UPL.NS",
        "AARTIIND.NS", "M&MFIN.NS", "LICHSGFIN.NS", "MANAPPURAM.NS",
        "ABCAPITAL.NS", "BIOCON.NS", "PRECWIRE.NS",
    ]
    try:
        url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            csv_data = response.read()
            df = pd.read_csv(io.BytesIO(csv_data))
            sym_col = [c for c in df.columns if "Symbol" in c or "SYMBOL" in c]
            if sym_col:
                fetched = [f"{sym.strip()}.NS" for sym in df[sym_col[0]].dropna().unique()]
                return sorted(list(set(fetched)))
    except Exception:
        pass
    return sorted(list(set(fallback)))


# =====================================================================
# 5. UNIFIED EVALUATION & TABLE ROUTING
# =====================================================================
def evaluate_symbol(symbol: str, min_p: float = 300.0, max_p: float = 600.0):
    df_3m, df_5m = load_intraday_data(symbol)
    if df_5m.empty or len(df_5m) < 15 or df_3m.empty or len(df_3m) < 15:
        return None

    df_3m_calc = process_smc_matrix(df_3m)
    df_5m_calc = process_smc_matrix(df_5m)

    ltp = round(float(df_5m_calc["Close"].iloc[-1]), 2)
    vwap_val = round(float(df_5m_calc["VWAP"].iloc[-1]), 2)

    # Price Filter
    if not (min_p <= ltp <= max_p):
        return None

    pms_val, cobi_val = compute_live_candle_scores(df_5m_calc)

    ema_3m_trend = int(df_3m_calc["Trend_SMC"].iloc[-1])
    ema_5m_trend = int(df_5m_calc["Trend_SMC"].iloc[-1])

    ema_3m_green = ema_3m_trend == 1
    ema_3m_red = ema_3m_trend == -1
    ema_5m_green = ema_5m_trend == 1
    ema_5m_red = ema_5m_trend == -1

    box_3m_demand = bool(df_3m_calc["Active_Demand_Box"].iloc[-1])
    box_3m_supply = bool(df_3m_calc["Active_Supply_Box"].iloc[-1])
    box_5m_demand = bool(df_5m_calc["Active_Demand_Box"].iloc[-1])
    box_5m_supply = bool(df_5m_calc["Active_Supply_Box"].iloc[-1])

    # Table 1 Setup Tagging
    if ema_3m_green and ema_5m_green and ltp >= vwap_val:
        setup_t1 = "🔥 GRADE-A+ BUY (3M+5M)"
    elif ema_3m_red and ema_5m_red and ltp <= vwap_val:
        setup_t1 = "💥 GRADE-A+ SELL (3M+5M)"
    elif ema_5m_green and ltp >= vwap_val:
        setup_t1 = "🟢 GRADE-A BUY (5M)"
    elif ema_5m_red and ltp <= vwap_val:
        setup_t1 = "🔴 GRADE-A SELL (5M)"
    else:
        setup_t1 = "⚪ WATCHLIST"

    base_row = {
        "Symbol": symbol.replace(".NS", ""),
        "LTP (₹)": ltp,
        "VWAP (₹)": vwap_val,
        "EMA13 (3M)": "🟢 GREEN" if ema_3m_green else "🔴 RED" if ema_3m_red else "🟡 YELLOW",
        "Box (3M)": "🟢 DEMAND" if box_3m_demand else "🔴 SUPPLY" if box_3m_supply else "⚪ NONE",
        "EMA13 (5M)": "🟢 GREEN" if ema_5m_green else "🔴 RED" if ema_5m_red else "🟡 YELLOW",
        "Box (5M)": "🟢 DEMAND" if box_5m_demand else "🔴 SUPPLY" if box_5m_supply else "⚪ NONE",
        "PMS": pms_val,
        "COBI": cobi_val,
    }

    # Table 2 Perfect Alignment: 
    # 1. 3M + 5M EMA13 dono same direction (GREEN/RED)
    # 2. Price VWAP ke sahi side ho
    # 3. Opposite Supply/Demand Zone me na fasa ho
    # 4. Positive COBI support
    table2_buy = (
        ema_3m_green
        and ema_5m_green
        and (ltp >= vwap_val)
        and (not box_3m_supply)
        and (not box_5m_supply)
        and (cobi_val >= 0.05)
    )

    table2_sell = (
        ema_3m_red
        and ema_5m_red
        and (ltp <= vwap_val)
        and (not box_3m_demand)
        and (not box_5m_demand)
        and (cobi_val <= -0.05)
    )

    if table2_buy:
        row_t2 = base_row.copy()
        row_t2["Signal Setup"] = "🟢 BUY (INSTITUTIONAL)"
        return ("T2", row_t2)
    elif table2_sell:
        row_t2 = base_row.copy()
        row_t2["Signal Setup"] = "🔴 SELL (INSTITUTIONAL)"
        return ("T2", row_t2)
    else:
        row_t1 = base_row.copy()
        row_t1["Signal Setup"] = setup_t1
        return ("T1", row_t1)


# =====================================================================
# 6. STREAMLIT APP UI & CONTROLS
# =====================================================================
st.title("⚡ Falcon Trinity Quant Screener")
st.caption("Real-Time Multi-Timeframe Institutional SMC & VWAP Engine")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    min_price = st.number_input("Min Price (₹)", value=300.0, step=10.0)
    max_price = st.number_input("Max Price (₹)", value=600.0, step=10.0)
    scan_limit = st.slider("Max Stocks to Scan", min_value=10, max_value=250, value=75, step=10)
    auto_refresh = st.checkbox("Auto Refresh (1 Min)", value=False)
    btn_scan = st.button("🚀 Run Live Scan", use_container_width=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()

symbols = get_expanded_symbols()[:scan_limit]

if btn_scan or "scanned_data" not in st.session_state:
    table1_rows, table2_rows = [], []
    progress_bar = st.progress(0, text="Scanning Market Universe...")

    for idx, sym in enumerate(symbols):
        progress_bar.progress((idx + 1) / len(symbols), text=f"Scanning ({idx + 1}/{len(symbols)}): {sym}")
        res = evaluate_symbol(sym, min_price, max_price)
        if res is not None:
            t_type, row_data = res
            if t_type == "T2":
                table2_rows.append(row_data)
            else:
                table1_rows.append(row_data)

    progress_bar.empty()
    st.session_state["table1_rows"] = table1_rows
    st.session_state["table2_rows"] = table2_rows
    st.session_state["scanned_data"] = True

table1_rows = st.session_state.get("table1_rows", [])
table2_rows = st.session_state.get("table2_rows", [])

# Quick Stats Ribbon
col1, col2, col3 = st.columns(3)
col1.metric("Scanned Universe", f"{len(symbols)} Stocks")
col2.metric("Watchlist Trades (T1)", f"{len(table1_rows)}")
col3.metric("Perfect Aligned (T2)", f"{len(table2_rows)}")

st.divider()

# TABLE 2: PERFECT TRINITY ALIGNED TRADES
st.subheader("💎 TABLE 2: 100% PERFECT TRINITY ALIGNED TRADES")
if table2_rows:
    df_t2 = pd.DataFrame(table2_rows).sort_values(by="Symbol")
    st.dataframe(
        df_t2,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "VWAP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "PMS": st.column_config.NumberColumn(format="%.4f"),
            "COBI": st.column_config.NumberColumn(format="%.4f"),
        },
    )
else:
    st.info("⚠️ No strictly aligned trades available for Table 2 at this moment.")

st.divider()

# TABLE 1: WATCHLIST & GRADE-A TRADES
st.subheader("📋 TABLE 1: QUANTITATIVE WATCHLIST & GRADE-A TRADES")
if table1_rows:
    df_t1 = pd.DataFrame(table1_rows).sort_values(by="Symbol")
    st.dataframe(
        df_t1,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "VWAP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "PMS": st.column_config.NumberColumn(format="%.4f"),
            "COBI": st.column_config.NumberColumn(format="%.4f"),
        },
    )
else:
    st.info("No stocks matched Table 1 criteria.")
