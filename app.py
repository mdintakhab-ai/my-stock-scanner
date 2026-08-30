import io
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------------
# 0. STREAMLIT PAGE CONFIG & MOBILE LANDSCAPE ZOOM VIEWPORT
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional SMC & Order Flow Radar",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enable pinch-to-zoom and responsive landscape single-page scaling
st.markdown(
    """
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=0.75, minimum-scale=0.2, maximum-scale=5.0, user-scalable=yes">
    </head>
    <style>
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100% !important;
        }
        header, footer {visibility: hidden !important;}
        body {background-color: #0E1118;}
    </style>
    """,
    unsafe_allow_html=True
)

# Suppress all notebook deprecation and runtime warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ------------------------------------------------------------------
# 1. MATHEMATICAL & TECHNICAL CALCULATORS
# ------------------------------------------------------------------
def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return max(min_val, min(float(value), max_val))


def calculate_ema(series: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    tr1 = high - low
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return pd.Series(tr).ewm(alpha=1.0 / length, adjust=False).mean().values


def calculate_rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_intraday_vwap(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, bars: int = 75
) -> float:
    n = min(len(close), bars)
    typical_price = (high[-n:] + low[-n:] + close[-n:]) / 3.0
    vols = volume[-n:]
    total_vol = float(np.sum(vols))
    return (
        round(float(np.sum(typical_price * vols) / total_vol), 2)
        if total_vol > 0
        else round(float(close[-1]), 2)
    )


def calculate_pms(
    vol_ratio: float, order_flow_bias: float, gap_atr: float, rs_metric: float
) -> float:
    w1, w2, w3, w4 = 0.35, 0.30, 0.20, 0.15
    score = (
        (w1 * clamp(vol_ratio, -1.0, 1.0))
        + (w2 * clamp(order_flow_bias, -1.0, 1.0))
        + (w3 * clamp(gap_atr, -1.0, 1.0))
        + (w4 * clamp(rs_metric, -1.0, 1.0))
    )
    return round(clamp(score, -1.0, 1.0), 4)


def calculate_cobi(bid_vol: float, ask_vol: float) -> float:
    total = bid_vol + ask_vol
    return round(clamp((bid_vol - ask_vol) / total if total > 0 else 0.0, -1.0, 1.0), 4)


# ------------------------------------------------------------------
# 2. MICROSTRUCTURE FLOW & SURGE DETECTOR
# ------------------------------------------------------------------
def detect_flow_shock(
    close_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray, vol_arr: np.ndarray
) -> Tuple[float, float, str, float, float]:
    if len(close_arr) < 5:
        return 50.0, 1.0, "NEUTRAL", 0.0, 0.0

    price_range = np.maximum(high_arr - low_arr, 1e-5)
    buy_share = np.clip((close_arr - low_arr) / price_range, 0.0, 1.0)
    buy_vol_arr = vol_arr * buy_share
    sell_vol_arr = vol_arr * (1.0 - buy_share)

    recent_buy = float(np.sum(buy_vol_arr[-5:]))
    recent_total = float(np.sum(vol_arr[-5:])) + 1e-5
    buy_pressure_pct = round((recent_buy / recent_total) * 100, 1)

    denom = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
    vol_acc = round(float(np.mean(vol_arr[-3:])) / (denom + 1e-5), 2)

    if buy_pressure_pct >= 68.0 and vol_acc >= 1.35:
        flow_shock = "DEMAND SURGE"
    elif buy_pressure_pct <= 32.0 and vol_acc >= 1.35:
        flow_shock = "SUPPLY SURGE"
    else:
        flow_shock = "NEUTRAL"

    window_10_buy = float(np.sum(buy_vol_arr[-10:]))
    window_10_sell = float(np.sum(sell_vol_arr[-10:]))

    return buy_pressure_pct, vol_acc, flow_shock, window_10_buy, window_10_sell


# ------------------------------------------------------------------
# 3. SMC STATE MACHINE ENGINE
# ------------------------------------------------------------------
def run_smc_state_machine(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    vol: np.ndarray, atr: np.ndarray, rsi: np.ndarray, ema13: np.ndarray,
    pivot_len: int = 3, state_life: int = 25, fvg_min_atr: float = 0.1
) -> Dict[str, any]:
    n = len(close)
    if n < 30:
        return {"bull_state": 0, "bear_state": 0, "smc_signal": "NONE", "status_desc": "No Pattern"}

    last_high, last_low = np.nan, np.nan
    bull_state, bull_bars = 0, 0
    bear_state, bear_bars = 0, 0

    for i in range(pivot_len * 2, n):
        bull_bars += 1
        bear_bars += 1

        p_idx = i - pivot_len
        left_highs = high[p_idx - pivot_len : p_idx]
        right_highs = high[p_idx + 1 : i + 1]
        if len(left_highs) > 0 and len(right_highs) > 0:
            if high[p_idx] >= np.max(left_highs) and high[p_idx] >= np.max(right_highs):
                last_high = high[p_idx]

        left_lows = low[p_idx - pivot_len : p_idx]
        right_lows = low[p_idx + 1 : i + 1]
        if len(left_lows) > 0 and len(right_lows) > 0:
            if low[p_idx] <= np.min(left_lows) and low[p_idx] <= np.min(right_lows):
                last_low = low[p_idx]

        bull_sweep = not np.isnan(last_low) and low[i] < last_low and close[i] > last_low
        bull_mss = not np.isnan(last_high) and close[i] > last_high and close[i - 1] <= last_high
        bull_fvg = (low[i] - high[i - 2]) > (atr[i] * fvg_min_atr) if i >= 2 else False

        bear_sweep = not np.isnan(last_high) and high[i] > last_high and close[i] < last_high
        bear_mss = not np.isnan(last_low) and close[i] < last_low and close[i - 1] >= last_low
        bear_fvg = (low[i - 2] - high[i]) > (atr[i] * fvg_min_atr) if i >= 2 else False

        if bull_sweep:
            bull_state, bull_bars = 1, 0
        if bull_state >= 1 and bull_mss:
            bull_state, bull_bars = 2, 0
        if bull_state >= 2 and bull_fvg:
            bull_state, bull_bars = 4, 0
        if bull_bars > state_life:
            bull_state = 0

        if bear_sweep:
            bear_state, bear_bars = 1, 0
        if bear_state >= 1 and bear_mss:
            bear_state, bear_bars = 2, 0
        if bear_state >= 2 and bear_fvg:
            bear_state, bear_bars = 4, 0
        if bear_bars > state_life:
            bear_state = 0

    ema_up = ema13[-1] > ema13[-2] if len(ema13) >= 2 else False
    ema_dn = ema13[-1] < ema13[-2] if len(ema13) >= 2 else False

    bull_confirm = (bull_state >= 2) and (close[-1] > ema13[-1]) and ema_up and (rsi[-1] >= 50.0)
    bear_confirm = (bear_state >= 2) and (close[-1] < ema13[-1]) and ema_dn and (rsi[-1] <= 50.0)

    signal = "BUY" if bull_confirm else ("SELL" if bear_confirm else "NONE")
    status_desc = "BULL MSS+FVG" if bull_state == 4 else ("BULL MSS" if bull_state == 2 else ("BEAR MSS+FVG" if bear_state == 4 else ("BEAR MSS" if bear_state == 2 else "PULLBACK")))

    return {
        "bull_state": bull_state,
        "bear_state": bear_state,
        "smc_signal": signal,
        "status_desc": status_desc,
    }


# ------------------------------------------------------------------
# 4. INTERMARKET PULSE ENGINE
# ------------------------------------------------------------------
SECTOR_LEADER_MAP = {
    "TATAMOTORS": "MARUTI.NS", "M&M": "MARUTI.NS", "ASHOKLEY": "MARUTI.NS",
    "TATASTEEL": "HINDALCO.NS", "SAIL": "HINDALCO.NS", "JINDALSTEL": "HINDALCO.NS",
    "SBIN": "HDFCBANK.NS", "ICICIBANK": "HDFCBANK.NS", "PNB": "HDFCBANK.NS",
    "CANBK": "HDFCBANK.NS", "FEDERALBNK": "HDFCBANK.NS",
    "INFY": "TCS.NS", "WIPRO": "TCS.NS", "HCLTECH": "TCS.NS", "TECHM": "TCS.NS",
    "CIPLA": "SUNPHARMA.NS", "DRREDDY": "SUNPHARMA.NS", "LUPIN": "SUNPHARMA.NS",
    "TATAPOWER": "RELIANCE.NS", "ONGC": "RELIANCE.NS", "BPCL": "RELIANCE.NS",
    "IOC": "RELIANCE.NS", "VEDL": "HINDALCO.NS"
}

def evaluate_intermarket_action(
    stock_pct_chg: float, leader_pct_chg: float, nifty_pct_chg: float,
    price: float, vwap_val: float, open_price: float
) -> Tuple[str, str]:
    stock_up = price >= vwap_val and price >= open_price
    stock_down = price <= vwap_val and price <= open_price

    if stock_up and stock_pct_chg > 1.2 and leader_pct_chg < -0.4 and nifty_pct_chg < 0.0:
        return "SELLERS OVERTAKE", "TRAP"
    if stock_up and stock_pct_chg > 0.8 and (leader_pct_chg < -0.2 or nifty_pct_chg < -0.15):
        return "FAKE RALLY", "TRAP"
    if stock_down and stock_pct_chg < -0.8 and leader_pct_chg > 0.2 and nifty_pct_chg > 0.15:
        return "FAKE DROP", "TRAP"
    if stock_up and leader_pct_chg >= 0.1 and nifty_pct_chg >= 0.05:
        return "INSTITUTIONAL SYNC", "BUY"
    if stock_down and leader_pct_chg <= -0.1 and nifty_pct_chg <= -0.05:
        return "BEARISH ROTATION", "SELL"

    return "MOMENTUM SYNC", ("BUY" if stock_up else ("SELL" if stock_down else "NEUTRAL"))


# ------------------------------------------------------------------
# 5. UNIVERSE FETCHER
# ------------------------------------------------------------------
def fetch_nse_symbols() -> List[str]:
    urls = [
        "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                if "Symbol" in df.columns:
                    return df["Symbol"].dropna().unique().tolist()
        except Exception:
            continue
    return [
        "TATAPOWER", "VEDL", "DLF", "FEDERALBNK", "IDFCFIRSTB", "CANBK",
        "PNB", "SAIL", "ARVINDFASN", "PATANJALI", "LICI", "NAZARA",
        "SONATSOFTW", "APOLLO", "KEC", "THYROCARE", "EUREKAFORB", "TRITURBINE"
    ]


symbols = fetch_nse_symbols()
ticker_list = [f"{sym}.NS" for sym in symbols]

# ------------------------------------------------------------------
# 6. LIVE SCANNER ENGINE (STREAMLIT INTEGRATED)
# ------------------------------------------------------------------
REFRESH_SECONDS = 30
radar_placeholder = st.empty()

while True:
    results = []
    unique_leaders = list(set(SECTOR_LEADER_MAP.values()))
    benchmarks_to_pull = ["^NSEI"] + unique_leaders
    leader_perf_map: Dict[str, float] = {}
    nifty_pct_chg = 0.0

    try:
        bench_data = yf.download(
            benchmarks_to_pull, period="2d", interval="5m", group_by="ticker", progress=False
        )
        
        def extract_clean_df(source_df: pd.DataFrame, ticker_sym: str) -> pd.DataFrame:
            if isinstance(source_df.columns, pd.MultiIndex):
                if ticker_sym in source_df.columns.levels[0]:
                    return source_df[ticker_sym].dropna()
                elif len(source_df.columns.levels) > 1 and ticker_sym in source_df.columns.levels[1]:
                    return source_df.xs(ticker_sym, axis=1, level=1).dropna()
            return source_df.dropna()

        nifty_df = extract_clean_df(bench_data, "^NSEI")
        if not nifty_df.empty:
            n_bars = min(75, len(nifty_df))
            nifty_open = float(nifty_df["Open"].iloc[-n_bars])
            nifty_close = float(nifty_df["Close"].iloc[-1])
            nifty_pct_chg = round(((nifty_close - nifty_open) / nifty_open) * 100, 2)

        for l_ticker in unique_leaders:
            l_df = extract_clean_df(bench_data, l_ticker)
            if not l_df.empty:
                l_bars = min(75, len(l_df))
                l_open = float(l_df["Open"].iloc[-l_bars])
                l_close = float(l_df["Close"].iloc[-1])
                leader_perf_map[l_ticker] = round(((l_close - l_open) / l_open) * 100, 2)
            else:
                leader_perf_map[l_ticker] = nifty_pct_chg
    except Exception:
        pass

    BATCH_SIZE = 100
    for i in range(0, len(ticker_list), BATCH_SIZE):
        batch_tickers = ticker_list[i : i + BATCH_SIZE]
        batch_symbols = symbols[i : i + BATCH_SIZE]

        try:
            data = yf.download(
                tickers=batch_tickers, period="5d", interval="5m",
                group_by="ticker", threads=True, progress=False
            )
        except Exception:
            continue

        for sym, ticker in zip(batch_symbols, batch_tickers):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]:
                        df = data[ticker].dropna()
                    elif len(data.columns.levels) > 1 and ticker in data.columns.levels[1]:
                        df = data.xs(ticker, axis=1, level=1).dropna()
                    else:
                        continue
                else:
                    df = data.dropna()

                if df.empty or len(df) < 35:
                    continue

                close = df["Close"].values.astype(float)
                open_p = df["Open"].values.astype(float)
                high = df["High"].values.astype(float)
                low = df["Low"].values.astype(float)
                vol = df["Volume"].values.astype(float)

                current_price = round(float(close[-1]), 2)
                if not (300.0 <= current_price <= 600.0):
                    continue

                ema13 = calculate_ema(close, 13)
                ema21 = calculate_ema(close, 21)
                atr = calculate_atr(high, low, close, 14)
                rsi = calculate_rsi(close, 14)
                vwap_val = calculate_intraday_vwap(high, low, close, vol)
                
                atr_val = float(atr[-1]) if atr[-1] > 0 else 1.0
                ema13_val = round(float(ema13[-1]), 2)
                ema_rising = ema13[-1] > ema13[-2] if len(ema13) >= 2 else False
                ema_falling = ema13[-1] < ema13[-2] if len(ema13) >= 2 else False

                is_bull_ribbon = (ema13[-1] > ema21[-1]) and ema_rising
                is_bear_ribbon = (ema13[-1] < ema21[-1]) and ema_falling

                bars_today = min(len(close), 75)
                session_open = float(open_p[-bars_today])
                prev_ref = float(close[-bars_today]) if len(close) >= bars_today else float(close[0])
                gap_pts = current_price - prev_ref
                stock_pct_chg = round(((current_price - session_open) / session_open) * 100, 2)

                buy_p_pct, vol_acc, flow_shock, b_vol, s_vol = detect_flow_shock(close, high, low, vol)

                recent_vol = float(np.sum(vol[-10:]))
                avg_vol = float(np.mean(vol[-bars_today:])) * 10 if len(vol) >= bars_today else float(np.mean(vol)) * 10
                vol_ratio = (recent_vol / avg_vol) - 1.0 if avg_vol > 0 else 0.0

                cobi_val = calculate_cobi(b_vol, s_vol)
                pms_val = calculate_pms(vol_ratio, cobi_val, gap_pts / atr_val, (gap_pts / prev_ref) * 10)
                smc_res = run_smc_state_machine(high, low, close, vol, atr, rsi, ema13)

                leader_ticker = SECTOR_LEADER_MAP.get(sym, None)
                leader_chg = leader_perf_map.get(leader_ticker, nifty_pct_chg)
                market_state, intermarket_dir = evaluate_intermarket_action(
                    stock_pct_chg, leader_chg, nifty_pct_chg, current_price, vwap_val, float(open_p[-1])
                )

                signal = "REJECT"
                action_type = ""
                limit_entry = current_price
                stop_loss = 0.0
                target_price = 0.0

                if market_state not in ["SELLERS OVERTAKE", "FAKE RALLY", "FAKE DROP"]:
                    if (
                        pms_val >= 0.28
                        and cobi_val >= 0.25
                        and current_price >= vwap_val
                        and is_bull_ribbon
                        and buy_p_pct >= 58.0
                        and vol_acc >= 1.12
                        and (smc_res["smc_signal"] == "BUY" or rsi[-1] >= 50.0)
                        and intermarket_dir in ["BUY", "NEUTRAL"]
                    ):
                        signal = "BUY"
                        limit_entry = ema13_val if (current_price - ema13_val) > (atr_val * 0.4) else current_price
                        stop_loss = round(limit_entry - (atr_val * 1.5), 2)
                        risk = limit_entry - stop_loss
                        target_price = round(limit_entry + (risk * 2.0), 2)
                        action_type = "LIMIT BUY" if limit_entry < current_price else "BUY NOW"

                    elif (
                        pms_val <= -0.28
                        and cobi_val <= -0.25
                        and current_price <= vwap_val
                        and is_bear_ribbon
                        and buy_p_pct <= 42.0
                        and vol_acc >= 1.12
                        and (smc_res["smc_signal"] == "SELL" or rsi[-1] <= 50.0)
                        and intermarket_dir in ["SELL", "NEUTRAL"]
                    ):
                        signal = "SELL"
                        limit_entry = ema13_val if (ema13_val - current_price) > (atr_val * 0.4) else current_price
                        stop_loss = round(limit_entry + (atr_val * 1.5), 2)
                        risk = stop_loss - limit_entry
                        target_price = round(limit_entry - (risk * 2.0), 2)
                        action_type = "LIMIT SELL" if limit_entry > current_price else "SELL NOW"

                if signal in ["BUY", "SELL"]:
                    results.append({
                        "Symbol": sym,
                        "LTP": f"₹{current_price:,.2f}",
                        "VWAP": f"₹{vwap_val:,.2f}",
                        "PMS": pms_val,
                        "COBI": cobi_val,
                        "Flow Pressure": buy_p_pct,
                        "Vol Acc": f"{vol_acc}x",
                        "Flow Shock": flow_shock,
                        "SMC Structure": smc_res["status_desc"],
                        "Market Pulse": market_state,
                        "Plan": action_type,
                        "Entry": f"₹{limit_entry:,.2f}",
                        "SL": f"₹{stop_loss:,.2f}",
                        "Target": f"₹{target_price:,.2f}",
                        "Strategy": signal
                    })
            except Exception:
                continue

    # ------------------------------------------------------------------
    # 7. ULTRA-COMPACT PROFESSIONAL RADAR UI (STREAMLIT RENDER)
    # ------------------------------------------------------------------
    current_time = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")
    df_trades = pd.DataFrame(results)

    with radar_placeholder.container():
        if not df_trades.empty:
            df_trades = df_trades.sort_values(by=["PMS", "COBI"], ascending=False)
            rows_html = ""
            for i, row in enumerate(df_trades.to_dict(orient="records"), 1):
                is_buy = row["Strategy"] == "BUY"
                
                # Action Badge Styling
                strat_badge = (
                    "background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676;"
                    if is_buy
                    else "background: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid #FF5252;"
                )

                # Flow Shock Badge Styling
                if row["Flow Shock"] == "DEMAND SURGE":
                    flow_badge = "background: rgba(0, 230, 118, 0.12); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.4);"
                elif row["Flow Shock"] == "SUPPLY SURGE":
                    flow_badge = "background: rgba(255, 82, 82, 0.12); color: #FF5252; border: 1px solid rgba(255, 82, 82, 0.4);"
                else:
                    flow_badge = "background: rgba(143, 156, 169, 0.1); color: #8F9CA9; border: 1px solid #333842;"

                bar_color = "#00E676" if row["Flow Pressure"] >= 50 else "#FF5252"

                rows_html += f"""
                <tr style="border-bottom: 1px solid #1E222D; transition: background 0.15s ease;" onmouseover="this.style.background='#161B26'" onmouseout="this.style.background='transparent'">
                    <td style="padding: 6px 8px; color: #5B6577; font-weight: 600; text-align: center; white-space: nowrap;">{i}</td>
                    <td style="padding: 6px 10px; font-weight: 700; color: #FFFFFF; text-align: left; white-space: nowrap; letter-spacing: 0.3px;">{row['Symbol']}</td>
                    <td style="padding: 6px 8px; font-weight: 600; color: #E1E7F5; text-align: right; white-space: nowrap;">{row['LTP']}</td>
                    <td style="padding: 6px 8px; color: #788293; text-align: right; white-space: nowrap;">{row['VWAP']}</td>
                    <td style="padding: 6px 8px; font-weight: 600; text-align: right; white-space: nowrap; color: {'#00E676' if row['PMS'] > 0 else '#FF5252'};">{row['PMS']:.4f}</td>
                    <td style="padding: 6px 8px; font-weight: 600; text-align: right; white-space: nowrap; color: {'#00E676' if row['COBI'] > 0 else '#FF5252'};">{row['COBI']:.4f}</td>
                    <td style="padding: 6px 8px; text-align: center; white-space: nowrap;">
                        <div style="display: inline-flex; align-items: center; gap: 6px;">
                            <div style="width: 34px; height: 4px; background-color: #242936; border-radius: 2px; overflow: hidden;">
                                <div style="width: {row['Flow Pressure']}%; height: 100%; background: {bar_color};"></div>
                            </div>
                            <span style="color: #DFE5F2; font-size: 11px; font-weight: 600;">{row['Flow Pressure']}%</span>
                        </div>
                    </td>
                    <td style="padding: 6px 8px; color: #C5CBD8; font-weight: 600; text-align: center; white-space: nowrap;">{row['Vol Acc']}</td>
                    <td style="padding: 6px 8px; text-align: center; white-space: nowrap;">
                        <span style="padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 10px; display: inline-block; line-height: 1.2; {flow_badge}">
                            {row['Flow Shock']}
                        </span>
                    </td>
                    <td style="padding: 6px 8px; color: #388BFD; font-weight: 600; font-size: 11px; text-align: center; white-space: nowrap;">{row['SMC Structure']}</td>
                    <td style="padding: 6px 8px; color: #A371F7; font-weight: 600; font-size: 11px; text-align: center; white-space: nowrap;">{row['Market Pulse']}</td>
                    <td style="padding: 6px 8px; font-weight: 700; color: #00E676; text-align: right; white-space: nowrap;">{row['Entry']}</td>
                    <td style="padding: 6px 8px; font-weight: 700; color: #FF5252; text-align: right; white-space: nowrap;">{row['SL']}</td>
                    <td style="padding: 6px 8px; font-weight: 700; color: #388BFD; text-align: right; white-space: nowrap;">{row['Target']}</td>
                    <td style="padding: 6px 8px; text-align: center; white-space: nowrap;">
                        <span style="padding: 2px 8px; border-radius: 3px; font-weight: 800; font-size: 10.5px; display: inline-block; line-height: 1.2; letter-spacing: 0.3px; {strat_badge}">
                            {row['Plan']}
                        </span>
                    </td>
                </tr>
                """

            custom_table = f"""
            <div style="background-color: #0E1118; padding: 10px 12px; border-radius: 8px; border: 1px solid #1E222D; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; width: 100%; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #1E222D; padding-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 8px; height: 8px; background: #00E676; border-radius: 50%; box-shadow: 0 0 6px #00E676;"></span>
                        <h3 style="margin: 0; color: #FFFFFF; font-size: 13.5px; font-weight: 700; letter-spacing: 0.4px;">INSTITUTIONAL SMC & ORDER FLOW RADAR</h3>
                    </div>
                    <span style="color: #00E676; font-size: 11.5px; font-weight: 600;">● LIVE ({current_time} IST) &nbsp;|&nbsp; <span style="color: #788293;">Nifty: {nifty_pct_chg}% &nbsp;|&nbsp; ₹300-₹600</span></span>
                </div>
                <div style="overflow-x: auto; width: 100%;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 11.5px; line-height: 1.2;">
                        <thead>
                            <tr style="border-bottom: 1.5px solid #232733; color: #6C7688; text-transform: uppercase; font-size: 9.5px; letter-spacing: 0.6px;">
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">#</th>
                                <th style="padding: 6px 10px; text-align: left; white-space: nowrap;">Symbol</th>
                                <th style="padding: 6px 8px; text-align: right; white-space: nowrap;">LTP</th>
                                <th style="padding: 6px 8px; text-align: right; white-space: nowrap;">VWAP</th>
                                <th style="padding: 6px 8px; text-align: right; white-space: nowrap;">PMS</th>
                                <th style="padding: 6px 8px; text-align: right; white-space: nowrap;">COBI</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Buy Flow</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Vol Acc</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Flow Shock</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">SMC Structure</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Market Pulse</th>
                                <th style="padding: 6px 8px; text-align: right; color: #00E676; white-space: nowrap;">Entry</th>
                                <th style="padding: 6px 8px; text-align: right; color: #FF5252; white-space: nowrap;">SL</th>
                                <th style="padding: 6px 8px; text-align: right; color: #388BFD; white-space: nowrap;">Target</th>
                                <th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
            """
            st.markdown(custom_table, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="background-color: #0E1118; padding: 16px; border-radius: 8px; border: 1px solid #1E222D; color: #DFE5F2; font-family: sans-serif; font-size: 13px;">
                    [{current_time} IST] 🔍 Market scanning ₹300-₹600 universe... No clean setups right now.
                </div>
                """,
                unsafe_allow_html=True
            )

    time.sleep(REFRESH_SECONDS)
