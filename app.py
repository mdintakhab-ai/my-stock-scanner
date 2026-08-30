import io
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# STREAMLIT MOBILE-FIRST APP CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional SMC & Order Flow Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Responsive CSS for Mobile Screens & Full-width table display
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    return pd.Series(tr).rolling(length, min_periods=1).mean().values


def calculate_rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
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
    return (
        round(clamp((bid_vol - ask_vol) / total if total > 0 else 0.0, -1.0, 1.0), 4)
    )


# ------------------------------------------------------------------
# 2. MICROSTRUCTURE FLOW & SURGE DETECTOR
# ------------------------------------------------------------------
def detect_flow_shock(
    close_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray, vol_arr: np.ndarray
) -> Tuple[float, float, str, float, float]:
    if len(close_arr) < 5:
        return 50.0, 1.0, "NEUTRAL", 0.0, 0.0

    price_range = np.maximum(high_arr - low_arr, 1e-5)
    buy_share = (close_arr - low_arr) / price_range
    buy_vol_arr = vol_arr * buy_share
    sell_vol_arr = vol_arr * (1.0 - buy_share)

    recent_buy = float(np.sum(buy_vol_arr[-5:]))
    recent_total = float(np.sum(vol_arr[-5:])) + 1e-5
    buy_pressure_pct = round((recent_buy / recent_total) * 100, 1)

    denom = (
        float(np.mean(vol_arr[-20:]))
        if len(vol_arr) >= 20
        else float(np.mean(vol_arr))
    )
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

    vol_sma14 = pd.Series(vol).rolling(14, min_periods=1).mean().values

    for i in range(pivot_len * 2, n):
        bull_bars += 1
        bear_bars += 1

        # Causal Pivot High & Low Detection (No forward-looking leak)
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

        # Sweeps, MSS & FVG Checks
        bull_sweep = not np.isnan(last_low) and low[i] < last_low and close[i] > last_low
        bull_mss = not np.isnan(last_high) and close[i] > last_high and close[i - 1] <= last_high
        bull_fvg = (low[i] - high[i - 2]) > (atr[i] * fvg_min_atr) if i >= 2 else False

        bear_sweep = not np.isnan(last_high) and high[i] > last_high and close[i] < last_high
        bear_mss = not np.isnan(last_low) and close[i] < last_low and close[i - 1] >= last_low
        bear_fvg = (low[i - 2] - high[i]) > (atr[i] * fvg_min_atr) if i >= 2 else False

        # State Transitions
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

    ema_up = ema13[-1] > ema13[-2]
    ema_dn = ema13[-1] < ema13[-2]

    bull_confirm = (bull_state >= 2) and (close[-1] > ema13[-1]) and ema_up and (rsi[-1] >= 50.0)
    bear_confirm = (bear_state >= 2) and (close[-1] < ema13[-1]) and ema_dn and (rsi[-1] <= 50.0)

    signal = "BUY" if bull_confirm else ("SELL" if bear_confirm else "NONE")
    status_desc = "BULL MSS+FVG" if bull_state == 4 else ("BULL MSS" if bull_state == 2 else ("BEAR MSS+FVG" if bear_state == 4 else ("BEAR MSS" if bear_state == 2 else "PULLBACK")))

    return {
        "bull_state": bull_state,
        "bear_state": bear_state,
        "smc_signal": signal,
        "status_desc": status_desc
    }


# ------------------------------------------------------------------
# 4. INTERMARKET PULSE ENGINE
# ------------------------------------------------------------------
SECTOR_LEADER_MAP = {
    "TATAMOTORS": "MARUTI.NS", "M&M": "MARUTI.NS", "ASHOKLEY": "MARUTI.NS",
    "TATASTEEL": "HINDALCO.NS", "SAIL": "HINDALCO.NS", "JINDALSTEL": "HINDALCO.NS",
    "SBIN": "HDFCBANK.NS", "ICICIBANK": "HDFCBANK.NS", "PNB": "HDFCBANK.NS", "CANBK": "HDFCBANK.NS", "FEDERALBNK": "HDFCBANK.NS",
    "INFY": "TCS.NS", "WIPRO": "TCS.NS", "HCLTECH": "TCS.NS", "TECHM": "TCS.NS",
    "CIPLA": "SUNPHARMA.NS", "DRREDDY": "SUNPHARMA.NS", "LUPIN": "SUNPHARMA.NS",
    "TATAPOWER": "RELIANCE.NS", "ONGC": "RELIANCE.NS", "BPCL": "RELIANCE.NS", "IOC": "RELIANCE.NS", "VEDL": "HINDALCO.NS"
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
# 5. UNIVERSE FETCHER & BENCHMARK INIT
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
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


# ------------------------------------------------------------------
# 6. MAIN STREAMLIT EXECUTION ENGINE
# ------------------------------------------------------------------
symbols = fetch_nse_symbols()

# App Header with Quick Refresh
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown("### ⚡ Institutional SMC & Order Flow Radar")
with col_btn:
    refresh_clicked = st.button("🔄 Scan Market", use_container_width=True)

status_container = st.empty()
status_container.info(f"Loaded {len(symbols)} NSE stocks. Syncing benchmarks...")

# Pre-fetch Benchmark & Leaders Performance
unique_leaders = list(set(SECTOR_LEADER_MAP.values()))
benchmarks_to_pull = ["^NSEI"] + unique_leaders
leader_perf_map: Dict[str, float] = {}
nifty_pct_chg = 0.0

try:
    bench_data = yf.download(
        benchmarks_to_pull, period="2d", interval="5m", group_by="ticker", progress=False
    )
    # Parse Nifty
    nifty_raw = bench_data["^NSEI"] if isinstance(bench_data.columns, pd.MultiIndex) and "^NSEI" in bench_data.columns.levels[0] else bench_data
    n_bars = min(75, len(nifty_raw))
    nifty_open = float(nifty_raw["Open"].dropna().iloc[-n_bars])
    nifty_close = float(nifty_raw["Close"].dropna().iloc[-1])
    nifty_pct_chg = round(((nifty_close - nifty_open) / nifty_open) * 100, 2)

    # Parse Leaders
    for l_ticker in unique_leaders:
        try:
            l_df = bench_data[l_ticker].dropna() if isinstance(bench_data.columns, pd.MultiIndex) else pd.DataFrame()
            if not l_df.empty:
                l_bars = min(75, len(l_df))
                l_open = float(l_df["Open"].iloc[-l_bars])
                l_close = float(l_df["Close"].iloc[-1])
                leader_perf_map[l_ticker] = round(((l_close - l_open) / l_open) * 100, 2)
        except Exception:
            leader_perf_map[l_ticker] = nifty_pct_chg
except Exception:
    pass

# ------------------------------------------------------------------
# 7. SCANNER & BATCH PROCESSING (₹300 - ₹600)
# ------------------------------------------------------------------
BATCH_SIZE = 100
results = []
ticker_list = [f"{sym}.NS" for sym in symbols]

progress_bar = st.progress(0)

for i in range(0, len(ticker_list), BATCH_SIZE):
    progress_bar.progress(min((i + BATCH_SIZE) / len(ticker_list), 1.0))
    batch_tickers = ticker_list[i : i + BATCH_SIZE]
    batch_symbols = symbols[i : i + BATCH_SIZE]

    try:
        data = yf.download(
            tickers=batch_tickers,
            period="5d",
            interval="5m",
            group_by="ticker",
            threads=True,
            progress=False,
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

            # Price Filter strictly ₹300 - ₹600
            if not (300.0 <= current_price <= 600.0):
                continue

            vwap_val = calculate_intraday_vwap(high, low, close, vol)
            ema13 = calculate_ema(close, 13)
            atr = calculate_atr(high, low, close, 14)
            rsi = calculate_rsi(close, 14)
            atr_val = round(float(atr[-1]), 2) if atr[-1] > 0 else 1.0

            bars_today = min(len(close), 75)
            session_open = float(open_p[-bars_today])
            prev_ref = float(close[-bars_today]) if len(close) >= bars_today else float(close[0])
            gap_pts = current_price - prev_ref
            stock_pct_chg = round(((current_price - session_open) / session_open) * 100, 2)

            (
                buy_pressure_pct,
                vol_acc,
                flow_shock,
                buy_vol_10,
                sell_vol_10,
            ) = detect_flow_shock(close, high, low, vol)

            recent_vol = float(np.sum(vol[-10:]))
            avg_vol = float(np.mean(vol[-bars_today:])) * 10 if len(vol) >= bars_today else float(np.mean(vol)) * 10
            vol_ratio = (recent_vol / avg_vol) - 1.0 if avg_vol > 0 else 0.0

            cobi_val = calculate_cobi(buy_vol_10, sell_vol_10)
            pms_val = calculate_pms(
                vol_ratio,
                cobi_val,
                gap_pts / atr_val,
                (gap_pts / prev_ref) * 10 if prev_ref > 0 else 0.0,
            )

            smc_result = run_smc_state_machine(high, low, close, vol, atr, rsi, ema13)

            # Intermarket Leader Data
            leader_ticker = SECTOR_LEADER_MAP.get(sym, None)
            leader_chg = leader_perf_map.get(leader_ticker, nifty_pct_chg)
            market_state, intermarket_dir = evaluate_intermarket_action(
                stock_pct_chg, leader_chg, nifty_pct_chg, current_price, vwap_val, float(open_p[-1])
            )

            # High-Conviction Synergy Execution
            final_signal = "REJECT"
            if market_state not in ["SELLERS OVERTAKE", "FAKE RALLY", "FAKE DROP"]:
                if (
                    pms_val >= 0.28
                    and cobi_val >= 0.25
                    and current_price >= vwap_val
                    and buy_pressure_pct >= 58.0
                    and vol_acc >= 1.12
                    and (smc_result["smc_signal"] == "BUY" or rsi[-1] >= 50.0)
                    and intermarket_dir in ["BUY", "NEUTRAL"]
                ):
                    final_signal = "BUY"
                elif (
                    pms_val <= -0.28
                    and cobi_val <= -0.25
                    and current_price <= vwap_val
                    and buy_pressure_pct <= 42.0
                    and vol_acc >= 1.12
                    and (smc_result["smc_signal"] == "SELL" or rsi[-1] <= 50.0)
                    and intermarket_dir in ["SELL", "NEUTRAL"]
                ):
                    final_signal = "SELL"

            if final_signal in ["BUY", "SELL"]:
                results.append(
                    {
                        "Symbol": sym,
                        "LTP": f"₹{current_price:,.2f}",
                        "VWAP": f"₹{vwap_val:,.2f}",
                        "PMS": pms_val,
                        "COBI": cobi_val,
                        "Flow Pressure": buy_pressure_pct,
                        "Vol Acc": f"{vol_acc}x",
                        "Flow Shock": flow_shock,
                        "SMC Structure": smc_result["status_desc"],
                        "Market Pulse": market_state,
                        "Strategy": final_signal,
                    }
                )
        except Exception:
            continue

progress_bar.empty()
status_container.empty()

# ------------------------------------------------------------------
# 8. HIGH-CONVICTION DASHBOARD DISPLAY
# ------------------------------------------------------------------
df_trades = pd.DataFrame(results)

if not df_trades.empty:
    df_trades = df_trades.sort_values(by=["PMS", "COBI"], ascending=False)
    df_trades.insert(0, "S.No", np.arange(1, len(df_trades) + 1))

    rows_html = ""
    for _, row in df_trades.iterrows():
        is_buy = row["Strategy"] == "BUY"
        strategy_badge = (
            "background: linear-gradient(135deg, #00B074 0%, #008f5d 100%); color: #ffffff; box-shadow: 0 0 10px rgba(0, 176, 116, 0.4);"
            if is_buy
            else "background: linear-gradient(135deg, #FF3B30 0%, #c41e15 100%); color: #ffffff; box-shadow: 0 0 10px rgba(255, 59, 48, 0.4);"
        )

        if row["Flow Shock"] == "DEMAND SURGE":
            flow_badge = "background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676;"
        elif row["Flow Shock"] == "SUPPLY SURGE":
            flow_badge = "background: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid #FF5252;"
        else:
            flow_badge = "background: rgba(143, 156, 169, 0.15); color: #8F9CA9; border: 1px solid #454D5A;"

        bar_color = "#00E676" if row["Flow Pressure"] >= 50 else "#FF5252"

        rows_html += f"""
        <tr style="border-bottom: 1px solid #1E222D; transition: background 0.2s;" onmouseover="this.style.background='#1A1F2C'" onmouseout="this.style.background='transparent'">
            <td style="padding: 12px 14px; color: #8F9CA9; text-align: center; font-weight: 600;">{row['S.No']}</td>
            <td style="padding: 12px 14px; font-weight: 700; color: #FFFFFF; letter-spacing: 0.5px;">{row['Symbol']}</td>
            <td style="padding: 12px 14px; text-align: right; color: #F0F3FA; font-weight: 600;">{row['LTP']}</td>
            <td style="padding: 12px 14px; text-align: right; color: #8F9CA9;">{row['VWAP']}</td>
            <td style="padding: 12px 14px; text-align: right; font-weight: 600; color: {'#00E676' if row['PMS'] > 0 else '#FF5252'};">{row['PMS']:.4f}</td>
            <td style="padding: 12px 14px; text-align: right; font-weight: 600; color: {'#00E676' if row['COBI'] > 0 else '#FF5252'};">{row['COBI']:.4f}</td>
            <td style="padding: 12px 14px; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                    <div style="flex: 1; max-width: 50px; height: 6px; background-color: #2A2E39; border-radius: 3px; overflow: hidden;">
                        <div style="width: {row['Flow Pressure']}%; height: 100%; background: {bar_color};"></div>
                    </div>
                    <span style="color: #F0F3FA; font-size: 12px; font-weight: 600;">{row['Flow Pressure']}%</span>
                </div>
            </td>
            <td style="padding: 12px 14px; text-align: center; color: #D1D4DC; font-weight: 600;">{row['Vol Acc']}</td>
            <td style="padding: 12px 14px; text-align: center;">
                <span style="padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; letter-spacing: 0.5px; display: inline-block; {flow_badge}">
                    {row['Flow Shock']}
                </span>
            </td>
            <td style="padding: 12px 14px; text-align: center; color: #388BFD; font-weight: 600; font-size: 12px;">{row['SMC Structure']}</td>
            <td style="padding: 12px 14px; text-align: center; color: #A371F7; font-weight: 600; font-size: 12px;">{row['Market Pulse']}</td>
            <td style="padding: 12px 14px; text-align: center;">
                <span style="padding: 4px 14px; border-radius: 6px; font-weight: 800; font-size: 12px; letter-spacing: 0.8px; display: inline-block; {strategy_badge}">
                    {row['Strategy']}
                </span>
            </td>
        </tr>
        """

    custom_table = f"""
    <div style="background-color: #131722; padding: 20px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; border-bottom: 1px solid #2A2E39; padding-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 10px; height: 10px; background: #00E676; border-radius: 50%; box-shadow: 0 0 8px #00E676;"></span>
                <h3 style="margin: 0; color: #FFFFFF; font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">INSTITUTIONAL SMC & ORDER FLOW RADAR</h3>
            </div>
            <span style="color: #8F9CA9; font-size: 12px;">Universe: ₹300 - ₹600 | 5-Min Timeframe | Nifty Pulse: {nifty_pct_chg}%</span>
        </div>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
                <thead>
                    <tr style="border-bottom: 2px solid #2A2E39; color: #8F9CA9; text-transform: uppercase; font-size: 11px; letter-spacing: 0.8px;">
                        <th style="padding: 10px 14px; text-align: center;">#</th>
                        <th style="padding: 10px 14px;">Symbol</th>
                        <th style="padding: 10px 14px; text-align: right;">LTP</th>
                        <th style="padding: 10px 14px; text-align: right;">VWAP</th>
                        <th style="padding: 10px 14px; text-align: right;">PMS Score</th>
                        <th style="padding: 10px 14px; text-align: right;">COBI Flow</th>
                        <th style="padding: 10px 14px; text-align: center;">Buy Flow</th>
                        <th style="padding: 10px 14px; text-align: center;">Vol Acc</th>
                        <th style="padding: 10px 14px; text-align: center;">Flow Shock</th>
                        <th style="padding: 10px 14px; text-align: center;">SMC Structure</th>
                        <th style="padding: 10px 14px; text-align: center;">Market Pulse</th>
                        <th style="padding: 10px 14px; text-align: center;">Execution</th>
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
    st.warning("⚠️ No high-conviction setups detected across SMC + Flow filters. Market is consolidating.")
