# =============================================================================
# INSTITUTIONAL QUANT ORDER FLOW & SMC PRO ULTRA-LIGHTSCANNER (STREAMLIT MOBILE UI)
# Seamless Live UI | Zero-Flicker Display | High-Speed Vectorized Engine
# =============================================================================

import io
import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import streamlit.components.v1 as components

# Suppress all non-critical warnings
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"

# Streamlit Page Configuration for Mobile / Responsive View
st.set_page_config(
    page_title="Quant SMC Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# GLOBAL SETTINGS & STRICT FILTERS
# -----------------------------------------------------------------------------
DISPLAY_TIMEZONE = 'Asia/Kolkata'
REFRESH_SECONDS = 3           # Live update interval
MIN_PRICE = 300.0             # Price filter lower limit
MAX_PRICE = 600.0             # Price filter upper limit
TOP_N_DISPLAY = 20            # Top stocks to display on screen
MIN_SAME_DIRECTION_TF = 3     # Minimum 3 timeframes aligned out of 4 (75%)
MIN_RVOL_FILTER = 1.5         # Minimum Volume expansion required
MIN_CONVICTION_SCORE = 70     # High conviction threshold
TIMEFRAMES_LIST = ["1m", "3m", "5m", "15m"]

# Persistent HTTP Session
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Referer': 'https://www.nseindia.com/'
})

# -----------------------------------------------------------------------------
# 1. DYNAMIC NSE UNIVERSE FETCHER
# -----------------------------------------------------------------------------
def fetch_dynamic_universe():
    tickers = set()
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv"
    ]

    try:
        SESSION.get("https://www.nseindia.com/", timeout=5)
    except Exception:
        pass

    for url in urls:
        try:
            res = SESSION.get(url, timeout=5)
            if res.status_code == 200 and len(res.text) > 100:
                df = pd.read_csv(io.StringIO(res.text))
                sym_col = [c for c in df.columns if 'symbol' in c.lower()]
                if sym_col:
                    for s in df[sym_col[0]].dropna().unique():
                        clean = str(s).strip().upper()
                        if clean and not clean.startswith("DUMMY") and clean != "SYMBOL":
                            tickers.add(f"{clean}.NS")
        except Exception:
            continue

    return sorted(list(tickers))

# -----------------------------------------------------------------------------
# 2. VECTORIZED QUANT & SMC CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_full_quant_smc(c, h, l, o, v):
    if len(c) < 20:
        return None

    # Delta Volume & Buy Pressure Ratio
    hl_range = np.where((h - l) == 0, 1e-6, h - l)
    delta_ratio = np.clip(((c - l) - (h - c)) / hl_range, -1.0, 1.0)
    delta_vol = v * delta_ratio
    buy_vol = (v + delta_vol) / 2.0
    delta_pct_bar = np.clip(buy_vol / (v + 1e-6), 0.0, 1.0)

    # Immediate Flow Pressure
    recent_buy = float(np.sum(buy_vol[-5:]))
    recent_total = float(np.sum(v[-5:])) + 1e-6
    flow_pressure_pct = round((recent_buy / recent_total) * 100, 1)

    # Order Book Imbalance (OBI) & COBI Z-Score
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    wick_denom = np.where((lower_wick + upper_wick) == 0, 1e-6, lower_wick + upper_wick)
    obi = (lower_wick - upper_wick) / wick_denom

    lookback_cobi = min(15, len(obi))
    cobi_raw = pd.Series(obi).rolling(lookback_cobi, min_periods=3).sum().to_numpy()
    cobi_std = np.nanstd(cobi_raw[-lookback_cobi:])
    cobi_mean = np.nanmean(cobi_raw[-lookback_cobi:])
    cobi_last = cobi_raw[-1] if np.isfinite(cobi_raw[-1]) else 0.0
    cobi_z = float((cobi_last - cobi_mean) / (cobi_std if cobi_std > 1e-6 else 1.0))

    # Relative Volume (RVOL)
    lookback_v = min(20, len(v))
    avg_vol_20 = np.mean(v[-lookback_v:]) + 1e-6
    rvol = float(v[-1] / avg_vol_20)

    # Session Anchored VWAP
    typical_price = (h + l + c) / 3.0
    cum_pv = np.cumsum(typical_price * v)
    cum_v = np.cumsum(v)
    vwap = cum_pv / (cum_v + 1e-6)
    vwap_val = float(vwap[-1])

    # ATR & PMS Calculation
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_series = pd.Series(tr).ewm(alpha=1.0/14, adjust=False).mean()
    atr_val = atr_series.iloc[-1] if not atr_series.empty else (h[-1] - l[-1])
    atr_val = 1e-6 if (atr_val is None or atr_val <= 0 or np.isnan(atr_val)) else float(atr_val)

    pms_val = float(((c[-1] - vwap_val) / atr_val) * min(rvol, 3.0) * (delta_pct_bar[-1] - 0.5) * 10.0)

    # Eliminates NaN or zero volume corrupt records
    if np.isnan(pms_val) or np.isnan(rvol) or np.isnan(cobi_z):
        return None

    # Delta Price Change %
    delta_price_pct = float((c[-1] - c[-2]) / c[-2] * 100) if len(c) > 1 else 0.0

    # RSI 14
    delta_c = pd.Series(c).diff()
    gain = delta_c.clip(lower=0.0)
    loss = -delta_c.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_arr = np.nan_to_num((100.0 - (100.0 / (1.0 + rs))).values, nan=50.0)
    rsi_val = float(rsi_arr[-1])

    # EMA 13 Trend
    ema13 = pd.Series(c).ewm(span=13, adjust=False).mean().to_numpy()
    ema_trend = "GREEN" if (ema13[-1] >= ema13[-2] and c[-1] >= ema13[-1]) else (
        "RED" if (ema13[-1] <= ema13[-2] and c[-1] <= ema13[-1]) else "YELLOW"
    )

    # High-Conviction Scoring Logic
    price = float(c[-1])
    bull_score = 0
    if price > vwap_val: bull_score += 20
    if delta_pct_bar[-1] >= 0.60: bull_score += 20
    if obi[-1] >= 0.20: bull_score += 15
    if cobi_z >= 0.40: bull_score += 15
    if rvol >= 1.5: bull_score += 15
    if rsi_val > 50: bull_score += 15

    bear_score = 0
    if price < vwap_val: bear_score += 20
    if delta_pct_bar[-1] <= 0.40: bear_score += 20
    if obi[-1] <= -0.20: bear_score += 15
    if cobi_z <= -0.40: bear_score += 15
    if rvol >= 1.5: bear_score += 15
    if rsi_val < 50: bear_score += 15

    conviction = max(bull_score, bear_score)

    # Strategy Assignment
    strategy = "NEUTRAL"
    smc_structure = "WAITING"
    if bull_score >= 60 and price > vwap_val and ema_trend in ["GREEN", "YELLOW"]:
        strategy = "BUY"
        smc_structure = "MSS CONFIRMED"
    elif bear_score >= 60 and price < vwap_val and ema_trend in ["RED", "YELLOW"]:
        strategy = "SELL"
        smc_structure = "BEAR MSS"

    return {
        'LTP': price,
        'VWAP': vwap_val,
        'Delta_Pct': delta_price_pct,
        'RVOL': rvol,
        'PMS': pms_val,
        'OBI': float(obi[-1]),
        'COBI': cobi_z,
        'Flow_Pressure': flow_pressure_pct,
        'SMC_Structure': smc_structure,
        'Strategy': strategy,
        'Conviction': conviction,
        'EMA_Color': ema_trend
    }

# -----------------------------------------------------------------------------
# 3. RESAMPLING & MULTI-TIMEFRAME EVALUATOR
# -----------------------------------------------------------------------------
def get_tf_signal_from_resample(df_1m, timeframe_str):
    try:
        resample_rule = timeframe_str.replace("m", "min")
        df_res = df_1m.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        if len(df_res) < 3:
            return "NEUTRAL"

        c = df_res['Close'].values.astype(float)
        ema13 = pd.Series(c).ewm(span=13, adjust=False).mean().values
        if c[-1] > ema13[-1]:
            return "UP"
        elif c[-1] < ema13[-1]:
            return "DOWN"
    except Exception:
        pass
    return "NEUTRAL"

def process_single_symbol(item, full_downloaded_data):
    ticker, clean_sym = item
    try:
        if isinstance(full_downloaded_data.columns, pd.MultiIndex):
            if ticker not in full_downloaded_data.columns.levels[0]:
                return None
            df_stock = full_downloaded_data[ticker].dropna(how='all')
        else:
            df_stock = full_downloaded_data.dropna(how='all')

        if df_stock.empty or len(df_stock) < 20:
            return None

        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in df_stock.columns for col in required_cols):
            return None

        c = df_stock['Close'].values.astype(float)
        h = df_stock['High'].values.astype(float)
        l = df_stock['Low'].values.astype(float)
        o = df_stock['Open'].values.astype(float)
        v = df_stock['Volume'].values.astype(float)

        current_price = c[-1]

        # Price Range Filter
        if not (MIN_PRICE <= current_price <= MAX_PRICE):
            return None

        m = calculate_full_quant_smc(c, h, l, o, v)
        if not m or m['Strategy'] == "NEUTRAL":
            return None

        # Filter low volume and zero-activity stocks
        if m['RVOL'] < MIN_RVOL_FILTER or np.isnan(m['PMS']):
            return None

        # Filter low conviction stocks
        if m['Conviction'] < MIN_CONVICTION_SCORE:
            return None

        # Multi-Timeframe Alignment
        mtf_dots = {}
        target_dir = "UP" if m['Strategy'] == "BUY" else "DOWN"
        matched_count = 0

        for tf in TIMEFRAMES_LIST:
            sig = get_tf_signal_from_resample(df_stock, tf)
            mtf_dots[tf] = sig
            if sig == target_dir:
                matched_count += 1

        # Strict Multi-Timeframe Filter
        if matched_count < MIN_SAME_DIRECTION_TF:
            return None

        return {
            'Symbol': clean_sym,
            'LTP': m['LTP'],
            'VWAP': m['VWAP'],
            'Delta_Pct': m['Delta_Pct'],
            'RVOL': m['RVOL'],
            'PMS': m['PMS'],
            'COBI': m['COBI'],
            'Flow_Pressure': m['Flow_Pressure'],
            'SMC_Structure': m['SMC_Structure'],
            'Strategy': m['Strategy'],
            'Conviction': m['Conviction'],
            'MTF_Dots': mtf_dots,
            'MTF_Match_Count': matched_count
        }
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 4. HTML TERMINAL RENDERER (RESPONSIVE VIEW)
# -----------------------------------------------------------------------------
def render_dot(tf_status):
    if tf_status == "UP":
        return '<span style="color:#00E676; font-size:14px;">●</span>'
    elif tf_status == "DOWN":
        return '<span style="color:#FF5252; font-size:14px;">●</span>'
    return '<span style="color:#555E6D; font-size:14px;">●</span>'

def build_terminal_html(results, scan_time, total_active_found):
    sorted_results = sorted(results, key=lambda x: (x['Conviction'], x['MTF_Match_Count'], x['PMS']), reverse=True)[:TOP_N_DISPLAY]

    rows_html = ""
    for i, r in enumerate(sorted_results, 1):
        is_buy = r['Strategy'] == "BUY"
        strat_badge = (
            "background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1.5px solid #00E676;"
            if is_buy
            else "background: rgba(255, 82, 82, 0.2); color: #FF5252; border: 1.5px solid #FF5252;"
        )

        dots_html = f"""<div style="display:flex;gap:4px;justify-content:center;align-items:center;">{''.join([render_dot(r['MTF_Dots'][tf]) for tf in TIMEFRAMES_LIST])}</div>"""

        delta_color = "#00E676" if r['Delta_Pct'] >= 0 else "#FF5252"
        pms_color = "#00E676" if r['PMS'] > 0 else "#FF5252"
        cobi_color = "#00E676" if r['COBI'] > 0 else "#FF5252"

        rows_html += f"""
        <tr style="border-bottom: 1px solid #232733; height: 38px;">
            <td style="padding: 6px; color: #8B949E; text-align: center; font-size: 12px;">{i}</td>
            <td style="padding: 6px 10px; font-weight: 700; color: #FFFFFF; font-size: 13px; text-align: left; white-space: nowrap; position: sticky; left: 0; background: #11151C;">{r['Symbol']}</td>
            <td style="padding: 6px; font-weight: 700; color: #FFFFFF; font-size: 13px; text-align: right; white-space: nowrap;">₹{r['LTP']:,.2f}</td>
            <td style="padding: 6px; color: #A3B1C6; font-size: 12px; text-align: right; white-space: nowrap;">₹{r['VWAP']:,.2f}</td>
            <td style="padding: 6px; font-weight: 700; font-size: 13px; text-align: right; color: {delta_color}; white-space: nowrap;">{r['Delta_Pct']:+.2f}%</td>
            <td style="padding: 6px; font-weight: 700; color: #4A9EEB; font-size: 13px; text-align: right; white-space: nowrap;">{r['RVOL']:.2f}x</td>
            <td style="padding: 6px; font-weight: 700; font-size: 13px; text-align: right; color: {pms_color}; white-space: nowrap;">{r['PMS']:.2f}</td>
            <td style="padding: 6px; font-weight: 700; font-size: 13px; text-align: right; color: {cobi_color}; white-space: nowrap;">{r['COBI']:.2f}</td>
            <td style="padding: 6px; text-align: center; font-size: 12px; font-weight: 600; color: #E1E7F5; white-space: nowrap;">{r['Flow_Pressure']}%</td>
            <td style="padding: 6px; color: #4A9EEB; font-weight: 700; font-size: 12px; text-align: center; white-space: nowrap;">{r['SMC_Structure']}</td>
            <td style="padding: 6px; text-align: center; white-space: nowrap;">{dots_html}</td>
            <td style="padding: 6px; text-align: center; white-space: nowrap;">
                <span style="padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 11px; letter-spacing: 0.5px; {strat_badge}">
                    {r['Strategy']}
                </span>
            </td>
        </tr>
        """

    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="12" style="text-align:center; padding:25px; color:#A3B1C6; font-size: 13px; font-weight:600;">
                Filtering market... Only scanning for High-Conviction Setups (rVOL ≥ 1.5x, MTF Alignment ≥ 75%)...
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            background-color: #0E1117;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #FFFFFF;
        }}
        ::-webkit-scrollbar {{
            height: 6px;
            width: 6px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #232733;
            border-radius: 3px;
        }}
    </style>
    </head>
    <body>
    <div style="background-color: #0B0E14; padding: 10px; border-radius: 8px; border: 1px solid #1E222D;">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 6px;">
            <span style="color: #00E676; font-weight: 800; font-size: 13px; letter-spacing: 0.5px;">
                ● QUANT SMC SCANNER (₹{MIN_PRICE:.0f}-₹{MAX_PRICE:.0f}) | {scan_time} IST
            </span>
            <span style="color: #A3B1C6; font-size: 12px; font-weight: 700;">Active Setups: {total_active_found}</span>
        </div>
        <div style="overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 6px;">
            <table style="width: 100%; border-collapse: collapse; text-align: left; background: #11151C; min-width: 800px;">
                <thead>
                    <tr style="border-bottom: 2px solid #232733; color: #8B949E; text-transform: uppercase; font-size: 10px; letter-spacing: 0.6px;">
                        <th style="padding: 8px 6px; text-align: center;">#</th>
                        <th style="padding: 8px 10px; text-align: left; position: sticky; left: 0; background: #11151C;">Symbol</th>
                        <th style="padding: 8px 6px; text-align: right;">LTP</th>
                        <th style="padding: 8px 6px; text-align: right;">VWAP</th>
                        <th style="padding: 8px 6px; text-align: right;">Delta</th>
                        <th style="padding: 8px 6px; text-align: right;">rVOL</th>
                        <th style="padding: 8px 6px; text-align: right;">PMS</th>
                        <th style="padding: 8px 6px; text-align: right;">COBI</th>
                        <th style="padding: 8px 6px; text-align: center;">Flow</th>
                        <th style="padding: 8px 6px; text-align: center;">SMC Struct</th>
                        <th style="padding: 8px 6px; text-align: center;">MTF</th>
                        <th style="padding: 8px 6px; text-align: center;">Action</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    </body>
    </html>
    """
    return html

# -----------------------------------------------------------------------------
# 5. STREAMLIT APP ENGINE
# -----------------------------------------------------------------------------
def main():
    placeholder = st.empty()

    symbols = []
    with st.spinner("Fetching dynamic NSE universe..."):
        while not symbols:
            symbols = fetch_dynamic_universe()
            if not symbols:
                time.sleep(2)

    ticker_map = {sym: sym.replace(".NS", "") for sym in symbols}
    ticker_list = list(ticker_map.keys())

    while True:
        try:
            start_time = time.time()
            scan_time = datetime.now(ZoneInfo(DISPLAY_TIMEZONE)).strftime('%H:%M:%S')

            data = yf.download(
                tickers=ticker_list,
                period="1d",
                interval="1m",
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=8
            )

            if data is None or data.empty:
                time.sleep(1)
                continue

            process_items = [(ticker, clean_sym) for ticker, clean_sym in ticker_map.items()]

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(process_single_symbol, item, data) for item in process_items]
                results = [f.result() for f in futures if f.result() is not None]

            html_table = build_terminal_html(results, scan_time, len(results))
            with placeholder.container():
                components.html(html_table, height=750, scrolling=True)

            elapsed = time.time() - start_time
            sleep_duration = max(0.5, REFRESH_SECONDS - elapsed)
            time.sleep(sleep_duration)

        except Exception:
            time.sleep(2)

if __name__ == "__main__":
    main()
