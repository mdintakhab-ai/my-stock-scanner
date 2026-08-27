# =============================================================================
# MOBILE STREAMLIT ENGINE (PWA-READY FOR ANDROID/IOS BROWSER)
# =============================================================================
import math
import time
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import streamlit as st

# Setup Streamlit Mobile Viewport & Theme
st.set_page_config(
    page_title="Quant 5M Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Silent unwanted internal logs
warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Mobile Optimized CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .metric-card { background-color: rgba(128, 128, 128, 0.1); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px; }
    .badge-buy { background-color: #2e7d32; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-sell { background-color: #c62828; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-neutral { background-color: #616161; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    table { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
WATCHLIST = [
    "ETERNAL", "BLEL", "REDINGTON", "HSCL", "CUPID", "ZAGGLE",
    "PAYTM", "CHENNPETRO", "MCX", "ASTERDM", "SBIN", "TATAPOWER", "IDFCFIRSTB",
    "FEDERALBNK", "IOC", "ONGC", "GAIL", "BHEL", "NATIONALUM", "EXIDEIND",
    "PFC", "RECLTD", "SAIL", "NMDC", "GMRP&UI", "HINDCOPPER", "TATASTEEL",
    "HINDALCO", "COALINDIA", "LT", "PRECWIRE", "RELAXO", "DELHIVERY"
]

PRICE_FILTER_ACTIVE = False
PRICE_MIN, PRICE_MAX = 300, 500
VWAP_BUFFER_PCT = 0.15

# ==================== TECHNICAL MATH HELPERS ====================
def clean_num(x):
    if x is None: return 0.0
    try: return float(str(x).replace(",", "").replace("%", ""))
    except (ValueError, TypeError): return 0.0

def rma(series, length):
    return series.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = rma(pd.Series(gain, index=series.index), length)
    avg_loss = rma(pd.Series(loss, index=series.index), length)
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(tr, length)

def create_robust_nse_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/get-quotes/equity?symbol=SBIN",
        "Connection": "keep-alive"
    })
    try:
        session.get("https://www.nseindia.com", timeout=3)
    except Exception:
        pass
    return session

def fetch_live_orderbook(session, symbol):
    try:
        formatted_sym = requests.utils.quote(symbol)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={formatted_sym}"
        res = session.get(url, timeout=2.0)
        if res.status_code == 200:
            data = res.json()
            market_dept = data.get("marketDeptOrderBook", {})
            total_book = market_dept.get("totalOrderBook", {})
            buy_q = clean_num(total_book.get("totalBuyQuantity"))
            sell_q = clean_num(total_book.get("totalSellQuantity"))

            if buy_q == 0 and sell_q == 0:
                price_info = data.get("priceInfo", {})
                buy_q = clean_num(price_info.get("totalBuyQuantity"))
                sell_q = clean_num(price_info.get("totalSellQuantity"))

            if buy_q > 0 or sell_q > 0:
                bs_ratio = round(buy_q / (sell_q if sell_q > 0 else 1), 2)
                imbalance = int(buy_q - sell_q)
                return bs_ratio, imbalance, True
    except Exception:
        pass
    return 1.0, 0, False

def fetch_benchmark_data():
    nifty_trend, idx_pd_close, idx_pd_close20, vix_close, vix_sma20 = "NEUTRAL", 0.0, 0.0, 15.0, 15.0
    try:
        nifty = yf.download("^NSEI", period="35d", interval="1d", progress=False, timeout=5)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        if len(nifty) >= 21:
            idx_pd_close = float(nifty["Close"].iloc[-2])
            idx_pd_close20 = float(nifty["Close"].iloc[-21])
            chg = ((nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2]) / nifty["Close"].iloc[-2]) * 100
            nifty_trend = "BULLISH" if chg > 0.1 else ("BEARISH" if chg < -0.1 else "NEUTRAL")
    except Exception:
        pass

    try:
        vix = yf.download("^INDIAVIX", period="35d", interval="1d", progress=False, timeout=5)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        if len(vix) >= 21:
            vix_close = float(vix["Close"].iloc[-2])
            vix_sma20 = float(vix["Close"].iloc[-21:-1].mean())
    except Exception:
        pass

    return nifty_trend, idx_pd_close, idx_pd_close20, vix_close, vix_sma20

def calculate_target_spread(pd_high, pd_low, pd_close, pd_close20, d_atr, d_ema20, d_ema50, d_ema200, pd_vol, d_vol_sma, open_price, idx_pd_close, idx_pd_close20, vix_close=15.0, vix_sma20=15.0):
    effective_vol = vix_close if (vix_close and vix_close > 0) else 15.0
    vix_ratio = (effective_vol / vix_sma20) if (vix_sma20 and vix_sma20 > 0) else 1.0
    vix_weight = min(max(0.50 * vix_ratio, 0.25), 0.75)
    atr_weight = 1.0 - vix_weight
    expected_move_macro = pd_close * ((effective_vol / 15.9058) / 100.0)

    trend_score = sum([pd_close > d_ema20, d_ema20 > d_ema50, d_ema50 > d_ema200])
    adaptive_trend_macro = min(max(1.0 + ((trend_score - 1.5) * 0.08 * vix_ratio), 0.85), 1.18)

    stk_20d_ret = (pd_close - pd_close20) / pd_close20 if pd_close20 > 0 else 0.0
    idx_20d_ret = (idx_pd_close - idx_pd_close20) / idx_pd_close20 if (idx_pd_close20 and idx_pd_close20 > 0) else 0.0
    rs_20d_diff = stk_20d_ret - idx_20d_ret

    rvol = (pd_vol / d_vol_sma) if (d_vol_sma and d_vol_sma > 0) else 1.0
    rvol_macro = min(max(1.0 + ((rvol - 1.0) * 0.10), 0.90), 1.15)

    pivot = (pd_high + pd_low + pd_close) / 3.0
    cpr_bc = (pd_high + pd_low) / 2.0
    cpr_tc = (pivot - cpr_bc) + pivot
    cpr_w = abs(cpr_tc - cpr_bc)
    cpr_ratio = (cpr_w / d_atr) if d_atr > 0 else 0.25
    cpr_macro = min(max(1.0 + ((0.25 - cpr_ratio) * 0.50), 0.80), 1.20)

    range_macro = ((expected_move_macro * vix_weight) + (d_atr * atr_weight)) * adaptive_trend_macro * cpr_macro * rvol_macro
    rs_shift_macro = range_macro * min(max(rs_20d_diff, -0.12), 0.12)

    r2_val = pd_close + range_macro + (rs_shift_macro if rs_shift_macro > 0 else 0.0)
    s2_val = pd_close - range_macro + (rs_shift_macro if rs_shift_macro < 0 else 0.0)
    return (r2_val - s2_val) / 2.0

# ==================== 5-MIN SMC & BOX ENGINE ====================
def compute_smc_matrix_5m(df_5m):
    if len(df_5m) < 35:
        return "NEUTRAL", "NONE", False, False

    df = df_5m.copy()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['EMA13_Up'] = df['EMA13'] > df['EMA13'].shift(1)
    df['EMA13_Dn'] = df['EMA13'] < df['EMA13'].shift(1)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['ATR14'] = calculate_atr(df, 14)
    df['Vol_SMA14'] = df['Volume'].rolling(14).mean()
    df['Vol_Breakout'] = (df['Volume'] > (df['Vol_SMA14'] * 1.3)) & (df['Volume'] > df['Volume'].shift(1).rolling(3).max())

    pivotLen_smc, fvgMinAtr, stateLife = 5, 0.1, 30
    n = len(df)
    highs, lows, closes, opens = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    atr, rsi, ema13 = df['ATR14'].values, df['RSI14'].values, df['EMA13'].values
    ema_up, ema_dn, vol_bo = df['EMA13_Up'].values, df['EMA13_Dn'].values, df['Vol_Breakout'].values

    last_high, last_low, prev_high, prev_low = np.nan, np.nan, np.nan, np.nan
    b_state, b_bars, s_state, s_bars, current_trend = 0, 0, 0, 0, 0

    for i in range(n):
        b_bars += 1
        s_bars += 1

        if i >= 2 * pivotLen_smc:
            p_idx = i - pivotLen_smc
            is_ph = all(highs[p_idx] > highs[p_idx - k] for k in range(1, pivotLen_smc + 1)) and \
                    all(highs[p_idx] > highs[p_idx + k] for k in range(1, pivotLen_smc + 1))
            is_pl = all(lows[p_idx] < lows[p_idx - k] for k in range(1, pivotLen_smc + 1)) and \
                    all(lows[p_idx] < lows[p_idx + k] for k in range(1, pivotLen_smc + 1))
            if is_ph: prev_high, last_high = last_high, highs[p_idx]
            if is_pl: prev_low, last_low = last_low, lows[p_idx]

        close_prev = closes[i-1] if i > 0 else closes[i]

        # Bull State Path
        bull_sweep = not np.isnan(last_low) and lows[i] < last_low and closes[i] > last_low
        bull_mss = not np.isnan(last_high) and closes[i] > last_high and close_prev <= last_high
        bull_bos = not np.isnan(prev_high) and not np.isnan(last_high) and last_high > prev_high and closes[i] > last_high and close_prev <= last_high
        bull_fvg = i >= 2 and lows[i] > highs[i-2] and (lows[i] - highs[i-2]) > (atr[i] * fvgMinAtr)

        if bull_sweep: b_state, b_bars = 1, 0
        if b_state >= 1 and bull_mss: b_state, b_bars = 2, 0
        if b_state >= 2 and bull_bos: b_state, b_bars = 3, 0
        if b_state >= 2 and bull_fvg: b_state, b_bars = 4, 0
        if b_bars > stateLife: b_state = 0

        # Bear State Path
        bear_sweep = not np.isnan(last_high) and highs[i] > last_high and closes[i] < last_high
        bear_mss = not np.isnan(last_low) and closes[i] < last_low and close_prev >= last_low
        bear_bos = not np.isnan(prev_low) and not np.isnan(last_low) and last_low < prev_low and closes[i] < last_low and close_prev >= last_low
        bear_fvg = i >= 2 and highs[i] < lows[i-2] and (lows[i-2] - highs[i]) > (atr[i] * fvgMinAtr)

        if bear_sweep: s_state, s_bars = 1, 0
        if s_state >= 1 and bear_mss: s_state, s_bars = 2, 0
        if s_state >= 2 and bear_bos: s_state, s_bars = 3, 0
        if s_state >= 2 and bear_fvg: s_state, s_bars = 4, 0
        if s_bars > stateLife: s_state = 0

        bull_confirm = (b_state == 4) and (closes[i] > ema13[i]) and ema_up[i] and vol_bo[i] and (rsi[i] > 50)
        bear_confirm = (s_state == 4) and (closes[i] < ema13[i]) and ema_dn[i] and vol_bo[i] and (rsi[i] < 50)

        if bull_confirm: current_trend = 1
        elif bear_confirm: current_trend = -1

        if current_trend == 1 and (closes[i] < ema13[i] and rsi[i] < 45): current_trend = 0
        if current_trend == -1 and (closes[i] > ema13[i] and rsi[i] > 55): current_trend = 0

    # Demand & Supply Box Logic
    pivotLen_matrix, mergeThresh = 2, 0.3
    demand_boxes, supply_boxes = [], []
    supply_created_recent = False
    demand_created_recent = False

    for i in range(2 * pivotLen_matrix, n):
        p_idx = i - pivotLen_matrix
        is_ph = all(highs[p_idx] > highs[p_idx - k] for k in range(1, pivotLen_matrix + 1)) and \
                all(highs[p_idx] > highs[p_idx + k] for k in range(1, pivotLen_matrix + 1))
        if is_ph:
            top_lvl = highs[p_idx]
            bot_lvl = max(opens[p_idx], closes[p_idx])
            is_dup = any(s['active'] and abs(s['top'] - top_lvl) < (atr[i] * mergeThresh) for s in supply_boxes)
            if not is_dup:
                supply_boxes.append({'top': top_lvl, 'bot': bot_lvl, 'active': True, 'tests': 0})
                if i >= n - 2:
                    supply_created_recent = True

        is_pl = all(lows[p_idx] < lows[p_idx - k] for k in range(1, pivotLen_matrix + 1)) and \
                all(lows[p_idx] < lows[p_idx + k] for k in range(1, pivotLen_matrix + 1))
        if is_pl:
            top_lvl = min(opens[p_idx], closes[p_idx])
            bot_lvl = lows[p_idx]
            is_dup = any(d['active'] and abs(d['bot'] - bot_lvl) < (atr[i] * mergeThresh) for d in demand_boxes)
            if not is_dup:
                demand_boxes.append({'top': top_lvl, 'bot': bot_lvl, 'active': True, 'tests': 0})
                if i >= n - 2:
                    demand_created_recent = True

    ema_color_str = "🟢 GREEN" if current_trend == 1 else ("🔴 RED" if current_trend == -1 else "🟡 YELLOW")
    box_str = "DEMAND" if demand_created_recent else ("SUPPLY" if supply_created_recent else "NONE")

    is_bull_signal = (current_trend == 1) and demand_created_recent
    is_bear_signal = (current_trend == -1) and supply_created_recent

    return ema_color_str, box_str, is_bull_signal, is_bear_signal

# ==================== MAIN DATA FETCHING ====================
def run_scan():
    session = create_robust_nse_session()
    nifty_trend, idx_pd_close, idx_pd_close20, vix_close, vix_sma20 = fetch_benchmark_data()
    yf_symbols = [f"{s}.NS" for s in WATCHLIST]

    try:
        data_daily = yf.download(yf_symbols, period="250d", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True, timeout=10)
        data_5m = yf.download(yf_symbols, period="5d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True, timeout=10)
    except Exception:
        return [], [], nifty_trend

    full_out = []
    filtered_signals = []

    HIGH_PROBABILITY_ACTIONS = [
        "🟢 SMC PRO BUY (5m)",
        "🔴 SMC PRO SELL (5m)",
        "🟢 STRONG BUY",
        "🔴 STRONG SELL"
    ]

    for sym in WATCHLIST:
        t_str = f"{sym}.NS"
        try:
            if isinstance(data_daily.columns, pd.MultiIndex):
                if t_str not in data_daily.columns.levels[0]: continue
                df_d = data_daily[t_str].dropna()
            else:
                if t_str not in data_daily: continue
                df_d = data_daily.dropna()

            if len(df_d) < 205: continue

            if isinstance(data_5m.columns, pd.MultiIndex):
                if t_str not in data_5m.columns.levels[0]: continue
                df_5 = data_5m[t_str].dropna()
            else:
                if t_str not in data_5m: continue
                df_5 = data_5m.dropna()

            if len(df_5) < 35: continue

            ltp = float(df_5.iloc[-1]["Close"])
            open_p = float(df_d.iloc[-1]["Open"])
            if PRICE_FILTER_ACTIVE and not (PRICE_MIN <= ltp <= PRICE_MAX):
                continue

            prev_close = float(df_d.iloc[-2]["Close"])
            p_change = ((ltp - prev_close) / prev_close) * 100

            today_vol = float(df_d.iloc[-1]["Volume"])
            hist_vols = df_d["Volume"].iloc[-6:-1]
            avg_vol_1w = hist_vols.mean() if len(hist_vols) > 0 else today_vol
            vol_chg_pct = ((today_vol - avg_vol_1w) / avg_vol_1w * 100) if avg_vol_1w > 0 else 0

            high, low, close = float(df_d.iloc[-1]["High"]), float(df_d.iloc[-1]["Low"]), float(df_d.iloc[-1]["Close"])
            approx_vwap = (high + low + close) / 3
            vwap_dist_pct = ((ltp - approx_vwap) / approx_vwap) * 100
            vwap_status = "ABOVE (+)" if vwap_dist_pct > VWAP_BUFFER_PCT else ("BELOW (-)" if vwap_dist_pct < -VWAP_BUFFER_PCT else "AT VWAP")

            bs_ratio, imbalance, has_depth = fetch_live_orderbook(session, sym)
            if not has_depth:
                bs_ratio = round(max(0.2, min(5.0, 1.0 + (p_change * 0.35))), 2)
                imbalance = int(today_vol * (p_change / 100))

            imb_vs_avg_vol_pct = ((imbalance / avg_vol_1w) * 100) if avg_vol_1w > 0 else 0.0

            ema_color, box_type, is_bull_smc, is_bear_smc = compute_smc_matrix_5m(df_5)

            if is_bull_smc:
                action = "🟢 SMC PRO BUY (5m)"
            elif is_bear_smc:
                action = "🔴 SMC PRO SELL (5m)"
            elif p_change > 0.5 and vol_chg_pct > 10 and bs_ratio >= 1.3 and imb_vs_avg_vol_pct > 1.0 and vwap_status == "ABOVE (+)":
                action = "🟢 STRONG BUY"
            elif p_change < -0.5 and vol_chg_pct > 10 and bs_ratio <= 0.7 and imb_vs_avg_vol_pct < -1.0 and vwap_status == "BELOW (-)":
                action = "🔴 STRONG SELL"
            elif bs_ratio > 2.0 and vwap_status == "ABOVE (+)":
                action = "🟢 ACCUMULATION"
            elif bs_ratio < 0.5 and vwap_status == "BELOW (-)":
                action = "🔴 DISTRIBUTION"
            else:
                action = "🟡 SIDEWAYS"

            pd_high = float(df_d.iloc[-2]["High"])
            pd_low = float(df_d.iloc[-2]["Low"])
            pd_close_val = float(df_d.iloc[-2]["Close"])
            pd_close20_val = float(df_d.iloc[-21]["Close"])
            pd_vol_val = float(df_d.iloc[-2]["Volume"])

            tr = pd.concat([
                df_d["High"] - df_d["Low"],
                (df_d["High"] - df_d["Close"].shift(1)).abs(),
                (df_d["Low"] - df_d["Close"].shift(1)).abs()
            ], axis=1).max(axis=1)

            d_atr = float(tr.iloc[-15:-1].mean())
            d_ema20 = float(df_d["Close"].ewm(span=20, adjust=False).mean().iloc[-2])
            d_ema50 = float(df_d["Close"].ewm(span=50, adjust=False).mean().iloc[-2])
            d_ema200 = float(df_d["Close"].ewm(span=200, adjust=False).mean().iloc[-2])
            d_vol_sma = float(df_d["Volume"].rolling(20).mean().iloc[-2])

            target_spread = calculate_target_spread(
                pd_high, pd_low, pd_close_val, pd_close20_val,
                d_atr, d_ema20, d_ema50, d_ema200,
                pd_vol_val, d_vol_sma, open_p,
                idx_pd_close, idx_pd_close20,
                vix_close, vix_sma20
            )

            record = {
                "Symbol": sym, 
                "Price": round(ltp, 2), 
                "Chg%": f"{p_change:+.2f}%", 
                "VolChg%": f"{vol_chg_pct:+.2f}%",
                "EMA13(5m)": ema_color, 
                "Box(5m)": box_type,
                "B/S": bs_ratio, 
                "Imbalance": imbalance, 
                "ImbChg%": f"{imb_vs_avg_vol_pct:+.2f}%",
                "VWAP": vwap_status, 
                "Spread": round(target_spread, 2), 
                "Action": action
            }

            full_out.append(record)

            if action in HIGH_PROBABILITY_ACTIONS:
                filtered_signals.append(record)

        except Exception:
            continue

    return full_out, filtered_signals, nifty_trend

# ==================== STREAMLIT UI ====================
st.title("⚡ 5M Quant Live Scanner")

# Header status
header_col1, header_col2, header_col3 = st.columns([1, 1, 1])

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True

with header_col1:
    if st.button("🔄 Refresh Data"):
        st.rerun()

data_all, data_conviction, nifty_status = run_scan()

with header_col2:
    st.metric("NIFTY 50", nifty_status)
with header_col3:
    st.metric("Updated", datetime.now().strftime("%H:%M:%S"))

# High Conviction Section
st.subheader("🎯 High Conviction Setups")
if data_conviction:
    df_conv = pd.DataFrame(data_conviction)
    st.dataframe(df_conv, use_container_width=True, hide_index=True)
else:
    st.info("⚡ No High-Conviction setups found right now (Waiting for SMC PRO or STRONG triggers).")

# Full Scanner Watchlist
st.subheader("📊 Full Market Scanner")
if data_all:
    df_all = pd.DataFrame(data_all)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
else:
    st.warning("Fetching market data... please wait or click refresh.")

# Auto-refresh loop for live mode (12 seconds)
time.sleep(12)
st.rerun()
