# =============================================================================
# ADVANCED SMC + COBI QUANT SCANNER (STREAMLIT CACHED & BULLETPROOF ENGINE)
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

st.set_page_config(
    page_title="SMC + COBI Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.markdown("""
<style>
    .block-container { padding-top: 0.8rem; padding-bottom: 1rem; padding-left: 0.4rem; padding-right: 0.4rem; }
    div[data-testid="stDataFrame"] { width: 100%; font-size: 11px !important; }
    table { font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)

PRICE_MIN = 300.0
PRICE_MAX = 600.0
VWAP_BUFFER_PCT = 0.15

WATCHLIST = [
    "PRECWIRE", "TATAPOWER", "BHEL", "NATIONALUM", "EXIDEIND", "PFC", "RECLTD", "SAIL",
    "NMDC", "GMRP&UI", "HINDCOPPER", "DELHIVERY", "REDINGTON", "HSCL", "PAYTM",
    "CHENNPETRO", "ASTERDM", "IDFCFIRSTB", "FEDERALBNK", "IOC", "ONGC", "GAIL",
    "CANBK", "UNIONBANK", "BANKBARODA", "ASHOKLEY", "BIOCON", "BANDHANBNK",
    "AMBUJACEM", "JUBLFOOD", "MANAPPURAM", "LICHSGFIN", "CUPID", "RELAXO", "ZAGGLE"
]

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
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return rma(tr, length)

def evaluate_pine_indicator(df_tf):
    if len(df_tf) < 15:
        return "🟡 YELLOW", "NONE", False, False

    df = df_tf.copy()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['EMA13_Up'] = df['EMA13'] > df['EMA13'].shift(1)
    df['EMA13_Dn'] = df['EMA13'] < df['EMA13'].shift(1)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['ATR14'] = calculate_atr(df, 14)

    n = len(df)
    highs, lows, closes, opens = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    atr, rsi, ema13 = df['ATR14'].values, df['RSI14'].values, df['EMA13'].values
    ema_up, ema_dn = df['EMA13_Up'].values, df['EMA13_Dn'].values

    p_smc, fvgMinAtr, stateLife = 4, 0.1, 25
    bullState, bullBars = 0, 0
    bearState, bearBars = 0, 0
    trendSMC = 0
    lastHigh, lastLow, prevHigh_smc, prevLow_smc = np.nan, np.nan, np.nan, np.nan

    p_mat = 2
    mergeThresh = 0.3
    maxTests = 4
    s_Boxes, d_Boxes = [], []

    for i in range(n):
        bullBars += 1
        bearBars += 1

        if i >= 2 * p_smc:
            p = i - p_smc
            if all(highs[p] > highs[p-k] for k in range(1, p_smc + 1)) and all(highs[p] > highs[p+k] for k in range(1, p_smc + 1)):
                prevHigh_smc = lastHigh
                lastHigh = highs[p]
            if all(lows[p] < lows[p-k] for k in range(1, p_smc + 1)) and all(lows[p] < lows[p+k] for k in range(1, p_smc + 1)):
                prevLow_smc = lastLow
                lastLow = lows[p]

        cp = closes[i-1] if i > 0 else closes[i]

        bullSweep = not np.isnan(lastLow) and lows[i] < lastLow and closes[i] > lastLow
        bullMSS = not np.isnan(lastHigh) and closes[i] > lastHigh and cp <= lastHigh
        bullBOS = not np.isnan(prevHigh_smc) and not np.isnan(lastHigh) and lastHigh > prevHigh_smc and closes[i] > lastHigh and cp <= lastHigh
        bullFVG = i >= 2 and lows[i] > highs[i-2] and (lows[i] - highs[i-2]) > (atr[i] * fvgMinAtr)

        if bullSweep: bullState, bullBars = 1, 0
        if bullState >= 1 and bullMSS: bullState, bullBars = 2, 0
        if bullState >= 2 and bullBOS: bullState, bullBars = 3, 0
        if bullState >= 2 and bullFVG: bullState, bullBars = 4, 0
        if bullBars > stateLife: bullState = 0

        bearSweep = not np.isnan(lastHigh) and highs[i] > lastHigh and closes[i] < lastHigh
        bearMSS = not np.isnan(lastLow) and closes[i] < lastLow and cp >= lastLow
        bearBOS = not np.isnan(prevLow_smc) and not np.isnan(lastLow) and lastLow < prevLow_smc and closes[i] < lastLow and cp >= lastLow
        bearFVG = i >= 2 and highs[i] < lows[i-2] and (lows[i-2] - highs[i]) > (atr[i] * fvgMinAtr)

        if bearSweep: bearState, bearBars = 1, 0
        if bearState >= 1 and bearMSS: bearState, bearBars = 2, 0
        if bearState >= 2 and bearBOS: bearState, bearBars = 3, 0
        if bearState >= 2 and bearFVG: bearState, bearBars = 4, 0
        if bearBars > stateLife: bearState = 0

        if (bullState == 4 or (closes[i] > ema13[i] and ema_up[i])) and rsi[i] > 50:
            trendSMC = 1
        elif (bearState == 4 or (closes[i] < ema13[i] and ema_dn[i])) and rsi[i] < 50:
            trendSMC = -1

        if trendSMC == 1 and (closes[i] < ema13[i] and rsi[i] < 45): trendSMC = 0
        if trendSMC == -1 and (closes[i] > ema13[i] and rsi[i] > 55): trendSMC = 0

        if i >= 2 * p_mat:
            pm = i - p_mat
            if all(highs[pm] > highs[pm-k] for k in range(1, p_mat + 1)) and all(highs[pm] > highs[pm+k] for k in range(1, p_mat + 1)):
                topLvl, botLvl = highs[pm], max(opens[pm], closes[pm])
                if not any(b['active'] and abs(b['top'] - topLvl) < (atr[i] * mergeThresh) for b in s_Boxes):
                    s_Boxes.append({'top': topLvl, 'bot': botLvl, 'active': True, 'tests': 0, 'created_bar': i})

            if all(lows[pm] < lows[pm-k] for k in range(1, p_mat + 1)) and all(lows[pm] < lows[pm+k] for k in range(1, p_mat + 1)):
                topLvl, botLvl = min(opens[pm], closes[pm]), lows[pm]
                if not any(b['active'] and abs(b['bot'] - botLvl) < (atr[i] * mergeThresh) for b in d_Boxes):
                    d_Boxes.append({'top': topLvl, 'bot': botLvl, 'active': True, 'tests': 0, 'created_bar': i})

        for b in s_Boxes:
            if b['active']:
                if highs[i] > b['bot'] and (i > 0 and highs[i-1] <= b['bot']): b['tests'] += 1
                if highs[i] > b['top'] or b['tests'] >= maxTests: b['active'] = False

        for b in d_Boxes:
            if b['active']:
                if lows[i] < b['top'] and (i > 0 and lows[i-1] >= b['top']): b['tests'] += 1
                if lows[i] < b['bot'] or b['tests'] >= maxTests: b['active'] = False

    recent_supply = any(b['active'] and (n - 1 - b['created_bar'] <= 3) for b in s_Boxes)
    recent_demand = any(b['active'] and (n - 1 - b['created_bar'] <= 3) for b in d_Boxes)

    box_status = "NONE"
    if recent_supply: box_status = "SUPPLY"
    elif recent_demand: box_status = "DEMAND"

    if trendSMC == 1 or (closes[-1] > ema13[-1] and ema_up[-1]):
        ema_color = "🟢 GREEN"
    elif trendSMC == -1 or (closes[-1] < ema13[-1] and ema_dn[-1]):
        ema_color = "🔴 RED"
    else:
        ema_color = "🟡 YELLOW"

    is_bullish_entry = (ema_color == "🟢 GREEN") and (box_status == "DEMAND")
    is_bearish_entry = (ema_color == "🔴 RED") and (box_status == "SUPPLY")

    return ema_color, box_status, is_bullish_entry, is_bearish_entry

@st.cache_data(ttl=15, show_spinner=False)
def load_all_market_data():
    tickers = [f"{s}.NS" for s in WATCHLIST]
    ticker_str = " ".join(tickers)
    try:
        df_daily = yf.download(ticker_str, period="10d", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        df_5m = yf.download(ticker_str, period="5d", interval="5m", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        return df_daily, df_5m
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ==================== STREAMLIT UI ====================
st.title("⚡ SMC + COBI Quant Scanner")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.metric("Range", f"₹{int(PRICE_MIN)} - ₹{int(PRICE_MAX)}")
with c2:
    st.metric("Time", datetime.now().strftime("%H:%M:%S"))
with c3:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

data_daily, data_5m = load_all_market_data()

filtered_rows = []
full_rows = []

if not data_daily.empty and not data_5m.empty:
    for sym in WATCHLIST:
        t_str = f"{sym}.NS"
        try:
            if isinstance(data_daily.columns, pd.MultiIndex):
                if t_str not in data_daily.columns.levels[0]: continue
                df_d = data_daily[t_str].dropna()
            else:
                df_d = data_daily.dropna()

            if isinstance(data_5m.columns, pd.MultiIndex):
                if t_str not in data_5m.columns.levels[0]: continue
                df_5 = data_5m[t_str].dropna()
            else:
                df_5 = data_5m.dropna()

            if len(df_d) < 2 or len(df_5) < 15: continue

            ltp = float(df_5["Close"].iloc[-1])
            if not (PRICE_MIN <= ltp <= PRICE_MAX): continue

            prev_close = float(df_d["Close"].iloc[-2])
            p_change = ((ltp - prev_close) / prev_close) * 100

            today_vol = float(df_d["Volume"].iloc[-1])
            avg_vol_1w = df_d["Volume"].iloc[:-1].mean() if len(df_d) > 1 else today_vol
            vol_chg_pct = ((today_vol - avg_vol_1w) / avg_vol_1w * 100) if avg_vol_1w > 0 else 0

            high, low, close = float(df_d["High"].iloc[-1]), float(df_d["Low"].iloc[-1]), float(df_d["Close"].iloc[-1])
            approx_vwap = (high + low + close) / 3
            vwap_dist_pct = ((ltp - approx_vwap) / approx_vwap) * 100
            vwap_status = "ABOVE (+)" if vwap_dist_pct > VWAP_BUFFER_PCT else ("BELOW (-)" if vwap_dist_pct < -VWAP_BUFFER_PCT else "AT VWAP")

            synth_cobi = max(min(p_change / 3.0, 1.0), -1.0)
            bs_ratio = round(max(0.2, min(5.0, 1.0 + (p_change * 0.35))), 2)
            imbalance = int(today_vol * (p_change / 100))

            df_3 = df_5.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()

            ema3_col, box3_t, is_bull_3m, is_bear_3m = evaluate_pine_indicator(df_3 if len(df_3) >= 15 else df_5)
            ema5_col, box5_t, is_bull_5m, is_bear_5m = evaluate_pine_indicator(df_5)

            tr = pd.concat([df_d["High"] - df_d["Low"], (df_d["High"] - df_d["Close"].shift(1)).abs(), (df_d["Low"] - df_d["Close"].shift(1)).abs()], axis=1).max(axis=1)
            spread = float(tr.iloc[-5:].mean()) * 1.2 if len(tr) >= 5 else 5.0

            if (is_bull_3m or is_bull_5m) and synth_cobi >= 0.35 and vwap_status == "ABOVE (+)":
                action = "🚀 INSTITUTIONAL PRO BUY"
            elif (is_bear_3m or is_bear_5m) and synth_cobi <= -0.35 and vwap_status == "BELOW (-)":
                action = "🔻 INSTITUTIONAL PRO SELL"
            elif is_bull_3m and is_bull_5m:
                action = "🟢 SMC PRO BUY (Multi-TF)"
            elif is_bull_3m or is_bull_5m:
                action = "🟢 SMC PRO BUY"
            elif is_bear_3m and is_bear_5m:
                action = "🔴 SMC PRO SELL (Multi-TF)"
            elif is_bear_3m or is_bear_5m:
                action = "🔴 SMC PRO SELL"
            elif synth_cobi >= 0.40 and p_change > 0.5 and vol_chg_pct > 10:
                action = "🟢 STRONG BUY"
            elif synth_cobi <= -0.40 and p_change < -0.5 and vol_chg_pct > 10:
                action = "🔴 STRONG SELL"
            elif synth_cobi > 0.25 and vwap_status == "ABOVE (+)":
                action = "🟢 ACCUMULATION"
            elif synth_cobi < -0.25 and vwap_status == "BELOW (-)":
                action = "🔴 DISTRIBUTION"
            else:
                action = "🟡 SIDEWAYS"

            row_data = {
                "Symbol": sym,
                "Price": round(ltp, 2),
                "Chg%": f"{p_change:+.2f}%",
                "Vol%": f"{vol_chg_pct:+.2f}%",
                "EMA(3m)": ema3_col,
                "Box(3m)": box3_t,
                "EMA(5m)": ema5_col,
                "Box(5m)": box5_t,
                "B/S": bs_ratio,
                "Imbalance": imbalance,
                "COBI": round(synth_cobi, 2),
                "VWAP": vwap_status,
                "Spread": round(spread, 2),
                "Action": action
            }

            if any(k in action for k in ["INSTITUTIONAL", "SMC PRO", "STRONG"]):
                filtered_rows.append(row_data)
            else:
                full_rows.append(row_data)
        except Exception:
            continue

st.subheader("🎯 High Conviction Trades")
if filtered_rows:
    st.dataframe(pd.DataFrame(filtered_rows), use_container_width=True, hide_index=True)
else:
    st.info("⚡ No High-Conviction setups currently in ₹300-₹600 range.")

st.subheader("📊 Other Watchlist Stocks (No Duplicates)")
if full_rows:
    st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)
else:
    st.warning("No other stocks currently match the filter range.")

# Background Auto-Refresh via JavaScript timer to eliminate page blanking
st.markdown("""
    <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 15000);
    </script>
""", unsafe_allow_html=True)
