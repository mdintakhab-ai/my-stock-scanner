import time
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Suppress background logs and warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Streamlit Mobile Configuration
st.set_page_config(
    page_title="Falcon Quant SMC Mobile",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Mobile-Optimized Dark CSS
st.markdown("""
<style>
    .stApp { background-color: #090c10; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; }
    .mobile-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .metric-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; margin-bottom: 4px; }
    h1, h2, h3 { color: #00e676 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HARD CONFIGURATION & UNIVERSE DUAL-RANGE
# -----------------------------------------------------------------------------
MIN_PRICE = 100.0
MAX_PRICE = 1500.0
DASHBOARD_MIN_PRICE = 300.0
DASHBOARD_MAX_PRICE = 600.0
HIGH_CONVICTION_TCS_THRESHOLD = 30
MIN_RUNWAY_PERCENT = 0.8
MAX_LOCKED_STOCKS = 7

STREAM_TICKERS = [
    "FEDERALBNK.NS", "CONCOR.NS", "JINDALSAW.NS", "SYNGENE.NS", "RELAXO.NS",
    "JSWENERGY.NS", "NUVOCO.NS", "BPCL.NS", "LICHSGFIN.NS", "HINDPETRO.NS",
    "TATAPOWER.NS", "NATIONALUM.NS", "COALINDIA.NS", "ASHOKLEY.NS", "CANBK.NS",
    "IDFCFIRSTB.NS", "CROMPTON.NS", "RBLBANK.NS", "POONAWALLA.NS", "GMDCLTD.NS",
    "HINDCOPPER.NS", "REDINGTON.NS", "PRECWIRE.NS", "BHEL.NS", "ELECON.NS", 
    "USHAMART.NS", "BEL.NS", "IOC.NS", "SAIL.NS", "NMDC.NS"
]

# -----------------------------------------------------------------------------
# NUMPY QUANT MATH & TRUE SMC ENGINE
# -----------------------------------------------------------------------------
def fast_atr(high, low, close, period=14):
    if len(close) < 2: return 5.0
    tr0 = high[1:] - low[1:]
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    if len(tr) < period: return float(np.mean(tr)) if len(tr) > 0 else 5.0
    return float(np.mean(tr[-period:]))

def fast_ema(arr, span):
    if len(arr) < span: return arr[-1] if len(arr) > 0 else 0.0
    alpha = 2.0 / (span + 1.0)
    ema = arr[0]
    for x in arr[1:]:
        ema = alpha * x + (1.0 - alpha) * ema
    return ema

def fast_rsi(close, period=14):
    if len(close) < period + 2: return 50.0
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))

def compute_bidirectional_cri(df_1m, ltp, target_zone_price, atr_14, current_direction="SUPPLY", span_bars=5):
    if len(df_1m) < span_bars:
        return 0.0, "TREND_STABLE", "HOLD_ORIGINAL"
    
    sub = df_1m.iloc[-span_bars:]
    c = sub['Close'].to_numpy(dtype=float)
    h = sub['High'].to_numpy(dtype=float)
    l = sub['Low'].to_numpy(dtype=float)
    o = sub['Open'].to_numpy(dtype=float)
    v = sub['Volume'].to_numpy(dtype=float)
    bar_range = (h - l) + 1e-6
    
    dist = max(0.0, ltp - target_zone_price) if current_direction == "SUPPLY" else max(0.0, target_zone_price - ltp)
    s_p = float(np.exp(-2.5 * (dist / (atr_14 + 1e-6))))
    
    weights = np.arange(1, span_bars + 1)
    power = ((c - l) / bar_range) if current_direction == "SUPPLY" else ((h - c) / bar_range)
    weighted_vol = v * weights
    a_v = float(np.sum(power * weighted_vol) / (np.sum(weighted_vol) + 1e-6))
    
    v_mean = np.mean(v) if len(v) > 0 else 1.0
    vol_mult = float(np.sqrt(min(2.0, max(0.5, v[-1] / (v_mean + 1e-6)))))
    raw_wick = (max(0.0, min(o[-1], c[-1]) - l[-1]) / bar_range[-1]) if current_direction == "SUPPLY" else (max(0.0, h[-1] - max(o[-1], c[-1])) / bar_range[-1])
    w_r = min(1.0, float(raw_wick * vol_mult))
    
    tot_v = np.sum(v) + 1e-6
    micro_vwap = np.sum(((h + l + c) / 3.0) * v) / tot_v
    alpha = 2.0 / (13.0 + 1.0)
    ema1 = c[0]
    for x in c[1:]: ema1 = alpha * x + (1.0 - alpha) * ema1
    
    norm_factor = 0.1 * atr_14 + 1e-6
    diff_ema = ((ltp - ema1) / norm_factor) if current_direction == "SUPPLY" else ((ema1 - ltp) / norm_factor)
    diff_vwap = ((ltp - micro_vwap) / norm_factor) if current_direction == "SUPPLY" else ((micro_vwap - ltp) / norm_factor)
    
    sig_ema = 1.0 / (1.0 + np.exp(-np.clip(diff_ema, -10, 10)))
    sig_vwap = 1.0 / (1.0 + np.exp(-np.clip(diff_vwap, -10, 10)))
    m_s = float(0.5 * sig_ema + 0.5 * sig_vwap)
    
    cri = float((0.35 * s_p + 0.25 * a_v + 0.25 * w_r + 0.15 * m_s) * 100.0)
    cri = min(100.0, max(0.0, cri))
    
    new_side = "BUY_CALL" if current_direction == "SUPPLY" else "SELL_PUT"
    if cri >= 80.0:
        status, action = "CRITICAL_REVERSAL", f"ENTER_{new_side}_NOW"
    elif cri >= 65.0:
        status, action = "EARLY_TRIGGER", f"READY_{new_side}"
    elif cri >= 40.0:
        status, action = "PULLBACK_TRAIL", "TIGHTEN_SL"
    else:
        status, action = "TREND_STABLE", f"HOLD_{'SHORT' if current_direction == 'SUPPLY' else 'LONG'}"
        
    return round(cri, 2), status, action

# -----------------------------------------------------------------------------
# DIRECT DATA PARSER
# -----------------------------------------------------------------------------
def fetch_ticker_data_instant(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="1d", interval="1m", auto_adjust=True)
        if df is not None and not df.empty and len(df) >= 3:
            return df
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# MAIN APP EXECUTION LOOP
# -----------------------------------------------------------------------------
st.markdown("<div style='color:#00e676; font-size:14px; font-weight:900;'>⚡ FALCON QUANT ENGINE | MOBILE ALPHA</div>", unsafe_allow_html=True)

if 'locked_symbols' not in st.session_state:
    st.session_state.locked_symbols = None
if 'is_universe_locked' not in st.session_state:
    st.session_state.is_universe_locked = False

status_placeholder = st.empty()
container_placeholder = st.container()

while True:
    t0 = time.time()
    current_rows = []
    
    for ticker in STREAM_TICKERS:
        df = fetch_ticker_data_instant(ticker)
        if df is None or df.empty or len(df) < 3:
            continue
        
        try:
            close = df['Close'].to_numpy(dtype=float)
            high = df['High'].to_numpy(dtype=float)
            low = df['Low'].to_numpy(dtype=float)
            open_arr = df['Open'].to_numpy(dtype=float)
            vol = df['Volume'].to_numpy(dtype=float)
            
            open_p = float(open_arr[0])
            ltp = float(close[-1])
            day_high = float(np.max(high))
            day_low = float(np.min(low))
            
            if not (MIN_PRICE <= open_p <= MAX_PRICE): 
                continue
            
            atr_val = fast_atr(high, low, close, 14)
            pnl_pct = ((ltp - open_p) / open_p) * 100.0
            direction = "DEMAND" if ltp >= open_p else "SUPPLY"
            
            if direction == "SUPPLY":
                target_price = ltp - max(atr_val * 1.2, ltp * 0.012)
                sl_best_entry = max(day_high, open_p + max(0.35 * atr_val, ltp * 0.004))
                zone_text = "SUPPLY (15m / 1D)"
            else:
                target_price = ltp + max(atr_val * 1.2, ltp * 0.012)
                sl_best_entry = min(day_low, open_p - max(0.35 * atr_val, ltp * 0.004))
                zone_text = "DEMAND (15m / 1D)"
            
            runway_gap = abs(target_price - ltp)
            runway_pct = (runway_gap / ltp) * 100.0
            if runway_pct < MIN_RUNWAY_PERCENT or runway_gap < 1.0:
                continue
            
            tp = (high + low + close) / 3.0
            cum_vol = np.sum(vol) + 1e-6
            vwap = float(np.sum(tp * vol) / cum_vol)
            rsi = fast_rsi(close, 14)
            
            ema1 = fast_ema(close[-15:], 13)
            ema3 = fast_ema(close[-45::3], 13) if len(close) >= 40 else ema1
            ema5 = fast_ema(close[-75::5], 13) if len(close) >= 65 else ema1
            ema15 = fast_ema(close[-225::15], 13) if len(close) >= 150 else ema1
            
            bull_cnt = sum([ltp > ema1, ltp > ema3, ltp > ema5, ltp > ema15])
            bear_cnt = 4 - bull_cnt
            
            bar_range = (high - low) + 1e-6
            buy_power = np.sum(vol * ((close - low) / bar_range))
            sell_power = np.sum(vol * ((high - close) / bar_range))
            tot_power = buy_power + sell_power + 1e-6
            
            directional_move = (ltp - open_p) if direction == 'DEMAND' else (open_p - ltp)
            pressure_pct = min(100.0, max(0.0, (directional_move / atr_val) * 100.0))
            
            cri_val, cri_status, cri_action = compute_bidirectional_cri(
                df, ltp, target_price, atr_val, current_direction=direction, span_bars=5
            )
            
            mtf_score = (max(bull_cnt, bear_cnt) / 4.0) * 25.0
            vwap_score = 20.0 if ((direction == 'DEMAND' and ltp > vwap) or (direction == 'SUPPLY' and ltp < vwap)) else 0.0
            rsi_score = 15.0 if ((direction == 'DEMAND' and 50 <= rsi <= 70) or (direction == 'SUPPLY' and 30 <= rsi <= 50)) else 5.0
            zone_score = 20.0
            pressure_component = min(20.0, pressure_pct * 0.2)
            tcs = int(min(100.0, max(0.0, mtf_score + vwap_score + rsi_score + zone_score + pressure_component)))
            
            if tcs >= HIGH_CONVICTION_TCS_THRESHOLD:
                current_rows.append({
                    "ticker": ticker,
                    "symbol": ticker.replace(".NS", ""),
                    "zone_text": zone_text,
                    "ltp": ltp,
                    "pnl": pnl_pct,
                    "pressure_pct": pressure_pct,
                    "cri_val": cri_val,
                    "cri_action": cri_action,
                    "sl_best_entry": sl_best_entry,
                    "target": target_price,
                    "tcs": tcs
                })
        except Exception:
            continue

    dashboard_eligible_rows = [r for r in current_rows if DASHBOARD_MIN_PRICE <= r['ltp'] <= DASHBOARD_MAX_PRICE]
    dashboard_eligible_rows.sort(key=lambda x: x['tcs'], reverse=True)

    if not st.session_state.is_universe_locked and len(dashboard_eligible_rows) >= 4:
        st.session_state.locked_symbols = [r['ticker'] for r in dashboard_eligible_rows[:MAX_LOCKED_STOCKS]]
        st.session_state.is_universe_locked = True
        
    if st.session_state.is_universe_locked and st.session_state.locked_symbols:
        display_rows = [r for r in dashboard_eligible_rows if r['ticker'] in st.session_state.locked_symbols]
        display_rows.sort(key=lambda x: st.session_state.locked_symbols.index(x['ticker']) if x['ticker'] in st.session_state.locked_symbols else 99)
    else:
        display_rows = dashboard_eligible_rows[:MAX_LOCKED_STOCKS]

    elapsed_ms = int((time.time() - t0) * 1000)
    now_time = datetime.datetime.now().strftime('%H:%M:%S')

    with status_placeholder.container():
        lock_label = "🔒 LOCKED" if st.session_state.is_universe_locked else "⚡ STREAMING"
        st.caption(f"{lock_label} | {now_time} IST | Latency: {elapsed_ms}ms")

    with container_placeholder.container():
        if not display_rows:
            st.info(f"Scanning range Rs {DASHBOARD_MIN_PRICE} - Rs {DASHBOARD_MAX_PRICE}...")
        
        for r in display_rows:
            pnl_color = "#00e676" if r['pnl'] >= 0 else "#ff5252"
            st.markdown(f"""
            <div class="mobile-card">
                <div style="display:flex; justify-content:space-between; font-weight:900; font-size:13px; border-bottom:1px dashed #30363d; padding-bottom:4px; margin-bottom:6px;">
                    <span>{r['symbol']} <span style="font-size:9px; color:#8b949e;">({r['zone_text']})</span></span>
                    <span style="color:{pnl_color};">Rs {r['ltp']:.2f} ({r['pnl']:+.2f}%)</span>
                </div>
                <div class="metric-row">
                    <span><b>SL / Entry:</b> Rs {r['sl_best_entry']:.2f}</span>
                    <span><b>Target:</b> <span style="color:#00e676;">Rs {r['target']:.2f}</span></span>
                </div>
                <div class="metric-row">
                    <span><b>Pressure:</b> {r['pressure_pct']:.1f}% | <b>CRI:</b> {r['cri_val']:.1f}%</span>
                    <span><b>TCS:</b> <span style="color:#00e676; font-weight:bold;">{r['tcs']}/100</span></span>
                </div>
                <div style="font-size:10px; color:#ffaa00; margin-top:2px;">⚡ {r['cri_action']}</div>
            </div>
            """, unsafe_allow_html=True)

    time.sleep(2)
