import time
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

# Suppress background logs and warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Streamlit Mobile & Desktop Configuration
st.set_page_config(
    page_title="Falcon Quant SMC Mobile Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 1. HARD CONFIGURATION & UNIVERSE DUAL-RANGE
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
# 2. NUMPY QUANT MATH & TRUE SMC ENGINE
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
# 3. DIRECT DATA PARSER
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
# 4. DASHBOARD RENDERER & BALANCED RESPONSIVE UI
# -----------------------------------------------------------------------------
def build_html_view(rows, timestamp, latency_ms, total_scanned, is_locked=False):
    rows_str = ""
    lock_badge = "<span style='color:#00e676; font-weight:bold;'>🔒 UNIVERSE LOCKED (ALPHA FREEZE)</span>" if is_locked else "<span style='color:#ffaa00; font-weight:bold;'>⚡ STREAMING QUANT DESK...</span>"
    
    if not rows:
        rows_str = """
        <tr>
            <td colspan="12" style="padding: 24px; color: #8b949e; text-align: center; font-style: italic;">
                Scanning background universe (Rs 100 - Rs 1500)... Filtered for Rs 300 - Rs 600 Desk.
            </td>
        </tr>
        """
    else:
        for r in rows:
            def dot(b): return "<span style='color:#00e676;'>🟢</span>" if b else "<span style='color:#ff5252;'>🔴</span>"
            emas_html = f"<span style='white-space:nowrap;'>{dot(r['e1'])} {dot(r['e3'])} {dot(r['e5'])} {dot(r['e15'])}</span>"
            
            rows_str += f"""
            <tr style='border-bottom: 1px dashed #30363d;'>
                <td style='font-weight: 900; text-align: left; color: #ffffff; padding: 10px 8px; border-right: 1px dashed #21262d; white-space: nowrap;'>{r['symbol']}</td>
                <td style='text-align: left; font-size: 9.5px; line-height: 1.35; border-right: 1px dashed #21262d; padding: 6px 8px; white-space: nowrap;'>{r['zone_html']}</td>
                <td style='border-right: 1px dashed #21262d; white-space: nowrap;'>Rs {r['open']:.2f}</td>
                <td style='font-weight: 700; border-right: 1px dashed #21262d; white-space: nowrap;'>Rs {r['ltp']:.2f}</td>
                <td style='color: {'#00e676' if r['pnl'] >= 0 else '#ff5252'}; font-weight: 800; border-right: 1px dashed #21262d; white-space: nowrap;'>{r['pnl']:+.2f}%</td>
                <td style='border-right: 1px dashed #21262d;'>{emas_html}</td>
                <td style='padding: 5px 6px; border-right: 1px dashed #21262d;'>{r['pressure_box']}</td>
                <td style='padding: 5px 6px; border-right: 1px dashed #21262d;'>{r['cri_box']}</td>
                <td style='padding: 5px 6px; border-right: 1px dashed #21262d;'>{r['sl_box']}</td>
                <td style='color: #00e676; font-weight: 800; border-right: 1px dashed #21262d; white-space: nowrap;'>Rs {r['target']:.2f}</td>
                <td style='border-right: 1px dashed #21262d; white-space: nowrap;'><span style='color: #00e676; font-weight: 900; font-size: 12px;'>{r['tcs']}/100</span></td>
                <td style='color: {'#00e676' if r['imbalance'] >= 0 else '#ff5252'}; font-weight: 700; white-space: nowrap; padding: 6px 10px;'>{r['cobi_html']}</td>
            </tr>
            """
        
    return f"""
    <div style="background-color: #090c10; border: 1.5px solid #30363d; border-radius: 8px; padding: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; color: #c9d1d9; overflow-x: auto;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed #30363d; padding-bottom: 8px; margin-bottom: 10px;">
            <div style="color: #00e676; font-size: 13px; font-weight: 900; letter-spacing: 0.5px;">⚡ FALCON QUANT ENGINE | SMC MOBILE ALPHA TERMINAL</div>
            <div style="color: #8b949e; font-size: 11px; background: #161b22; padding: 4px 10px; border-radius: 4px; border: 1px dashed #30363d;">
                {lock_badge} | LIVE: {timestamp} IST | Active: {len(rows)}/{total_scanned} | Latency: {latency_ms}ms
            </div>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 11px; text-align: center; border: 1px dashed #30363d;">
            <thead>
                <tr style="background: #161b22; color: #8b949e; text-transform: uppercase; font-size: 9.5px; border-bottom: 1px dashed #30363d;">
                    <th style="text-align: left; padding: 8px 6px; border-right: 1px dashed #30363d;">Symbol</th>
                    <th style="text-align: left; padding: 8px 6px; border-right: 1px dashed #30363d;">Zone Alignments</th>
                    <th style="border-right: 1px dashed #30363d;">Open</th>
                    <th style="border-right: 1px dashed #30363d;">LTP</th>
                    <th style="border-right: 1px dashed #30363d;">Change</th>
                    <th style="border-right: 1px dashed #30363d;">EMAS (1m|3m|5m|15m)</th>
                    <th style="border-right: 1px dashed #30363d; min-width: 140px;">Supply/Demand Delta Box</th>
                    <th style="border-right: 1px dashed #30363d; min-width: 140px;">Reversal Engine (CRI)</th>
                    <th style="border-right: 1px dashed #30363d; min-width: 115px;">SL / Best Entry</th>
                    <th style="border-right: 1px dashed #30363d;">Target</th>
                    <th style="border-right: 1px dashed #30363d;">TCS Score</th>
                    <th style="padding: 8px 10px; min-width: 125px;">Buyer/Seller (COBI)</th>
                </tr>
            </thead>
            <tbody>
                {rows_str}
            </tbody>
        </table>
    </div>
    """

# -----------------------------------------------------------------------------
# 5. EXECUTION ENGINE (STREAMLIT MOBILE ADAPTED)
# -----------------------------------------------------------------------------
if 'locked_symbols' not in st.session_state:
    st.session_state.locked_symbols = None
if 'is_universe_locked' not in st.session_state:
    st.session_state.is_universe_locked = False

terminal_placeholder = st.empty()

while True:
    try:
        t0 = time.time()
        tickers_to_query = STREAM_TICKERS
        current_rows = []
        
        for ticker in tickers_to_query:
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
                
                # Background Universe Filter (Rs 100 to Rs 1500)
                if not (MIN_PRICE <= open_p <= MAX_PRICE): 
                    continue
                
                atr_val = fast_atr(high, low, close, 14)
                pnl_pct = ((ltp - open_p) / open_p) * 100.0
                
                direction = "DEMAND" if ltp >= open_p else "SUPPLY"
                
                # 1. Target & Corrected Mathematical SL / Best Entry Engine
                if direction == "SUPPLY":
                    target_price = ltp - max(atr_val * 1.2, ltp * 0.012)
                    
                    # Corrected Supply SL: Must be ABOVE LTP/Open (e.g. Day High or Open + ATR buffer)
                    sl_best_entry = max(day_high, open_p + max(0.35 * atr_val, ltp * 0.004))
                    
                    sl_border_color = "#ff7675"
                    sl_bg_color = "rgba(255, 118, 117, 0.12)"
                    sl_title = "SELL ENTRY / SL"
                    zone_html = "<span style='color:#ff5252; font-weight:700;'>SUPPLY (15m)</span><br><span style='color:#ff7675; font-size:8.5px;'>SUPPLY (1D)</span>"
                else:
                    target_price = ltp + max(atr_val * 1.2, ltp * 0.012)
                    
                    # Corrected Demand SL: Must be BELOW LTP/Open (e.g. Day Low or Open - ATR buffer)
                    sl_best_entry = min(day_low, open_p - max(0.35 * atr_val, ltp * 0.004))
                    
                    sl_border_color = "#55efc4"
                    sl_bg_color = "rgba(85, 239, 196, 0.12)"
                    sl_title = "BUY ENTRY / SL"
                    zone_html = "<span style='color:#00e676; font-weight:700;'>DEMAND (15m)</span><br><span style='color:#55efc4; font-size:8.5px;'>DEMAND (1D)</span>"
                
                # Hard Chop/Runway Guard
                runway_gap = abs(target_price - ltp)
                runway_pct = (runway_gap / ltp) * 100.0
                if runway_pct < MIN_RUNWAY_PERCENT or runway_gap < 1.0:
                    continue
                
                # SL / Best Entry Box UI
                sl_box_html = f"""
                <div style='border: 1px dashed {sl_border_color}; background-color: {sl_bg_color}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
                    <div style='font-size: 8.5px; font-weight: 800; color: {sl_border_color}; white-space: nowrap;'>{sl_title}</div>
                    <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>Rs {sl_best_entry:.2f}</div>
                </div>
                """
                
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
                
                buy_pct = (buy_power / tot_power) * 100.0
                imbalance_pct = ((buy_power - sell_power) / tot_power) * 100.0
                cobi_html = f"{buy_pct:.0f}% Buy ({imbalance_pct:+.1f}%)"
                
                directional_move = (ltp - open_p) if direction == 'DEMAND' else (open_p - ltp)
                pressure_pct = min(100.0, max(0.0, (directional_move / atr_val) * 100.0))
                
                border_c = "#ff3838" if direction == "SUPPLY" else "#00e676"
                bg_c = "rgba(255, 56, 56, 0.12)" if direction == "SUPPLY" else "rgba(0, 230, 118, 0.12)"
                status_t = "SUPPLY ACCUMULATION" if direction == "SUPPLY" else "DEMAND ABSORPTION"
                retest_html = "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px; white-space:nowrap;'>⚠️ ZONE RE-TEST REJECTION</div>" if pressure_pct > 75.0 else ""
                
                pressure_box_html = f"""
                <div style='border: 1px dashed {border_c}; background-color: {bg_c}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
                    <div style='font-size: 8.5px; font-weight: 800; color: {border_c}; white-space: nowrap;'>{status_t}</div>
                    <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{pressure_pct:.1f}%</div>
                    {retest_html}
                </div>
                """
                
                cri_val, cri_status, cri_action = compute_bidirectional_cri(
                    df, ltp, target_price, atr_val, current_direction=direction, span_bars=5
                )
                
                if cri_val >= 80.0:
                    cri_border = "#00e676" if "BUY" in cri_action else "#ff3838"
                    cri_bg = "rgba(0, 230, 118, 0.15)" if "BUY" in cri_action else "rgba(255, 56, 56, 0.15)"
                    action_color = "#00e676" if "BUY" in cri_action else "#ff5252"
                elif cri_val >= 65.0:
                    cri_border = "#ffaa00"
                    cri_bg = "rgba(255, 170, 0, 0.12)"
                    action_color = "#ffaa00"
                elif cri_val >= 40.0:
                    cri_border = "#ffc107"
                    cri_bg = "rgba(255, 193, 7, 0.08)"
                    action_color = "#ffc107"
                else:
                    cri_border = "#30363d"
                    cri_bg = "#161b22"
                    action_color = "#8b949e"
                    
                cri_box_html = f"""
                <div style='border: 1px dashed {cri_border}; background-color: {cri_bg}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
                    <div style='font-size: 8.5px; font-weight: 800; color: {action_color}; white-space: nowrap;'>{cri_status}</div>
                    <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{cri_val:.1f}%</div>
                    <div style='color: {action_color}; font-size: 8px; font-weight: 900; margin-top: 1px; white-space: nowrap;'>⚡ {cri_action}</div>
                </div>
                """
                
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
                        "zone_html": zone_html,
                        "open": open_p,
                        "ltp": ltp,
                        "pnl": pnl_pct,
                        "pressure_box": pressure_box_html,
                        "cri_box": cri_box_html,
                        "sl_box": sl_box_html,
                        "target": target_price,
                        "tcs": tcs,
                        "cobi_html": cobi_html,
                        "imbalance": imbalance_pct,
                        "e1": ltp > ema1, "e3": ltp > ema3, "e5": ltp > ema5, "e15": ltp > ema15
                    })
            except Exception:
                continue
        
        # Dashboard Price Filter (Rs 300 to Rs 600 only)
        dashboard_eligible_rows = [r for r in current_rows if DASHBOARD_MIN_PRICE <= r['ltp'] <= DASHBOARD_MAX_PRICE]
        dashboard_eligible_rows.sort(key=lambda x: x['tcs'], reverse=True)
        
        # Lock Top 7 after establishing clean active stream
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
        
        html_output = build_html_view(display_rows, now_time, elapsed_ms, len(STREAM_TICKERS), st.session_state.is_universe_locked)
        with terminal_placeholder.container():
            components.html(html_output, height=520, scrolling=True)
            
        time.sleep(2)
        
    except Exception:
        time.sleep(2)
        continue
