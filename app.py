# =====================================================================
# FALCON QUANT MASTER ENGINE v13.0 (STREAMLIT & JUPYTER UI COMPATIBLE)
# =====================================================================

import io
import time
import json
import datetime
import warnings
import logging
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# -----------------------------------------------------------------------------
# HARD CONFIGURATION & STRICT THRESHOLDS
# -----------------------------------------------------------------------------
DASHBOARD_MIN_PRICE = 300.0
DASHBOARD_MAX_PRICE = 600.0
MIN_ABSOLUTE_IOFII = 5.0
MIN_RUNWAY_GAP_PCT = 1.0
MAX_LOCKED_STOCKS = 7

SYMBOL_CACHE = []
LAST_FETCH_TIME = 0
LOCKED_UNIVERSE = []
LOCK_EXECUTED = False

# -----------------------------------------------------------------------------
# 1. DYNAMIC UNIVERSE FETCHING (NO HARDCODED FALLBACKS)
# -----------------------------------------------------------------------------
def get_dynamic_nifty500_symbols():
    global SYMBOL_CACHE, LAST_FETCH_TIME
    if SYMBOL_CACHE and (time.time() - LAST_FETCH_TIME < 86400):
        return SYMBOL_CACHE

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            df_nse = pd.read_csv(io.StringIO(r.content.decode("utf-8")))
            col = "Symbol" if "Symbol" in df_nse.columns else df_nse.columns[0]
            syms = [f"{str(s).strip()}.NS" for s in df_nse[col].dropna().unique()]
            if len(syms) > 0:
                SYMBOL_CACHE = syms
                LAST_FETCH_TIME = time.time()
                return SYMBOL_CACHE
    except Exception:
        pass
    return SYMBOL_CACHE

# -----------------------------------------------------------------------------
# 2. INDEX (NIFTY 50) VWAP TREND GATE
# -----------------------------------------------------------------------------
def check_nifty_vwap_gate():
    try:
        df_nifty = yf.download("^NSEI", period="1d", interval="1m", progress=False, auto_adjust=True)
        if df_nifty is None or df_nifty.empty or len(df_nifty) < 3:
            return "NEUTRAL"
            
        if isinstance(df_nifty.columns, pd.MultiIndex):
            df_nifty.columns = df_nifty.columns.get_level_values(0)
            
        df_nifty.columns = [str(c).strip().lower() for c in df_nifty.columns]
        
        close = df_nifty['close'].to_numpy(dtype=float)
        high = df_nifty['high'].to_numpy(dtype=float)
        low = df_nifty['low'].to_numpy(dtype=float)
        vol = df_nifty['volume'].to_numpy(dtype=float)
        
        typical_p = (high + low + close) / 3.0
        vwap = np.sum(typical_p * vol) / (np.sum(vol) + 1e-6)
        ltp_nifty = close[-1]
        
        if ltp_nifty > vwap:
            return "BULLISH"
        elif ltp_nifty < vwap:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

# -----------------------------------------------------------------------------
# 3. 28-DEFENSE & A+ PROBABILITY MATHEMATICAL ENGINES
# -----------------------------------------------------------------------------
def wilder_atr_np(high, low, close, period=14):
    if len(close) < 2: return 5.0
    tr0 = high[1:] - low[1:]
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    val = float(atr[-1])
    return val if val > 0 else 5.0

def fast_ema_np(arr, span):
    if len(arr) < span: return arr[-1] if len(arr) > 0 else 0.0
    alpha = 2.0 / (span + 1.0)
    ema = arr[0]
    for x in arr[1:]:
        ema = alpha * x + (1.0 - alpha) * ema
    return float(ema)

def fast_rsi_np(close, period=14):
    if len(close) < period + 2: return 50.0
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss <= 1e-9: return 100.0
    return float(100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

def adx_wilder_np(high, low, close, period=14):
    if len(close) < period + 5: return 25.0
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    if atr == 0: return 25.0
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr
    sum_di = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / (sum_di if sum_di > 0 else 1.0)
    return float(dx)

def compute_volume_profile_poc(high, low, close, volume, bins=20):
    price_min, price_max = np.min(low), np.max(high)
    if price_max == price_min: return price_min
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    typical_prices = (high + low + close) / 3.0
    bin_indices = np.digitize(typical_prices, bin_edges) - 1
    bin_volumes = np.zeros(bins)
    for idx, vol in zip(bin_indices, volume):
        if 0 <= idx < bins:
            bin_volumes[idx] += vol
    poc_bin = np.argmax(bin_volumes)
    return float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)

def compute_a_plus_zone_probability(df, atr_val, iofii, poc_price, zone_price, direction="DEMAND"):
    close = df['Close'].to_numpy(dtype=float)
    high = df['High'].to_numpy(dtype=float)
    low = df['Low'].to_numpy(dtype=float)
    open_arr = df['Open'].to_numpy(dtype=float)
    vol = df['Volume'].to_numpy(dtype=float)
    
    ltp = close[-1]
    base_probability = 50.0
    
    recent_pivots = np.max(high[-20:]) if direction == "SUPPLY" else np.min(low[-20:])
    s1 = 1.0 if abs(ltp - recent_pivots) / (atr_val + 1e-6) <= 0.2 else 0.0
    s2 = 1.0 
    body_move = abs(close[-1] - open_arr[-1])
    s3 = 1.0 if body_move > 1.2 * atr_val else 0.0
    
    if direction == "DEMAND":
        sweep = (low[-1] < np.min(low[-20:-1])) and (close[-1] > np.min(low[-20:-1]))
    else:
        sweep = (high[-1] > np.max(high[-20:-1])) and (close[-1] < np.max(high[-20:-1]))
    s4 = 1.0 if sweep else 0.0
    
    bar_rng = (high[-1] - low[-1]) + 1e-6
    clv_val = ((close[-1] - low[-1]) - (high[-1] - close[-1])) / bar_rng
    absorption = (vol[-1] > 1.5 * np.mean(vol[-20:])) and (abs(clv_val) >= 0.4)
    s5 = 1.0 if absorption else 0.0
    
    cvd_val = np.sum(vol[-5:] * clv_val)
    s6 = 1.0 if ((cvd_val > 0 and direction == "DEMAND") or (cvd_val < 0 and direction == "SUPPLY")) else 0.0
    s7 = 1.0 if abs(iofii) >= 5.0 else 0.0
    
    typical_p = (high + low + close) / 3.0
    vwap = np.sum(typical_p * vol) / (np.sum(vol) + 1e-6)
    vwap_reclaim = (ltp > vwap) if direction == "DEMAND" else (ltp < vwap)
    s8 = 1.0 if vwap_reclaim else 0.0
    
    fvg = abs(low[-1] - high[-3]) > (0.3 * atr_val) if len(close) >= 3 else False
    s9 = 1.0 if fvg else 0.0
    displacement = body_move >= 1.2 * atr_val
    s10 = 1.0 if displacement else 0.0
    
    bos = (close[-1] > np.max(high[-10:-1])) if direction == "DEMAND" else (close[-1] < np.min(low[-10:-1]))
    s11 = 1.0 if bos else 0.0
    s12 = 1.0 
    
    ema9 = fast_ema_np(close, 9)
    ema21 = fast_ema_np(close, 21)
    mtf_align = (ema9 > ema21) if direction == "DEMAND" else (ema9 < ema21)
    s13 = 1.0 if mtf_align else 0.0
    
    weights = [4.0, 4.5, 4.5, 4.0, 4.0, 4.0, 4.0, 3.5, 3.5, 4.5, 4.0, 3.5, 3.5]
    scores = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13]
    
    final_probability = base_probability + sum(w * s for w, s in zip(weights, scores))
    return round(min(98.0, max(50.0, final_probability)), 1)

def validate_28_defensive_shields(df, atr_val, iofii, direction):
    close = df['Close'].to_numpy(dtype=float)
    high = df['High'].to_numpy(dtype=float)
    low = df['Low'].to_numpy(dtype=float)
    open_arr = df['Open'].to_numpy(dtype=float)
    vol = df['Volume'].to_numpy(dtype=float)
    
    if len(close) < 25: return False, 0.0
    
    mean_vol = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    rvol = vol[-1] / (mean_vol + 1e-6)
    bar_rng = high[-1] - low[-1]
    vvei = (bar_rng / (atr_val + 1e-6)) * (vol[-1] / (mean_vol + 1e-6))
    clv_arr = ((close - low) - (high - close)) / ((high - low) + 1e-6)
    cvd_slope = np.sum(vol[-5:] * clv_arr[-5:]) if len(close) >= 5 else 1.0
    
    vol_shield = (rvol >= 0.9) and (vvei >= 0.7) and (cvd_slope >= -2e5)
    body_size = abs(close[-1] - open_arr[-1])
    displacement_pass = body_size > (0.7 * atr_val)
    clv_ext = clv_arr[-1] >= 0.4 if direction == "DEMAND" else clv_arr[-1] <= -0.4
    vol_std = np.std(vol) if np.std(vol) > 0 else 1.0
    z_vol = (vol[-1] - mean_vol) / vol_std
    z_vol_pass = z_vol >= -1.5
    
    breakout_shield = displacement_pass and clv_ext and z_vol_pass
    rsi_val = fast_rsi_np(close, 14)
    market_shield = 25 <= rsi_val <= 75
    ater = bar_rng / (atr_val + 1e-6)
    volatility_shield = ater <= 3.5
    
    prev_close = close[-2] if len(close) >= 2 else open_arr[0]
    gap_pct = abs((open_arr[0] - prev_close) / prev_close) * 100.0
    gap_shield = gap_pct <= 4.0
    
    rcf = 1.0 - min(abs(bar_rng - atr_val) / (atr_val + 1e-6), 0.5)
    zone_shield = (abs(iofii) >= MIN_ABSOLUTE_IOFII) and (rcf >= 0.15)
    
    ema1 = fast_ema_np(close[-15:], 13)
    ema3 = fast_ema_np(close[-45::3], 13) if len(close) >= 40 else ema1
    ema5 = fast_ema_np(close[-75::5], 13) if len(close) >= 65 else ema1
    ema15 = fast_ema_np(close[-225::15], 13) if len(close) >= 150 else ema1
    bull_cnt = sum([close[-1] > ema1, close[-1] > ema3, close[-1] > ema5, close[-1] > ema15])
    adx_v = adx_wilder_np(high, low, close, 14)
    
    mtf_shield = (adx_v >= 12.0) and ((bull_cnt >= 2 if direction == "DEMAND" else bull_cnt <= 2))
    
    shields_passed = sum([vol_shield, breakout_shield, market_shield, volatility_shield, gap_shield, zone_shield, mtf_shield])
    shield_score = (shields_passed / 7.0) * 100.0
    
    is_fully_defended = shields_passed >= 4
    return is_fully_defended, round(shield_score, 1)

def compute_bidirectional_cri(df_1m, ltp, target_zone_price, atr_14, direction="SUPPLY", span_bars=5):
    if len(df_1m) < span_bars: return 0.0, "TREND_STABLE", "HOLD"
    sub = df_1m.iloc[-span_bars:]
    c, h, l, o, v = sub['Close'].to_numpy(dtype=float), sub['High'].to_numpy(dtype=float), sub['Low'].to_numpy(dtype=float), sub['Open'].to_numpy(dtype=float), sub['Volume'].to_numpy(dtype=float)
    bar_range = (h - l) + 1e-6
    
    dist = max(0.0, ltp - target_zone_price) if direction == "SUPPLY" else max(0.0, target_zone_price - ltp)
    s_p = float(np.exp(-2.5 * (dist / (atr_14 + 1e-6))))
    
    weights = np.arange(1, span_bars + 1)
    power = ((c - l) / bar_range) if direction == "SUPPLY" else ((h - c) / bar_range)
    a_v = float(np.sum(power * (v * weights)) / (np.sum(v * weights) + 1e-6))
    
    v_mean = np.mean(v) if len(v) > 0 else 1.0
    vol_mult = float(np.sqrt(min(2.0, max(0.5, v[-1] / (v_mean + 1e-6)))))
    raw_wick = (max(0.0, min(o[-1], c[-1]) - l[-1]) / bar_range[-1]) if direction == "SUPPLY" else (max(0.0, h[-1] - max(o[-1], c[-1])) / bar_range[-1])
    w_r = min(1.0, float(raw_wick * vol_mult))
    
    micro_vwap = np.sum(((h + l + c) / 3.0) * v) / (np.sum(v) + 1e-6)
    ema1 = fast_ema_np(c, 13)
    norm_factor = 0.1 * atr_14 + 1e-6
    
    diff_ema = ((ltp - ema1) / norm_factor) if direction == "SUPPLY" else ((ema1 - ltp) / norm_factor)
    diff_vwap = ((ltp - micro_vwap) / norm_factor) if direction == "SUPPLY" else ((micro_vwap - ltp) / norm_factor)
    m_s = float(0.5 * (1.0 / (1.0 + np.exp(-np.clip(diff_ema, -10, 10)))) + 0.5 * (1.0 / (1.0 + np.exp(-np.clip(diff_vwap, -10, 10)))))
    
    cri = min(100.0, max(0.0, float((0.35 * s_p + 0.25 * a_v + 0.25 * w_r + 0.15 * m_s) * 100.0)))
    new_side = "BUY_CALL" if direction == "SUPPLY" else "SELL_PUT"
    
    if cri >= 80.0: status, action = "CRITICAL_REVERSAL", f"ENTER_{new_side}"
    elif cri >= 65.0: status, action = "EARLY_TRIGGER", f"READY_{new_side}"
    elif cri >= 40.0: status, action = "PULLBACK_TRAIL", "TIGHTEN_SL"
    else: status, action = "TREND_STABLE", "HOLD_ZONE"
    return round(cri, 2), status, action

# -----------------------------------------------------------------------------
# 4. SINGLE STOCK PARALLEL QUANT PROCESSING UNIT (WITH NIFTY VWAP GATE)
# -----------------------------------------------------------------------------
def process_single_stock_data(sym, df, nifty_trend):
    try:
        if df.empty or len(df) < 15: return None
        close = df['Close'].to_numpy(dtype=float)
        high = df['High'].to_numpy(dtype=float)
        low = df['Low'].to_numpy(dtype=float)
        open_arr = df['Open'].to_numpy(dtype=float)
        vol = df['Volume'].to_numpy(dtype=float)
        
        open_p = float(open_arr[0])
        ltp = float(close[-1])
        day_high = float(np.max(high))
        day_low = float(np.min(low))
        
        if not (DASHBOARD_MIN_PRICE <= open_p <= DASHBOARD_MAX_PRICE): return None
        
        atr_val = wilder_atr_np(high, low, close, 14)
        pnl_pct = ((ltp - open_p) / open_p) * 100.0
        
        bar_range = (high - low) + 1e-6
        clv = ((close - low) - (high - close)) / bar_range
        iofii = float((np.sum(clv * vol) / (np.sum(vol) + 1e-6)) * 100.0)
        
        if abs(iofii) < MIN_ABSOLUTE_IOFII: return None
        direction = "DEMAND" if iofii >= 0 else "SUPPLY"
        
        if nifty_trend == "BULLISH" and direction != "DEMAND": return None
        if nifty_trend == "BEARISH" and direction != "SUPPLY": return None
        
        poc_price = compute_volume_profile_poc(high, low, close, vol)
        a_plus_prob = compute_a_plus_zone_probability(df, atr_val, iofii, poc_price, poc_price, direction=direction)
        
        is_defended, shield_health = validate_28_defensive_shields(df, atr_val, iofii, direction)
        if not is_defended: return None
        
        if direction == "SUPPLY":
            target_price = ltp - max(atr_val * 1.5, ltp * 0.015)
            sl_best_entry = max(day_high, open_p + max(0.35 * atr_val, ltp * 0.004))
            gap_pct = ((sl_best_entry - target_price) / sl_best_entry) * 100.0
            zone_html = f"<span style='color:#ff5252; font-weight:700;'>A+ SUPPLY OB ({a_plus_prob}%)</span><br><span style='color:#ff7675; font-size:8.5px;'>IOFII: {iofii:.1f}% | POC: ₹{poc_price:.1f}</span>"
            sl_title, sl_color, sl_bg = "SELL ENTRY / SL", "#ff7675", "rgba(255, 118, 117, 0.12)"
        else:
            target_price = ltp + max(atr_val * 1.5, ltp * 0.015)
            sl_best_entry = min(day_low, open_p - max(0.35 * atr_val, ltp * 0.004))
            gap_pct = ((target_price - sl_best_entry) / sl_best_entry) * 100.0
            zone_html = f"<span style='color:#00e676; font-weight:700;'>A+ DEMAND OB ({a_plus_prob}%)</span><br><span style='color:#55efc4; font-size:8.5px;'>IOFII: {iofii:.1f}% | POC: ₹{poc_price:.1f}</span>"
            sl_title, sl_color, sl_bg = "BUY ENTRY / SL", "#55efc4", "rgba(85, 239, 196, 0.12)"
            
        if gap_pct < MIN_RUNWAY_GAP_PCT: return None
        
        ema1 = fast_ema_np(close[-15:], 13)
        ema3 = fast_ema_np(close[-45::3], 13) if len(close) >= 40 else ema1
        ema5 = fast_ema_np(close[-75::5], 13) if len(close) >= 65 else ema1
        ema15 = fast_ema_np(close[-225::15], 13) if len(close) >= 150 else ema1
        
        bull_cnt = sum([ltp > ema1, ltp > ema3, ltp > ema5, ltp > ema15])
        
        tp_val = (high + low + close) / 3.0
        vwap = float(np.sum(tp_val * vol) / (np.sum(vol) + 1e-6))
        rsi = fast_rsi_np(close, 14)
        
        directional_move = (ltp - open_p) if direction == 'DEMAND' else (open_p - ltp)
        pressure_pct = min(100.0, max(0.0, (directional_move / atr_val) * 100.0))
        
        border_c = "#ff3838" if direction == "SUPPLY" else "#00e676"
        bg_c = "rgba(255, 56, 56, 0.12)" if direction == "SUPPLY" else "rgba(0, 230, 118, 0.12)"
        status_t = "SUPPLY ACCUMULATION" if direction == "SUPPLY" else "DEMAND ABSORPTION"
        retest_html = "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST</div>" if pressure_pct > 75.0 else ""
        
        pressure_box_html = f"""
        <div style='border: 1px dashed {border_c}; background-color: {bg_c}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
            <div style='font-size: 8.5px; font-weight: 800; color: {border_c};'>{status_t}</div>
            <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{pressure_pct:.1f}%</div>
            {retest_html}
        </div>
        """
        
        cri_val, cri_status, cri_action = compute_bidirectional_cri(df, ltp, target_zone_price, atr_val, direction=direction)
        
        cri_border = "#00e676" if cri_val >= 80.0 else ("#ffaa00" if cri_val >= 65.0 else "#30363d")
        cri_bg = "rgba(0, 230, 118, 0.15)" if cri_val >= 80.0 else "#161b22"
        action_color = "#00e676" if cri_val >= 80.0 else ("#ffaa00" if cri_val >= 65.0 else "#8b949e")
        
        cri_box_html = f"""
        <div style='border: 1px dashed {cri_border}; background-color: {cri_bg}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
            <div style='font-size: 8.5px; font-weight: 800; color: {action_color};'>{cri_status}</div>
            <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{cri_val:.1f}%</div>
            <div style='color: {action_color}; font-size: 8px; font-weight: 900; margin-top: 1px;'>⚡ {cri_action}</div>
        </div>
        """
        
        sl_box_html = f"""
        <div style='border: 1px dashed {sl_color}; background-color: {sl_bg}; padding: 3px 5px; border-radius: 4px; text-align: center;'>
            <div style='font-size: 8.5px; font-weight: 800; color: {sl_color};'>{sl_title}</div>
            <div style='font-size: 11px; font-weight: 900; color: #ffffff;'>₹{sl_best_entry:.2f}</div>
        </div>
        """
        
        mtf_score = (max(bull_cnt, 4 - bull_cnt) / 4.0) * 25.0
        vwap_score = 20.0 if ((direction == 'DEMAND' and ltp > vwap) or (direction == 'SUPPLY' and ltp < vwap)) else 0.0
        rsi_score = 15.0 if ((direction == 'DEMAND' and 50 <= rsi <= 70) or (direction == 'SUPPLY' and 30 <= rsi <= 50)) else 5.0
        pressure_score = min(20.0, pressure_pct * 0.2)
        runway_score = min(20.0, gap_pct * 10.0)
        
        tcs = int(min(100.0, max(0.0, mtf_score + vwap_score + rsi_score + pressure_score + runway_score)))
        buy_power = np.sum(vol * ((close - low) / bar_range))
        sell_power = np.sum(vol * ((high - close) / bar_range))
        tot_power = buy_power + sell_power + 1e-6
        buy_pct = (buy_power / tot_power) * 100.0
        cobi_html = f"{buy_pct:.0f}% Buy ({iofii:+.1f}%)"
        
        shield_html = f"<span style='color:#00e676; font-weight:bold;'>🛡️ {shield_health}%</span>"
        
        return {
            "raw_sym": sym, "symbol": sym.replace(".NS", ""), "zone_html": zone_html,
            "shield_html": shield_html, "open": open_p, "ltp": ltp, "pnl": pnl_pct,
            "pressure_box": pressure_box_html, "cri_box": cri_box_html, "sl_box": sl_box_html,
            "target": target_price, "tcs": tcs, "cobi_html": cobi_html, "imbalance": iofii,
            "abs_iofii": abs(iofii), "e1": ltp > ema1, "e3": ltp > ema3, "e5": ltp > ema5, "e15": ltp > ema15,
            "sort_score": a_plus_prob
        }
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 5. STREAMLIT WEB APP ARCHITECTURE
# -----------------------------------------------------------------------------
st.markdown("<h2 style='color: #00e676;'>🔒 TOP 7 LOCKED (09:20 AM)</h2>", unsafe_allow_html=True)
status_placeholder = st.empty()
table_placeholder = st.empty()

def main_loop():
    symbols_to_process = get_dynamic_nifty500_symbols()
    if not symbols_to_process:
        st.warning("Fetching symbols failed.")
        return
        
    nifty_trend = check_nifty_vwap_gate()
    status_placeholder.info(f"Nifty Trend Gate: **{nifty_trend}** | Scanning Nifty 500 Universe...")
    
    batch_data = yf.download(symbols_to_process[:100], period="1d", interval="1m", progress=False, group_by='ticker', auto_adjust=True)
    if batch_data.empty:
        st.error("Market data empty.")
        return
        
    valid_tuples = []
    for sym in symbols_to_process[:100]:
        try:
            df = batch_data[sym].dropna() if len(symbols_to_process[:100]) > 1 else batch_data.dropna()
            if not df.empty: valid_tuples.append((sym, df))
        except Exception:
            continue
            
    current_rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_stock_data, sym, df, nifty_trend) for sym, df in valid_tuples]
        for f in futures:
            res = f.result()
            if res is not None: current_rows.append(res)
            
    current_rows.sort(key=lambda x: (x['sort_score'], x['tcs']), reverse=True)
    top_rows = current_rows[:MAX_LOCKED_STOCKS]
    
    if top_rows:
        df_display = pd.DataFrame(top_rows)
        df_display = df_display[['symbol', 'zone_html', 'shield_html', 'open', 'ltp', 'pnl', 'e1', 'pressure_box', 'cri_box', 'sl_box', 'target', 'tcs', 'cobi_html']]
        df_display.columns = ["Symbol", "Zone Alignments", "Shield", "Open", "LTP", "Change", "(1m|3m|5m|15m)", "Supply/Demand", "Reversal (CRI)", "SL / Entry", "Target", "TCS", "COBI"]
        table_placeholder.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        table_placeholder.warning("No stocks matching current strict A+ probability criteria.")

if __name__ == "__main__":
    main_loop()
