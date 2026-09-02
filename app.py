import time
import datetime
import warnings
import concurrent.futures
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION (Mobile Responsive & Dark Mode Engine)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ SMC QUANT LIVE ENGINE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MIN_PRICE = 300.0
MAX_PRICE = 600.0

# -----------------------------------------------------------------------------
# 1. DYNAMIC UNIVERSE SCANNER (Free & Direct Market Scanner)
# -----------------------------------------------------------------------------
def fetch_dynamic_universe():
    dynamic_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BPCL.NS", 
        "NAVA.NS", "AIIL.NS", "IGIL.NS", "AADHARHFC.NS", "CONCOR.NS", "POONAWALLA.NS", 
        "GMDCLTD.NS", "LICHSGFIN.NS", "JSWINFRA.NS", "REDINGTON.NS", "ASHOKLEY.NS", 
        "FEDERALBNK.NS", "IDFCFIRSTB.NS", "CROMPTON.NS", "NATIONALUM.NS", "RBLBANK.NS",
        "NUVOCO.NS", "COALINDIA.NS", "VBL.NS", "SYNGENE.NS", "ELECON.NS", "JINDALSAW.NS",
        "TATAPOWER.NS", "JSWENERGY.NS", "USHAMART.NS", "NTPC.NS", "ICICIPRULI.NS"
    ]
    return list(set(dynamic_tickers))

# -----------------------------------------------------------------------------
# 2. QUANT MATHEMATICS & ADVANCED SMC ENGINE
# -----------------------------------------------------------------------------
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def extract_advanced_zones(df: pd.DataFrame, pivot_len: int = 2):
    if len(df) < 25: return [], []
    df = df.copy()
    high, low, open_p, close, vol = df['High'].values, df['Low'].values, df['Open'].values, df['Close'].values, df['Volume'].values
    atr = calculate_atr(df, 14).values
    vol_sma = df['Volume'].rolling(20).mean().values
    
    demand_boxes, supply_boxes = [], []
    
    for i in range(pivot_len, len(df) - pivot_len):
        mid = i
        vsa_valid = (vol[mid] > 1.2 * vol_sma[mid]) if not np.isnan(vol_sma[mid]) else True
        
        if high[mid] == max(high[mid - pivot_len : mid + pivot_len + 1]) and vsa_valid:
            top_lvl, bot_lvl = high[mid], max(open_p[mid], close[mid])
            if not any(abs(b['top'] - top_lvl) < (atr[mid] * 0.3) for b in supply_boxes if b['active']):
                supply_boxes.append({'top': top_lvl, 'bot': bot_lvl, 'tests': 0, 'active': True})
                
        if low[mid] == min(low[mid - pivot_len : mid + pivot_len + 1]) and vsa_valid:
            top_lvl, bot_lvl = min(open_p[mid], close[mid]), low[mid]
            if not any(abs(b['bot'] - bot_lvl) < (atr[mid] * 0.3) for b in demand_boxes if b['active']):
                demand_boxes.append({'top': top_lvl, 'bot': bot_lvl, 'tests': 0, 'active': True})
                
        cur_h, cur_l = high[mid + pivot_len], low[mid + pivot_len]
        for b in supply_boxes:
            if b['active']:
                if cur_h > b['bot']: b['tests'] += 1
                if cur_h > b['top'] or b['tests'] >= 3: b['active'] = False
                    
        for b in demand_boxes:
            if b['active']:
                if cur_l < b['top']: b['tests'] += 1
                if cur_l < b['bot'] or b['tests'] >= 3: b['active'] = False

    return [b for b in demand_boxes if b['active']], [b for b in supply_boxes if b['active']]

def get_enhanced_ema13_signal(df: pd.DataFrame):
    if len(df) < 15: return "NEUTRAL"
    close = df['Close'].values
    ema13 = df['Close'].ewm(span=13, adjust=False).mean().values
    
    curr_close = close[-1]
    curr_ema = ema13[-1]
    prev_ema = ema13[-2]
    
    if curr_close > curr_ema and curr_ema > prev_ema:
        return "BULLISH"
    elif curr_close < curr_ema and curr_ema < prev_ema:
        return "BEARISH"
    return "NEUTRAL"

# -----------------------------------------------------------------------------
# 3. ADVANCED QUANT ENGINE (TCS, COBI, TARGETS & PRESSURE DELTA)
# -----------------------------------------------------------------------------
def calculate_trade_clearance_score(df_live, df_1d, open_p, current_price, mtf_score, supply_pct):
    try:
        atr = calculate_atr(df_1d, 14).iloc[-1]
        prev_close = df_1d['Close'].iloc[-2] if len(df_1d) > 1 else open_p
        
        gap_diff = abs(open_p - prev_close)
        s_gap = 100 * max(0, 1 - (gap_diff / (atr if atr > 0 else 1)))
        s_supply = 100 * min(1.0, (supply_pct / 3.0))
        
        vol_curr = df_live['Volume'].sum()
        vol_avg = df_1d['Volume'].mean() / 375 
        rovl = vol_curr / (vol_avg * len(df_live) + 1e-5)
        s_vol = min(100, rovl * 50)
        
        s_mtf = min(100, mtf_score * 25)
        s_regime = 100 if current_price > df_1d['Close'].ewm(span=13).mean().iloc[-1] else 0
        
        tcs = (0.25 * s_gap) + (0.25 * s_supply) + (0.20 * s_vol) + (0.15 * s_mtf) + (0.15 * s_regime)
        return min(100.0, max(0.0, tcs))
    except Exception:
        return 50.0

def calculate_cobi_and_imbalance(df_live):
    if len(df_live) < 2: return 50.0, 0.0, 50.0
    
    close = df_live['Close'].values
    open_p = df_live['Open'].values
    vol = df_live['Volume'].values
    
    buying_vol = np.sum(vol[close >= open_p])
    selling_vol = np.sum(vol[close < open_p])
    total_vol = buying_vol + selling_vol + 1e-5
    
    buyer_ratio = (buying_vol / total_vol) * 100
    seller_ratio = (selling_vol / total_vol) * 100
    imbalance_delta_pct = ((buying_vol - selling_vol) / total_vol) * 100
    cobi = buyer_ratio - seller_ratio
    
    return buyer_ratio, imbalance_delta_pct, cobi

def calculate_projected_target(current_price, atr_val, direction="SELL"):
    target_distance = atr_val * 1.618
    if direction == "SELL":
        return current_price - target_distance
    return current_price + target_distance

# -----------------------------------------------------------------------------
# 4. THREAD WORKERS
# -----------------------------------------------------------------------------
def scan_initial_universe(ticker):
    try:
        stock = yf.Ticker(ticker)
        df_1d = stock.history(period="30d", interval="1d")
        if df_1d.empty or len(df_1d) < 15: return None
        
        open_price = df_1d['Open'].iloc[-1]
        if not (MIN_PRICE <= open_price <= MAX_PRICE): return None
        
        df_15m = stock.history(period="7d", interval="15m")
        df_1h = stock.history(period="15d", interval="60m")
        
        tf_zones = {
            '15m': extract_advanced_zones(df_15m) if not df_15m.empty else ([], []),
            '1H': extract_advanced_zones(df_1h) if not df_1h.empty else ([], []),
            '1D': extract_advanced_zones(df_1d)
        }
        
        matched_tf = []
        confluence_score = 0
        
        for tf_name, (dem_boxes, sup_boxes) in tf_zones.items():
            for b in dem_boxes:
                if b['bot'] <= open_price <= b['top']: 
                    matched_tf.append(f"<span style='color:#00e676; font-weight:700;'>DEMAND ({tf_name})</span>")
                    confluence_score += 1
            for b in sup_boxes:
                if b['bot'] <= open_price <= b['top']: 
                    matched_tf.append(f"<span style='color:#ff5252; font-weight:700;'>SUPPLY ({tf_name})</span>")
                    confluence_score += 1
                
        if matched_tf:
            return {
                "Ticker": ticker,
                "Symbol": ticker.replace(".NS", ""),
                "Open Price": open_price,
                "Zones": " | ".join(matched_tf),
                "Score": confluence_score,
                "df_1d": df_1d
            }
        else:
            # Persistent Zone Monitor Fallback
            dem_1d, sup_1d = tf_zones['1D']
            if dem_1d or sup_1d:
                tag = f"<span style='color:#00e676; font-weight:700;'>DEMAND (1D)</span>" if dem_1d else f"<span style='color:#ff5252; font-weight:700;'>SUPPLY (1D)</span>"
                return {
                    "Ticker": ticker,
                    "Symbol": ticker.replace(".NS", ""),
                    "Open Price": open_price,
                    "Zones": tag,
                    "Score": 1,
                    "df_1d": df_1d
                }
    except Exception:
        return None

def fetch_live_updates(stock_info):
    ticker = stock_info["Ticker"]
    try:
        stock = yf.Ticker(ticker)
        df_live = stock.history(period="1d", interval="1m")
        df_3m = stock.history(period="3d", interval="2m")
        df_5m = stock.history(period="5d", interval="5m")
        df_15m = stock.history(period="5d", interval="15m")
        
        # Cloud/Off-market fallback to ensure continuous data render
        if df_live is None or df_live.empty:
            df_live = stock_info["df_1d"].tail(10)
        
        current_price = df_live['Close'].iloc[-1]
        open_p = stock_info["Open Price"]
        pnl_pct = ((current_price - open_p) / open_p) * 100
        
        ema_1m = get_enhanced_ema13_signal(df_live)
        ema_3m = get_enhanced_ema13_signal(df_3m if (df_3m is not None and not df_3m.empty) else df_live)
        ema_5m = get_enhanced_ema13_signal(df_5m if (df_5m is not None and not df_5m.empty) else df_live)
        ema_15m = get_enhanced_ema13_signal(df_15m if (df_15m is not None and not df_15m.empty) else df_live)

        def dot_badge(status):
            if status == "BULLISH": return "<span title='Bullish' style='color:#00e676; font-size:16px;'>🟢</span>"
            elif status == "BEARISH": return "<span title='Bearish' style='color:#ff5252; font-size:16px;'>🔴</span>"
            return "<span title='Neutral' style='color:#484f58; font-size:14px;'>●</span>"

        buyer_pct, imbalance_delta, cobi = calculate_cobi_and_imbalance(df_live)
        atr_val = calculate_atr(stock_info["df_1d"], 14).iloc[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = current_price * 0.015
            
        supply_pressure_pct = min(100.0, max(0.0, (abs(current_price - open_p) / atr_val) * 100))
        
        if "SUPPLY" in stock_info["Zones"]:
            border_color = "#ff3838"
            bg_color = "rgba(255, 56, 56, 0.12)"
            status_txt = "SUPPLY ACCUMULATION"
            proj_target = calculate_projected_target(current_price, atr_val, "SELL")
        else:
            border_color = "#00e676"
            bg_color = "rgba(0, 230, 118, 0.12)"
            status_txt = "DEMAND ABSORPTION"
            proj_target = calculate_projected_target(current_price, atr_val, "BUY")

        retest_alert = ""
        if supply_pressure_pct > 75.0:
            retest_alert = "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST REJECTION</div>"

        pressure_box_html = f"<div style='border: 1.5px solid {border_color}; background-color: {bg_color}; padding: 4px 6px; border-radius: 5px; text-align: center;'><div style='font-size: 10px; font-weight: 800; color: {border_color};'>{status_txt}</div><div style='font-size: 12px; font-weight: 900; color: #ffffff;'>{supply_pressure_pct:.1f}%</div>{retest_alert}</div>"

        tcs_score = calculate_trade_clearance_score(
            df_live, stock_info["df_1d"], open_p, current_price, stock_info["Score"], supply_pressure_pct
        )
        tcs_color = "#00e676" if tcs_score >= 65 else ("#ffaa00" if tcs_score >= 40 else "#ff5252")
        tcs_html = f"<span style='color:{tcs_color}; font-weight:900;'>{tcs_score:.0f}/100</span>"
        change_color = "#00e676" if pnl_pct >= 0 else "#ff5252"

        return {
            "symbol": stock_info['Symbol'],
            "zones": stock_info["Zones"],
            "open": f"₹{open_p:.2f}",
            "ltp": f"₹{current_price:.2f}",
            "change": f"<span style='color:{change_color}; font-weight:bold;'>{pnl_pct:+.2f}%</span>",
            "ema1m": dot_badge(ema_1m),
            "ema3m": dot_badge(ema_3m),
            "ema5m": dot_badge(ema_5m),
            "ema15m": dot_badge(ema_15m),
            "pressure_box": pressure_box_html,
            "target": f"<span style='color:#00e676; font-weight:700;'>₹{proj_target:.2f}</span>",
            "tcs": tcs_html,
            "cobi": f"<span style='color:{'#00e676' if imbalance_delta >= 0 else '#ff5252'}'>{buyer_pct:.0f}% Buy ({imbalance_delta:+.1f}%)</span>",
            "_score": stock_info["Score"]
        }
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 5. ZERO-FLICKER DASHBOARD RENDERER & PERSISTENT STREAM
# -----------------------------------------------------------------------------
st.markdown("""
<style>
header, #MainMenu, footer {visibility: hidden !important; display: none !important;}
.block-container {
    padding: 0rem !important;
    max-width: 100% !important;
}
iframe {
    width: 100% !important;
    border: none !important;
    background-color: #090c10 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_universe():
    universe = fetch_dynamic_universe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(scan_initial_universe, universe))
        locked = [r for r in results if r is not None]
    return sorted(locked, key=lambda x: x['Score'], reverse=True)

locked_universe = get_cached_universe()

if 'cached_live' not in st.session_state:
    st.session_state.cached_live = []

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    fresh_data = list(executor.map(fetch_live_updates, locked_universe))
    valid_data = [d for d in fresh_data if d is not None]
    if valid_data:
        st.session_state.cached_live = sorted(valid_data, key=lambda x: x['_score'], reverse=True)

display_data = [dict(r) for r in st.session_state.cached_live]
for row in display_data:
    if '_score' in row:
        del row['_score']

payload = {
    "timestamp": datetime.datetime.now().strftime('%H:%M:%S'),
    "data": display_data
}

base_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    body {{
        background-color: #090c10;
        margin: 0;
        padding: 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #ffffff !important;
    }}
    .quant-container {{
        background-color: #090c10;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 10px;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
    .quant-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1f293d;
        padding-bottom: 8px;
        margin-bottom: 10px;
    }}
    .quant-title {{
        color: #00e676;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }}
    .quant-clock {{
        color: #8b949e;
        font-size: 11px;
        background: #161b22;
        padding: 4px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }}
    .quant-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        white-space: nowrap;
    }}
    .quant-table th {{
        background-color: #161b22;
        color: #8b949e;
        text-align: center;
        padding: 8px 6px;
        border-bottom: 2px solid #21262d;
        font-size: 10px;
        text-transform: uppercase;
    }}
    .quant-table td {{
        padding: 8px 6px;
        border-bottom: 1px solid #161b22;
        color: #ffffff !important;
        font-weight: 600;
        text-align: center;
        vertical-align: middle;
    }}
    .quant-table tr:nth-child(even) {{ background-color: #0d1117; }}
    .quant-table tr:nth-child(odd) {{ background-color: #090c10; }}
    .symbol-text {{
        color: #ffffff !important;
        font-weight: 800;
        font-size: 13px;
        text-align: left !important;
    }}
</style>
</head>
<body>
    <div class="quant-container">
        <div class="quant-header">
            <div class="quant-title">⚡ FREE QUANT ENGINE | SMC LIVE PRESSURE DASHBOARD</div>
            <div id="quant-clock-val" class="quant-clock">LIVE STREAM: {payload['timestamp']} IST</div>
        </div>
        <table class="quant-table">
            <thead>
                <tr>
                    <th style="text-align:left;">Symbol</th>
                    <th style="text-align:left;">Zone Alignments</th>
                    <th>Open</th>
                    <th>LTP</th>
                    <th>Change</th>
                    <th>EMAs (1m|3m|5m|15m)</th>
                    <th>Supply/Demand Delta Box</th>
                    <th>Target (₹)</th>
                    <th>TCS Score</th>
                    <th>Buyer/Seller (COBI)</th>
                </tr>
            </thead>
            <tbody id="quant-table-body"></tbody>
        </table>
    </div>

    <script>
        const payload = {json.dumps(payload)};
        document.getElementById('quant-clock-val').innerText = 'LIVE STREAM: ' + payload.timestamp + ' IST';
        const tbody = document.getElementById('quant-table-body');
        let html = '';
        payload.data.forEach(row => {{
            html += `<tr>
                <td class="symbol-text">${{row.symbol}}</td>
                <td style="text-align:left;">${{row.zones}}</td>
                <td>${{row.open}}</td>
                <td>${{row.ltp}}</td>
                <td>${{row.change}}</td>
                <td>${{row.ema1m}} ${{row.ema3m}} ${{row.ema5m}} ${{row.ema15m}}</td>
                <td>${{row.pressure_box}}</td>
                <td>${{row.target}}</td>
                <td>${{row.tcs}}</td>
                <td>${{row.cobi}}</td>
            </tr>`;
        }});
        tbody.innerHTML = html;
    </script>
</body>
</html>
"""

components.html(base_html, height=780, scrolling=True)

# 0-delay auto sync loop
time.sleep(3)
st.rerun()
