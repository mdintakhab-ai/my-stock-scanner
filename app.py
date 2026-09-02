import time
import datetime
import warnings
import logging
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

# Suppress background logs and warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & FIXES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SMC Live Quant Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Lock entire Streamlit viewport to permanent dark container
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #090c10 !important;
        overflow: hidden !important;
    }
    .block-container {
        padding: 0rem !important;
        margin: 0rem !important;
        max-width: 100% !important;
    }
    header, footer, #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        background-color: #090c10 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. HARD CONFIGURATION & UNIVERSE
# -----------------------------------------------------------------------------
MIN_PRICE = 300.0
MAX_PRICE = 600.0
HIGH_CONVICTION_TCS_THRESHOLD = 65

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BPCL.NS", 
    "NAVA.NS", "CONCOR.NS", "POONAWALLA.NS", "GMDCLTD.NS", "LICHSGFIN.NS", 
    "JSWINFRA.NS", "REDINGTON.NS", "ASHOKLEY.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", 
    "CROMPTON.NS", "NATIONALUM.NS", "RBLBANK.NS", "NUVOCO.NS", "COALINDIA.NS", 
    "VBL.NS", "SYNGENE.NS", "ELECON.NS", "JINDALSAW.NS", "TATAPOWER.NS", 
    "JSWENERGY.NS", "USHAMART.NS", "NTPC.NS", "ICICIPRULI.NS", "BATAINDIA.NS", 
    "CANBK.NS", "DLF.NS", "HINDPETRO.NS", "IOC.NS", "AADHARHFC.NS", "IGIL.NS"
]

# -----------------------------------------------------------------------------
# 2. NUMPY QUANT MATH & TRUE SMC ENGINE
# -----------------------------------------------------------------------------
def fast_atr_1d(high, low, close, period=14):
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

def extract_true_smc_zones(df: pd.DataFrame, tf_name: str, pivot_len: int = 2):
    """
    True Institutional Order Block Engine with FVG (Fair Value Gap) and Displacement Validation.
    """
    if df.empty or len(df) < 20: return []
    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    o = df['Open'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)
    v = df['Volume'].to_numpy(dtype=float)
    v_mean = np.mean(v) if len(v) > 0 else 1.0
    atr_arr = np.abs(h - l)
    
    zones = []
    for i in range(pivot_len, len(c) - pivot_len - 1):
        mid = i
        current_atr = atr_arr[mid] if atr_arr[mid] > 0 else 1.0
        is_vol = v[mid] > (1.2 * v_mean)
        
        # Institutional Supply Block: Bearish Displacement + FVG Formation
        if h[mid] == np.max(h[mid - pivot_len : mid + pivot_len + 1]):
            disp_down = (o[mid + 1] - c[mid + 1]) > (0.5 * current_atr) if (mid + 1 < len(c)) else False
            has_fvg = (l[mid] > h[mid + 2]) if (mid + 2 < len(c)) else False
            if is_vol or disp_down or has_fvg:
                zones.append({
                    'type': 'SUPPLY', 
                    'tf': tf_name, 
                    'top': float(h[mid]), 
                    'bot': float(max(o[mid], c[mid])),
                    'origin_idx': mid
                })
                
        # Institutional Demand Block: Bullish Displacement + FVG Formation
        if l[mid] == np.min(l[mid - pivot_len : mid + pivot_len + 1]):
            disp_up = (c[mid + 1] - o[mid + 1]) > (0.5 * current_atr) if (mid + 1 < len(c)) else False
            has_fvg = (h[mid] < l[mid + 2]) if (mid + 2 < len(c)) else False
            if is_vol or disp_up or has_fvg:
                zones.append({
                    'type': 'DEMAND', 
                    'tf': tf_name, 
                    'top': float(min(o[mid], c[mid])), 
                    'bot': float(l[mid]),
                    'origin_idx': mid
                })
                
    return zones[-2:]

# -----------------------------------------------------------------------------
# 3. UNIVERSE INITIALIZATION & CONFLICT-FREE FILTERING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def init_universe_mtf(tickers):
    hist_1d = yf.download(tickers=tickers, period="30d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=True)
    hist_1h = yf.download(tickers=tickers, period="10d", interval="60m", group_by="ticker", threads=True, progress=False, auto_adjust=True)
    hist_15m = yf.download(tickers=tickers, period="5d", interval="15m", group_by="ticker", threads=True, progress=False, auto_adjust=True)

    universe = {}
    for t in tickers:
        try:
            df_1d = hist_1d[t].dropna() if isinstance(hist_1d.columns, pd.MultiIndex) else hist_1d.dropna()
            if df_1d.empty or len(df_1d) < 10: continue
            
            open_today = float(df_1d['Open'].iloc[-1])
            # Strict Filter 1: ₹300 <= Price <= ₹600
            if not (MIN_PRICE <= open_today <= MAX_PRICE): continue
            
            df_1h = hist_1h[t].dropna() if isinstance(hist_1h.columns, pd.MultiIndex) else hist_1h.dropna()
            df_15m = hist_15m[t].dropna() if isinstance(hist_15m.columns, pd.MultiIndex) else hist_15m.dropna()
            
            z_1d = extract_true_smc_zones(df_1d, '1D')
            z_1h = extract_true_smc_zones(df_1h, '1H')
            z_15m = extract_true_smc_zones(df_15m, '15m')
            
            all_zones = z_15m + z_1h + z_1d
            if not all_zones: continue

            # Strict Filter 2: Open Price must lie inside valid zone
            matched_zones = [z for z in all_zones if (z['bot'] * 0.998 <= open_today <= z['top'] * 1.002)]
            if not matched_zones: continue
            
            has_supply = any(z['type'] == 'SUPPLY' for z in matched_zones)
            has_demand = any(z['type'] == 'DEMAND' for z in matched_zones)
            
            # Strict Filter 3: Reject Dual-Zone Conflicts
            if has_supply and has_demand: continue

            atr_val = fast_atr_1d(df_1d['High'].to_numpy(float), df_1d['Low'].to_numpy(float), df_1d['Close'].to_numpy(float), 14)

            # Extract Opposing Targets
            opposing_zones = [z for z in all_zones if z['type'] != ('SUPPLY' if has_supply else 'DEMAND')]

            universe[t] = {
                "symbol": t.replace(".NS", ""),
                "open": open_today,
                "atr": atr_val,
                "zones": matched_zones,
                "opposing_zones": opposing_zones,
                "direction": "SUPPLY" if has_supply else "DEMAND"
            }
        except Exception:
            continue
            
    return universe

# -----------------------------------------------------------------------------
# 4. INITIAL BATCH CALCULATION
# -----------------------------------------------------------------------------
universe = init_universe_mtf(TICKERS)
tickers_list = list(universe.keys())

def compute_all_metrics(universe):
    if not universe: return [], 0
    t0 = time.time()
    batch = yf.download(
        tickers=list(universe.keys()), period="1d", interval="1m",
        group_by="ticker", threads=True, progress=False, auto_adjust=True
    )
    high_conviction_rows = []
    
    for ticker, info in universe.items():
        try:
            df = batch[ticker].dropna() if isinstance(batch.columns, pd.MultiIndex) else batch.dropna()
            if df.empty or len(df) < 5: continue
            
            close = df['Close'].to_numpy(dtype=float)
            high = df['High'].to_numpy(dtype=float)
            low = df['Low'].to_numpy(dtype=float)
            open_arr = df['Open'].to_numpy(dtype=float)
            vol = df['Volume'].to_numpy(dtype=float)
            
            ltp = float(close[-1])
            open_p = info['open']
            atr_val = info['atr']
            pnl_pct = ((ltp - open_p) / open_p) * 100.0
            
            # 1. Background VWAP & Momentum RSI
            tp = (high + low + close) / 3.0
            cum_vol = np.sum(vol) + 1e-6
            vwap = float(np.sum(tp * vol) / cum_vol)
            rsi = fast_rsi(close, 14)
            
            # 2. Accurate 1m, 3m, 5m, 15m EMAs
            ema1 = fast_ema(close[-15:], 13)
            ema3 = fast_ema(close[-45::3], 13) if len(close) >= 40 else ema1
            ema5 = fast_ema(close[-75::5], 13) if len(close) >= 65 else ema1
            ema15 = fast_ema(close[-225::15], 13) if len(close) >= 150 else ema1
            
            bull_cnt = sum([ltp > ema1, ltp > ema3, ltp > ema5, ltp > ema15])
            bear_cnt = 4 - bull_cnt
            
            # 3. Clean Zone Badges
            matched_zones = []
            for z in info['zones']:
                if z['type'] == 'SUPPLY':
                    matched_zones.append(f"<span style='color:#ff5252; font-weight:700;'>SUPPLY ({z['tf']})</span>")
                else:
                    matched_zones.append(f"<span style='color:#00e676; font-weight:700;'>DEMAND ({z['tf']})</span>")
                        
            zone_html = " | ".join(matched_zones)
            
            # 4. Intra-Bar Price-Volume Imbalance (Refined COBI)
            bar_range = (high - low) + 1e-6
            buy_power = np.sum(vol * ((close - low) / bar_range))
            sell_power = np.sum(vol * ((high - close) / bar_range))
            tot_power = buy_power + sell_power + 1e-6
            
            buy_pct = (buy_power / tot_power) * 100.0
            imbalance_pct = ((buy_power - sell_power) / tot_power) * 100.0
            cobi_html = f"{buy_pct:.0f}% Buy ({imbalance_pct:+.1f}%)"
            
            # 5. True Directional Institutional Pressure Index
            directional_move = (ltp - open_p) if info['direction'] == 'DEMAND' else (open_p - ltp)
            pressure_pct = min(100.0, max(0.0, (directional_move / atr_val) * 100.0))
            
            # 6. Advanced Target Engine
            if info['direction'] == "SUPPLY":
                border_c = "#ff3838"
                bg_c = "rgba(255, 56, 56, 0.12)"
                status_t = "SUPPLY ACCUMULATION"
                opp_targets = [z['top'] for z in info['opposing_zones'] if z['top'] < ltp]
                target_price = max(opp_targets) if opp_targets else (ltp - (atr_val * 1.618))
            else:
                border_c = "#00e676"
                bg_c = "rgba(0, 230, 118, 0.12)"
                status_t = "DEMAND ABSORPTION"
                opp_targets = [z['bot'] for z in info['opposing_zones'] if z['bot'] > ltp]
                target_price = min(opp_targets) if opp_targets else (ltp + (atr_val * 1.618))
                
            retest_html = "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST REJECTION</div>" if pressure_pct > 75.0 else ""
            
            pressure_box_html = f"""<div style='border: 1px dashed {border_c}; background-color: {bg_c}; padding: 3px 5px; border-radius: 4px; text-align: center;'><div style='font-size: 9px; font-weight: 800; color: {border_c};'>{status_t}</div><div style='font-size: 11px; font-weight: 900; color: #ffffff;'>{pressure_pct:.1f}%</div>{retest_html}</div>"""
            
            # 7. Weighted Institutional Conviction Score (TCS)
            mtf_score = (max(bull_cnt, bear_cnt) / 4.0) * 25.0
            vwap_score = 20.0 if ((info['direction'] == 'DEMAND' and ltp > vwap) or (info['direction'] == 'SUPPLY' and ltp < vwap)) else 0.0
            rsi_score = 15.0 if ((info['direction'] == 'DEMAND' and 50 <= rsi <= 70) or (info['direction'] == 'SUPPLY' and 30 <= rsi <= 50)) else 5.0
            zone_confluence_score = min(20.0, len(matched_zones) * 10.0)
            pressure_component = min(20.0, pressure_pct * 0.2)
            
            tcs = int(min(100.0, max(0.0, mtf_score + vwap_score + rsi_score + zone_confluence_score + pressure_component)))
            
            # 8. Filter Selection
            if tcs >= HIGH_CONVICTION_TCS_THRESHOLD:
                def dot(b): return "<span style='color:#00e676;'>🟢</span>" if b else "<span style='color:#ff5252;'>🔴</span>"
                emas_html = f"{dot(ltp > ema1)} {dot(ltp > ema3)} {dot(ltp > ema5)} {dot(ltp > ema15)}"
                pnl_color = '#00e676' if pnl_pct >= 0 else '#ff5252'
                cobi_color = '#00e676' if imbalance_pct >= 0 else '#ff5252'

                high_conviction_rows.append({
                    "symbol": info['symbol'],
                    "zone_html": zone_html,
                    "open": f"₹{open_p:.2f}",
                    "ltp": f"₹{ltp:.2f}",
                    "pnl": f"{pnl_pct:+.2f}%",
                    "pnl_color": pnl_color,
                    "emas_html": emas_html,
                    "pressure_box": pressure_box_html,
                    "target": f"₹{target_price:.2f}",
                    "tcs": tcs,
                    "cobi_html": cobi_html,
                    "cobi_color": cobi_color
                })
        except Exception:
            continue
            
    high_conviction_rows.sort(key=lambda x: x['tcs'], reverse=True)
    elapsed_ms = int((time.time() - t0) * 1000)
    return high_conviction_rows, elapsed_ms

init_rows, init_latency = compute_all_metrics(universe)
init_timestamp = datetime.datetime.now().strftime('%H:%M:%S')

init_payload = {
    "rows": init_rows,
    "timestamp": init_timestamp,
    "latency_ms": init_latency,
    "total_scanned": len(tickers_list),
    "universe_data": {
        k: {
            "symbol": v["symbol"],
            "open": v["open"],
            "atr": v["atr"],
            "direction": v["direction"],
            "zones": v["zones"],
            "opposing_zones": v["opposing_zones"]
        } for k, v in universe.items()
    }
}

json_payload_str = json.dumps(init_payload)

# -----------------------------------------------------------------------------
# 5. SOLID-FRAME ZERO FLICKER / ZERO BLANK JAVASCRIPT APP
# -----------------------------------------------------------------------------
zero_blink_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    html, body {{
        background-color: #090c10 !important;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }}
    .dashboard-container {{
        background-color: #090c10;
        border: 1.5px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin: 6px;
        height: calc(100vh - 14px);
        display: flex;
        flex-direction: column;
    }}
    .header-bar {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px dashed #30363d;
        padding-bottom: 8px;
        margin-bottom: 8px;
        gap: 6px;
    }}
    .brand-title {{
        color: #00e676;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }}
    .stream-tag {{
        color: #8b949e;
        font-size: 11px;
        background: #161b22;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px dashed #30363d;
    }}
    .table-wrapper {{
        flex: 1;
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px dashed #30363d;
        border-radius: 6px;
        background: #090c10;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        text-align: center;
        white-space: nowrap;
    }}
    thead th {{
        position: sticky;
        top: 0;
        background: #161b22;
        color: #8b949e;
        text-transform: uppercase;
        font-size: 9.5px;
        padding: 8px 6px;
        border-bottom: 1px dashed #30363d;
        border-right: 1px dashed #30363d;
        z-index: 2;
    }}
    td {{
        padding: 8px 6px;
        border-bottom: 1px dashed #30363d;
        border-right: 1px dashed #21262d;
    }}
</style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header-bar">
            <div class="brand-title">⚡ FREE QUANT ENGINE | SMC LIVE PRESSURE DASHBOARD</div>
            <div class="stream-tag" id="status-tag">
                LIVE STREAM: {init_timestamp} IST | Active: {len(init_rows)}/{len(tickers_list)} | Latency: {init_latency}ms
            </div>
        </div>
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>
                        <th style="text-align: left;">Symbol</th>
                        <th style="text-align: left;">Zone Alignments</th>
                        <th>Open</th>
                        <th>LTP</th>
                        <th>Change</th>
                        <th>EMAS (1m|3m|5m|15m)</th>
                        <th style="min-width: 150px;">Supply/Demand Delta Box</th>
                        <th>Target (₹)</th>
                        <th>TCS Score</th>
                        <th>Buyer/Seller (COBI)</th>
                    </tr>
                </thead>
                <tbody id="table-body"></tbody>
            </table>
        </div>
    </div>

<script>
    const globalState = {json_payload_str};
    const tbody = document.getElementById('table-body');
    const statusTag = document.getElementById('status-tag');

    // Smooth DOM Update without removing nodes or causing screen redraws
    function updateDOMTable(rows, timestamp, scanned, latency) {{
        statusTag.innerText = `LIVE STREAM: ${{timestamp}} IST | Active: ${{rows.length}}/${{scanned}} | Latency: ${{latency}}ms`;
        
        if (!rows || rows.length === 0) {{
            tbody.innerHTML = `<tr><td colspan="10" style="padding: 24px; color: #8b949e; text-align: center; font-style: italic;">⏳ Scanning in background... No high-conviction pure-zone setups (₹300-₹600) right now.</td></tr>`;
            return;
        }}

        let out = "";
        for (let r of rows) {{
            out += `
                <tr style="border-bottom: 1px dashed #30363d;">
                    <td style="font-weight: 900; text-align: left; color: #ffffff; padding: 10px 8px; border-right: 1px dashed #21262d;">${{r.symbol}}</td>
                    <td style="text-align: left; font-size: 10px; border-right: 1px dashed #21262d;">${{r.zone_html}}</td>
                    <td style="border-right: 1px dashed #21262d;">${{r.open}}</td>
                    <td style="font-weight: 700; border-right: 1px dashed #21262d;">${{r.ltp}}</td>
                    <td style="color: ${{r.pnl_color}}; font-weight: 800; border-right: 1px dashed #21262d;">${{r.pnl}}</td>
                    <td style="border-right: 1px dashed #21262d;">${{r.emas_html}}</td>
                    <td style="padding: 6px; border-right: 1px dashed #21262d;">${{r.pressure_box}}</td>
                    <td style="color: #00e676; font-weight: 800; border-right: 1px dashed #21262d;">${{r.target}}</td>
                    <td style="border-right: 1px dashed #21262d;"><span style="color: #00e676; font-weight: 900; font-size: 12px;">${{r.tcs}}/100</span></td>
                    <td style="color: ${{r.cobi_color}}; font-weight: 700;">${{r.cobi_html}}</td>
                </tr>
            `;
        }}
        tbody.innerHTML = out;
    }}

    // Initial Static Paint
    updateDOMTable(globalState.rows, globalState.timestamp, globalState.total_scanned, globalState.latency_ms);

    // Fast RSI in JS
    function fastRsi(close, period=14) {{
        if (close.length < period + 2) return 50.0;
        let diffs = [];
        for (let i = 1; i < close.length; i++) diffs.push(close[i] - close[i-1]);
        let gains = diffs.map(d => d > 0 ? d : 0);
        let losses = diffs.map(d => d < 0 ? -d : 0);
        let avgGain = gains.slice(0, period).reduce((a,b)=>a+b, 0) / period;
        let avgLoss = losses.slice(0, period).reduce((a,b)=>a+b, 0) / period;
        for (let i = period; i < diffs.length; i++) {{
            avgGain = (avgGain * (period - 1) + gains[i]) / period;
            avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
        }}
        if (avgLoss === 0) return 100.0;
        let rs = avgGain / avgLoss;
        return 100.0 - (100.0 / (1.0 + rs));
    }}

    // Fast EMA in JS
    function fastEma(arr, span) {{
        if (!arr || arr.length === 0) return 0.0;
        if (arr.length < span) return arr[arr.length - 1];
        let alpha = 2.0 / (span + 1.0);
        let ema = arr[0];
        for (let i = 1; i < arr.length; i++) {{
            ema = alpha * arr[i] + (1.0 - alpha) * ema;
        }}
        return ema;
    }}

    // Live In-Place Data Refresher (Zero Page Reload)
    async function liveCycle() {{
        const tickers = Object.keys(globalState.universe_data);
        if (tickers.length === 0) return;
        const t0 = performance.now();
        
        try {{
            // Yahoo Finance Query v8 direct endpoint (Fastest JSON response)
            const promises = tickers.map(t => 
                fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${{t}}?interval=1m&range=1d`)
                    .then(res => res.json())
                    .then(data => ({{ ticker: t, data: data }}))
                    .catch(() => null)
            );

            const results = await Promise.all(promises);
            let high_conviction_rows = [];

            for (let item of results) {{
                if (!item || !item.data || !item.data.chart || !item.data.chart.result) continue;
                const meta = item.data.chart.result[0];
                const info = globalState.universe_data[item.ticker];
                const quote = meta.indicators.quote[0];
                
                const closeArr = quote.close.filter(x => x !== null && x !== undefined);
                const highArr = quote.high.filter(x => x !== null && x !== undefined);
                const lowArr = quote.low.filter(x => x !== null && x !== undefined);
                const openArr = quote.open.filter(x => x !== null && x !== undefined);
                const volArr = quote.volume.filter(x => x !== null && x !== undefined);

                if (closeArr.length < 5) continue;

                const ltp = closeArr[closeArr.length - 1];
                const open_p = info.open;
                const atr_val = info.atr;
                const pnl_pct = ((ltp - open_p) / open_p) * 100.0;

                // VWAP & RSI
                let sumTpVol = 0, sumVol = 0;
                for (let i = 0; i < closeArr.length; i++) {{
                    let tp = (highArr[i] + lowArr[i] + closeArr[i]) / 3.0;
                    let v = volArr[i] || 0;
                    sumTpVol += tp * v;
                    sumVol += v;
                }}
                const vwap = sumVol > 0 ? (sumTpVol / sumVol) : ltp;
                const rsi = fastRsi(closeArr, 14);

                // EMAs
                const ema1 = fastEma(closeArr.slice(-15), 13);
                const s3 = []; for (let i = closeArr.length - 1; i >= 0 && s3.length < 15; i -= 3) s3.unshift(closeArr[i]);
                const s5 = []; for (let i = closeArr.length - 1; i >= 0 && s5.length < 15; i -= 5) s5.unshift(closeArr[i]);
                const s15 = []; for (let i = closeArr.length - 1; i >= 0 && s15.length < 15; i -= 15) s15.unshift(closeArr[i]);

                const ema3 = s3.length >= 13 ? fastEma(s3, 13) : ema1;
                const ema5 = s5.length >= 13 ? fastEma(s5, 13) : ema1;
                const ema15 = s15.length >= 13 ? fastEma(s15, 13) : ema1;

                const e1 = ltp > ema1, e3 = ltp > ema3, e5 = ltp > ema5, e15 = ltp > ema15;
                const bull_cnt = [e1, e3, e5, e15].filter(Boolean).length;
                const bear_cnt = 4 - bull_cnt;

                // Zones
                let matched_zones = [];
                for (let z of info.zones) {{
                    if (z.type === 'SUPPLY') {{
                        matched_zones.push(`<span style='color:#ff5252; font-weight:700;'>SUPPLY (${{z.tf}})</span>`);
                    }} else {{
                        matched_zones.push(`<span style='color:#00e676; font-weight:700;'>DEMAND (${{z.tf}})</span>`);
                    }}
                }}
                const zone_html = matched_zones.join(" | ");

                // COBI
                let buyPower = 0, sellPower = 0;
                for (let i = 0; i < closeArr.length; i++) {{
                    let br = (highArr[i] - lowArr[i]) + 1e-6;
                    let v = volArr[i] || 0;
                    buyPower += v * ((closeArr[i] - lowArr[i]) / br);
                    sellPower += v * ((highArr[i] - closeArr[i]) / br);
                }}
                let totPower = buyPower + sellPower + 1e-6;
                let buy_pct = (buyPower / totPower) * 100.0;
                let imbalance_pct = ((buyPower - sellPower) / totPower) * 100.0;
                let cobi_html = `${{buy_pct.toFixed(0)}}% Buy (${{imbalance_pct >= 0 ? '+' : ''}}${{imbalance_pct.toFixed(1)}}%)`;

                // Directional Institutional Pressure Index
                let directional_move = info.direction === 'DEMAND' ? (ltp - open_p) : (open_p - ltp);
                let pressure_pct = Math.min(100.0, Math.max(0.0, (directional_move / atr_val) * 100.0));

                // Advanced Target Engine
                let border_c, bg_c, status_t, target_price;
                if (info.direction === 'SUPPLY') {{
                    border_c = "#ff3838"; bg_c = "rgba(255, 56, 56, 0.12)"; status_t = "SUPPLY ACCUMULATION";
                    let opp = info.opposing_zones.map(z => z.top).filter(x => x < ltp);
                    target_price = opp.length > 0 ? Math.max(...opp) : (ltp - (atr_val * 1.618));
                }} else {{
                    border_c = "#00e676"; bg_c = "rgba(0, 230, 118, 0.12)"; status_t = "DEMAND ABSORPTION";
                    let opp = info.opposing_zones.map(z => z.bot).filter(x => x > ltp);
                    target_price = opp.length > 0 ? Math.min(...opp) : (ltp + (atr_val * 1.618));
                }}

                let retest_html = pressure_pct > 75.0 ? "<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST REJECTION</div>" : "";
                let pressure_box_html = `<div style='border: 1px dashed ${{border_c}}; background-color: ${{bg_c}}; padding: 3px 5px; border-radius: 4px; text-align: center;'><div style='font-size: 9px; font-weight: 800; color: ${{border_c}};'>${{status_t}}</div><div style='font-size: 11px; font-weight: 900; color: #ffffff;'>${{pressure_pct.toFixed(1)}}%</div>${{retest_html}}</div>`;

                // TCS
                let mtf_score = (Math.max(bull_cnt, bear_cnt) / 4.0) * 25.0;
                let vwap_score = ((info.direction === 'DEMAND' && ltp > vwap) || (info.direction === 'SUPPLY' && ltp < vwap)) ? 20.0 : 0.0;
                let rsi_score = ((info.direction === 'DEMAND' && rsi >= 50 && rsi <= 70) || (info.direction === 'SUPPLY' && rsi >= 30 && rsi <= 50)) ? 15.0 : 5.0;
                let zone_confluence_score = Math.min(20.0, matched_zones.length * 10.0);
                let pressure_component = Math.min(20.0, pressure_pct * 0.2);

                let tcs = Math.floor(Math.min(100.0, Math.max(0.0, mtf_score + vwap_score + rsi_score + zone_confluence_score + pressure_component)));

                if (tcs >= 65) {{
                    const dot = (b) => b ? "<span style='color:#00e676;'>🟢</span>" : "<span style='color:#ff5252;'>🔴</span>";
                    high_conviction_rows.push({{
                        symbol: info.symbol,
                        zone_html: zone_html,
                        open: `₹${{open_p.toFixed(2)}}`,
                        ltp: `₹${{ltp.toFixed(2)}}`,
                        pnl: `${{pnl_pct >= 0 ? '+' : ''}}${{pnl_pct.toFixed(2)}}%`,
                        pnl_color: pnl_pct >= 0 ? '#00e676' : '#ff5252',
                        emas_html: `${{dot(e1)}} ${{dot(e3)}} ${{dot(e5)}} ${{dot(e15)}}`,
                        pressure_box: pressure_box_html,
                        target: `₹${{target_price.toFixed(2)}}`,
                        tcs: tcs,
                        cobi_html: cobi_html,
                        cobi_color: imbalance_pct >= 0 ? '#00e676' : '#ff5252'
                    }});
                }}
            }}

            high_conviction_rows.sort((a,b) => b.tcs - a.tcs);
            const elapsed = Math.floor(performance.now() - t0);
            const nowTime = new Date().toLocaleTimeString('en-GB');

            if (high_conviction_rows.length > 0) {{
                updateDOMTable(high_conviction_rows, nowTime, tickers.length, elapsed);
            }}
        }} catch(err) {{
            console.error("Fetch background silent bypass:", err);
        }}
    }}

    // In-Place Dynamic DOM Refresh Loop without Streamlit Screen Reload
    setInterval(liveCycle, 2000);
</script>
</body>
</html>
"""

components.html(zero_blink_html, height=800, scrolling=False)
