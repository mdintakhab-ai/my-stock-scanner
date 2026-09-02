import datetime
import warnings
import concurrent.futures
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string, jsonify

warnings.filterwarnings('ignore')

app = Flask(__name__)

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
            '15m': extract_advanced_zones(df_15m),
            '1H': extract_advanced_zones(df_1h),
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
        
        if df_live.empty: return None
        
        current_price = df_live['Close'].iloc[-1]
        open_p = stock_info["Open Price"]
        pnl_pct = ((current_price - open_p) / open_p) * 100
        
        ema_1m = get_enhanced_ema13_signal(df_live)
        ema_3m = get_enhanced_ema13_signal(df_3m)
        ema_5m = get_enhanced_ema13_signal(df_5m)
        ema_15m = get_enhanced_ema13_signal(df_15m)

        def dot_badge(status):
            if status == "BULLISH": return "<span title='Bullish' style='color:#00e676; font-size:16px;'>🟢</span>"
            elif status == "BEARISH": return "<span title='Bearish' style='color:#ff5252; font-size:16px;'>🔴</span>"
            return "<span title='Neutral' style='color:#484f58; font-size:14px;'>●</span>"

        buyer_pct, imbalance_delta, cobi = calculate_cobi_and_imbalance(df_live)
        atr_val = calculate_atr(stock_info["df_1d"], 14).iloc[-1]
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
            retest_alert = f"<div style='color:#ffaa00; font-size:9px; font-weight:bold; margin-top:2px;'>⚠️ ZONE RE-TEST REJECTION</div>"

        pressure_box_html = f"""
        <div style='border: 1.5px solid {border_color}; background-color: {bg_color}; padding: 4px 6px; border-radius: 5px; text-align: center;'>
            <div style='font-size: 10px; font-weight: 800; color: {border_color};'>{status_txt}</div>
            <div style='font-size: 12px; font-weight: 900; color: #ffffff;'>{supply_pressure_pct:.1f}%</div>
            {retest_alert}
        </div>
        """

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

# Global cache to keep scanned locked universe
LOCKED_UNIVERSE = []

def initialize_scanner():
    global LOCKED_UNIVERSE
    universe = fetch_dynamic_universe()
    print("⏳ Scanning Universe & Initializing Free SMC Engine...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(scan_initial_universe, universe))
        LOCKED_UNIVERSE = [r for r in results if r is not None]
    LOCKED_UNIVERSE = sorted(LOCKED_UNIVERSE, key=lambda x: x['Score'], reverse=True)
    print(f"✅ Filtered {len(LOCKED_UNIVERSE)} Stocks Locked Successfully!")

# Initialize on startup
initialize_scanner()

# -----------------------------------------------------------------------------
# 5. MOBILE RESPONSIVE WEB ROUTES
# -----------------------------------------------------------------------------
MOBILE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ FREE QUANT ENGINE | SMC MOBILE</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #090c10;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .quant-container {
            background-color: #090c10;
            border: 1px solid #1f293d;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
        }
        .quant-header {
            display: flex;
            flex-direction: column;
            gap: 5px;
            border-bottom: 1px solid #1f293d;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }
        @media(min-width: 768px) {
            .quant-header { flex-direction: row; justify-content: space-between; align-items: center; }
        }
        .quant-title {
            color: #00e676;
            font-size: 14px;
            font-weight: 800;
        }
        .quant-clock {
            color: #8b949e;
            font-size: 10px;
            background: #161b22;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
        }
        /* Mobile Card Layout for extreme responsiveness */
        .stock-card {
            background-color: #161b22;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .card-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            margin-bottom: 6px;
        }
        .symbol-text {
            color: #ffffff;
            font-weight: 800;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container-fluid px-2">
        <div class="quant-container">
            <div class="quant-header">
                <div class="quant-title">⚡ FREE QUANT ENGINE | SMC MOBILE</div>
                <div id="quant-clock-val" class="quant-clock">INITIALIZING STREAM...</div>
            </div>
            
            <div id="mobile-cards-container">
                <div class="text-center text-muted py-4">Fetching live market data... Please wait.</div>
            </div>
        </div>
    </div>

    <script>
        function fetchLiveData() {
            fetch('/api/live-data')
                .then(response => response.json())
                .then(payload => {
                    document.getElementById('quant-clock-val').innerText = 'LIVE STREAM: ' + payload.timestamp + ' IST';
                    const container = document.getElementById('mobile-cards-container');
                    let html = '';
                    
                    payload.data.forEach(row => {
                        html += `
                        <div class="stock-card">
                            <div class="card-row">
                                <span class="symbol-text">${row.symbol}</span>
                                <span>${row.ltp} (${row.change})</span>
                            </div>
                            <div class="card-row" style="font-size:11px; color:#8b949e;">
                                <span>Zones: ${row.zones}</span>
                                <span>Open: ${row.open}</span>
                            </div>
                            <hr style="border-color: #21262d; margin: 6px 0;">
                            <div class="card-row">
                                <div><strong>EMAs (1|3|5|15m):</strong><br>${row.ema1m} ${row.ema3m} ${row.ema5m} ${row.ema15m}</div>
                                <div><strong>Target:</strong> ${row.target}</div>
                            </div>
                            <div class="card-row mt-2">
                                <div style="flex-grow: 1; margin-right: 5px;">${row.pressure_box}</div>
                                <div style="text-align: right;">
                                    <div style="font-size:10px; color:#8b949e;">TCS Score</div>
                                    <div>${row.tcs}</div>
                                </div>
                            </div>
                            <div class="card-row mt-1" style="font-size: 11px;">
                                <span><strong>COBI:</strong> ${row.cobi}</span>
                            </div>
                        </div>
                        `;
                    });
                    container.innerHTML = html;
                })
                .catch(err => console.error("Error fetching live data:", err));
        }

        // Fetch immediately and poll every 3 seconds
        fetchLiveData();
        setInterval(fetchLiveData, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(MOBILE_HTML_TEMPLATE)

@app.route('/api/live-data')
def live_data_api():
    global LOCKED_UNIVERSE
    if not LOCKED_UNIVERSE:
        initialize_scanner()
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        live_data = list(executor.map(fetch_live_updates, LOCKED_UNIVERSE))
        live_data = [d for d in live_data if d is not None]

    live_data = sorted(live_data, key=lambda x: x['_score'], reverse=True)
    for row in live_data: del row['_score']

    payload = {
        "timestamp": datetime.datetime.now().strftime('%H:%M:%S'),
        "data": live_data
    }
    return jsonify(payload)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
