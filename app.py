import datetime
import io
import time
import warnings
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

warnings.filterwarnings("ignore")

# Streamlit Page Config
st.set_page_config(
    page_title="Lightspeed Quant Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global Cache to prevent session overload and thread crash
if "SYMBOL_CACHE" not in st.session_state:
    st.session_state.SYMBOL_CACHE = []
if "LAST_SYMBOL_FETCH" not in st.session_state:
    st.session_state.LAST_SYMBOL_FETCH = 0


def fetch_nifty500_symbols():
    if st.session_state.SYMBOL_CACHE and (
        time.time() - st.session_state.LAST_SYMBOL_FETCH < 3600
    ):
        return st.session_state.SYMBOL_CACHE

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            df_nse = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
            symbols = [
                f"{sym.strip()}.NS"
                for sym in df_nse["Symbol"].dropna().unique()
            ]
            if len(symbols) > 50:
                st.session_state.SYMBOL_CACHE = symbols
                st.session_state.LAST_SYMBOL_FETCH = time.time()
                return symbols
    except Exception:
        pass

    url_50 = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    try:
        response = requests.get(url_50, headers=headers, timeout=5)
        df_nse = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        symbols = [f"{sym.strip()}.NS" for sym in df_nse["Symbol"].dropna().unique()]
        st.session_state.SYMBOL_CACHE = symbols
        st.session_state.LAST_SYMBOL_FETCH = time.time()
        return symbols
    except Exception:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


def get_scanner_data():
    symbols = fetch_nifty500_symbols()

    all_tickers = list(set(symbols + ["^NSEI"]))

    try:
        daily_data = yf.download(
            all_tickers, period="1mo", interval="1d", progress=False, threads=False
        )
    except Exception:
        return pd.DataFrame(), 0.0

    if daily_data.empty:
        return pd.DataFrame(), 0.0

    try:
        nifty_close = daily_data["Close", "^NSEI"].dropna()
        nifty_change = (
            (nifty_close.iloc[-1] - nifty_close.iloc[-2]) / nifty_close.iloc[-2]
        ) * 100
    except Exception:
        nifty_change = 0.0

    try:
        intra_5m = yf.download(
            symbols, period="2d", interval="5m", progress=False, threads=False
        )
    except Exception:
        intra_5m = pd.DataFrame()

    results = []

    for ticker in symbols:
        sym = ticker.replace(".NS", "")
        try:
            close_s = daily_data["Close", ticker].dropna()
            high_s = daily_data["High", ticker].dropna()
            low_s = daily_data["Low", ticker].dropna()
            open_s = daily_data["Open", ticker].dropna()
            vol_s = daily_data["Volume", ticker].dropna()

            if close_s.empty or len(close_s) < 15:
                continue

            last_close = close_s.iloc[-1]

            # Price Filter (₹300 - ₹600)
            if not (300 <= last_close <= 600):
                continue

            high_p, low_p, vol_p, open_p = (
                high_s.iloc[-1],
                low_s.iloc[-1],
                vol_s.iloc[-1],
                open_s.iloc[-1],
            )
            avg_vol = vol_s.iloc[-10:].mean()

            delta_pct = round(
                ((last_close - close_s.iloc[-2]) / close_s.iloc[-2]) * 100, 2
            )
            rvol = round(vol_p / avg_vol, 2) if avg_vol > 0 else 0.0

            # Dynamic VWAP
            typical_price = (high_s + low_s + close_s) / 3
            vwap = (typical_price * vol_s).tail(10).sum() / vol_s.tail(10).sum()
            vwap_dist = round(((last_close - vwap) / vwap) * 100, 2)

            # COBI Calculation
            rng = high_p - low_p
            cobi = round((last_close - open_p) / rng, 4) if rng > 0 else 0.0

            # Relative Strength vs Nifty
            rel_strength = round(delta_pct - nifty_change, 2)

            # Demand / Supply Calculation (+50% D to -50% S)
            flow_pressure = (
                ((last_close - low_p) / rng) * 100 if rng > 0 else 50.0
            )
            ds_val = round(flow_pressure - 50.0, 1)

            if ds_val > 0:
                ds_badge = f"<span style='color: #00e676; font-weight: bold;'>🟢 +{ds_val}% D</span>"
            elif ds_val < 0:
                ds_badge = f"<span style='color: #ff5252; font-weight: bold;'>🔴 {ds_val}% S</span>"
            else:
                ds_badge = f"<span style='color: #ffb300; font-weight: bold;'>🟡 0.0%</span>"

            # Multi-Timeframe Check
            b5 = False
            if not intra_5m.empty:
                try:
                    df_5m_stock = intra_5m.xs(ticker, axis=1, level=1).dropna()
                    if len(df_5m_stock) >= 13:
                        ema13 = df_5m_stock["Close"].ewm(span=13, adjust=False).mean()
                        b5 = df_5m_stock["Close"].iloc[-1] > ema13.iloc[-1]
                except Exception:
                    b5 = False

            dot = "🟢" if b5 else "🔴"
            mtf_dots = f"{dot} {dot} {dot} {dot}"

            # Priority & Scoring Rules
            score = 0
            if rvol >= 1.3:
                score += 10
            if last_close > vwap:
                score += 10
            if rel_strength > 0.5:
                score += 10
            if ds_val > 10:
                score += 10

            is_valid_entry = b5 and (vwap_dist <= 2.5) and (score >= 30)

            if is_valid_entry:
                action_badge = "<span style='background-color: #004d40; color: #00e676; padding: 4px 10px; border-radius: 4px; font-weight: bold;'>STRONG BUY</span>"
                priority = 1
            elif b5 and score >= 20:
                action_badge = "<span style='background-color: #37474f; color: #ffb300; padding: 4px 10px; border-radius: 4px;'>WATCH</span>"
                priority = 0
            else:
                action_badge = "<span style='background-color: #21262d; color: #8b949e; padding: 4px 10px; border-radius: 4px;'>SKIP</span>"
                priority = -1

            smc_signal = (
                "<span style='color: #00e676; font-weight: bold;'>MSS CONFIRMED</span>"
                if (last_close > high_s.iloc[-2] and rvol > 1.2)
                else "<span style='color: #8b949e;'>CONSOLIDATION</span>"
            )

            results.append(
                {
                    "SYMBOL": sym,
                    "LTP": f"₹{last_close:.2f}",
                    "VWAP": f"₹{vwap:.2f}",
                    "DELTA": delta_pct,
                    "RVOL": f"{rvol}x",
                    "COBI": cobi,
                    "RS vs NIFTY": f"<span style='color:#00e676;'>+{rel_strength}%</span>" if rel_strength > 0 else f"{rel_strength}%",
                    "DEMAND_SUPPLY": ds_badge,
                    "SMC STRUCTURE": smc_signal,
                    "MTF": mtf_dots,
                    "ACTION": action_badge,
                    "priority": priority,
                    "score": score,
                    "rvol_raw": rvol,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(
            by=["priority", "score", "rvol_raw"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        df["#"] = df.index + 1

    return df, nifty_change


placeholder = st.empty()

while True:
    df, _ = get_scanner_data()
    current_time = datetime.datetime.now().strftime("%H:%M:%S") + " IST"

    if df.empty:
        table_rows = "<tr><td colspan='12' style='text-align:center; padding: 25px; color: #8b949e;'>Fetching Realtime Market Data...</td></tr>"
        stock_count = 0
    else:
        stock_count = len(df)
        rows_list = []
        for _, row in df.iterrows():
            delta_color = "#00e676" if row["DELTA"] >= 0 else "#ff5252"
            delta_str = (
                f"+{row['DELTA']}%" if row["DELTA"] >= 0 else f"{row['DELTA']}%"
            )
            row_class = "priority-row" if row["priority"] == 1 else ""

            rows_list.append(
                f"""
                <tr class="{row_class}">
                    <td class="col-sticky-1" style="color: #8b949e;">{row['#']}</td>
                    <td class="col-sticky-2">{row['SYMBOL']}</td>
                    <td class="col-sticky-3">{row['LTP']}</td>
                    <td class="col-sticky-4">{row['VWAP']}</td>
                    <td style="color: {delta_color}; font-weight: bold;">{delta_str}</td>
                    <td style="color: #58a6ff; font-weight: bold;">{row['RVOL']}</td>
                    <td style="color: #c9d1d9;">{row['COBI']}</td>
                    <td style="text-align: center;">{row['RS vs NIFTY']}</td>
                    <td style="text-align: center;">{row['DEMAND_SUPPLY']}</td>
                    <td style="text-align: center;">{row['SMC STRUCTURE']}</td>
                    <td style="text-align: center;">{row['MTF']}</td>
                    <td style="text-align: center;">{row['ACTION']}</td>
                </tr>
                """
            )
        table_rows = "".join(rows_list)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            padding: 4px;
            background-color: #0E1117;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #c9d1d9;
        }}
        .scanner-wrapper {{
            background-color: #0d1117;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #30363d;
            box-sizing: border-box;
            width: 100%;
        }}
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #21262d;
            padding-bottom: 8px;
            margin-bottom: 8px;
        }}
        .table-responsive {{
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: 6px;
        }}
        .scanner-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            text-align: right;
            font-size: 12px;
            white-space: nowrap;
            background-color: #0d1117;
        }}
        .scanner-table th, .scanner-table td {{
            padding: 7px 8px;
            border-bottom: 1px solid #21262d;
            background-color: #0d1117;
        }}
        .scanner-table th {{
            color: #8b949e;
            border-bottom: 2px solid #30363d;
            height: 30px;
            font-size: 11px;
            text-transform: uppercase;
        }}
        
        /* Sticky Freeze Columns for Mobile Horizontal Scroll */
        .col-sticky-1 {{
            position: sticky;
            left: 0px;
            z-index: 2;
            width: 28px;
            min-width: 28px;
            text-align: center !important;
            border-right: 1px solid #21262d;
        }}
        .col-sticky-2 {{
            position: sticky;
            left: 28px;
            z-index: 2;
            min-width: 95px;
            text-align: left !important;
            font-weight: bold;
            color: #ffffff !important;
        }}
        .col-sticky-3 {{
            position: sticky;
            left: 123px;
            z-index: 2;
            min-width: 75px;
            font-weight: bold;
            color: #ffffff !important;
        }}
        .col-sticky-4 {{
            position: sticky;
            left: 198px;
            z-index: 2;
            min-width: 75px;
            border-right: 2px solid #388bfd;
            color: #8b949e !important;
        }}

        .priority-row td {{
            background-color: #161b22 !important;
        }}

        ::-webkit-scrollbar {{
            height: 6px;
            width: 6px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #30363d;
            border-radius: 3px;
        }}
    </style>
    </head>
    <body>
        <div class="scanner-wrapper">
            <div class="header-bar">
                <div style="font-weight: bold; font-size: 13px; color: #00e676;">
                    ● LIGHTSPEED QUANT | <span style="color: #ffffff;">{current_time}</span>
                </div>
                <div style="color: #8b949e; font-size: 11px;">
                    Stocks (₹300-₹600): <span style="color: #ffffff; font-weight: bold;">{stock_count}</span>
                </div>
            </div>
            <div class="table-responsive">
                <table class="scanner-table">
                    <thead>
                        <tr>
                            <th class="col-sticky-1">#</th>
                            <th class="col-sticky-2">SYMBOL</th>
                            <th class="col-sticky-3">LTP</th>
                            <th class="col-sticky-4">VWAP</th>
                            <th style="padding: 4px 8px;">DELTA</th>
                            <th style="padding: 4px 8px;">RVOL</th>
                            <th style="padding: 4px 8px;">COBI</th>
                            <th style="text-align: center; padding: 4px 8px;">RS vs NIFTY</th>
                            <th style="text-align: center; padding: 4px 8px;">DEMAND / SUPPLY</th>
                            <th style="text-align: center; padding: 4px 8px;">SMC STRUCTURE</th>
                            <th style="text-align: center; padding: 4px 8px;">MTF</th>
                            <th style="text-align: center; padding: 4px 8px;">ACTION</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    with placeholder.container():
        components.html(full_html, height=800, scrolling=True)

    time.sleep(10)
