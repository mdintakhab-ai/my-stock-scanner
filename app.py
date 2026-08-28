# =============================================================================
# 1. DEPENDENCY INSTALLATION
# =============================================================================
!pip install yfinance pandas numpy requests -q

# =============================================================================
# 2. ADVANCED SMC + COBI MICROSTRUCTURE LIVE QUANT ENGINE (MOBILE NO-JUMP)
# =============================================================================
import math
import time
import sys
import io
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from IPython.display import clear_output

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Configuration Parameters
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

def create_robust_nse_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com",
        "Connection": "keep-alive"
    })
    try:
        session.get("https://www.nseindia.com", timeout=3)
    except Exception:
        pass
    return session

# =============================================================================
# LEVEL-2 ORDER BOOK QUANT ENGINE (OBIR, WOB, MPDR, COBI)
# =============================================================================
def fetch_live_orderbook_cobi(session, symbol, p_change, today_vol):
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

            raw_bids = market_dept.get("bid", [])
            raw_asks = market_dept.get("ask", [])

            bids = [(clean_num(b.get("price")), clean_num(b.get("quantity"))) for b in raw_bids if clean_num(b.get("quantity")) > 0][:5]
            asks = [(clean_num(a.get("price")), clean_num(a.get("quantity"))) for a in raw_asks if clean_num(a.get("quantity")) > 0][:5]

            if len(bids) > 0 and len(asks) > 0 and (buy_q > 0 or sell_q > 0):
                # 1. OBIR Calculation
                obir = (buy_q - sell_q) / (buy_q + sell_q) if (buy_q + sell_q) > 0 else 0.0

                # 2. WOB Calculation
                depth_len = min(len(bids), len(asks))
                depth_weights = 1.0 / np.arange(1, depth_len + 1)
                w_bid_sum = np.sum([b[1] for b in bids[:depth_len]] * depth_weights)
                w_ask_sum = np.sum([a[1] for a in asks[:depth_len]] * depth_weights)
                wob = (w_bid_sum - w_ask_sum) / (w_bid_sum + w_ask_sum) if (w_bid_sum + w_ask_sum) > 0 else 0.0

                # 3. MPDR Calculation
                p_bid1, q_bid1 = bids[0][0], bids[0][1]
                p_ask1, q_ask1 = asks[0][0], asks[0][1]
                spread = max(p_ask1 - p_bid1, 0.05)
                p_mid = (p_bid1 + p_ask1) / 2.0
                p_micro = (p_bid1 * q_ask1 + p_ask1 * q_bid1) / (q_bid1 + q_ask1) if (q_bid1 + q_ask1) > 0 else p_mid
                mpdr = max(min((p_micro - p_mid) / (spread / 2.0), 1.0), -1.0)

                # 4. Composite COBI Index
                cobi_score = (0.30 * obir) + (0.40 * wob) + (0.30 * mpdr)
                bs_ratio = round(buy_q / (sell_q if sell_q > 0 else 1), 2)
                imbalance = int(buy_q - sell_q)

                return bs_ratio, imbalance, cobi_score, True
    except Exception:
        pass

    # Mathematical synthetic proxy during off-market
    synth_cobi = max(min(p_change / 3.0, 1.0), -1.0)
    bs_ratio = round(max(0.2, min(5.0, 1.0 + (p_change * 0.35))), 2)
    imbalance = int(today_vol * (p_change / 100))
    return bs_ratio, imbalance, synth_cobi, False

# =============================================================================
# EXACT PINE SCRIPT SMC ENGINE (PERSISTENCE & ZONE MITIGATION)
# =============================================================================
def evaluate_pine_indicator(df_tf):
    if len(df_tf) < 30:
        return "YELLOW", "NONE", False, False

    df = df_tf.copy()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df['EMA13_Up'] = df['EMA13'] > df['EMA13'].shift(1)
    df['EMA13_Dn'] = df['EMA13'] < df['EMA13'].shift(1)
    df['RSI14'] = calculate_rsi(df['Close'], 14)
    df['ATR14'] = calculate_atr(df, 14)
    df['Vol_SMA14'] = df['Volume'].rolling(14).mean()

    n = len(df)
    highs, lows, closes, opens = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    atr, rsi, ema13 = df['ATR14'].values, df['RSI14'].values, df['EMA13'].values
    ema_up, ema_dn = df['EMA13_Up'].values, df['EMA13_Dn'].values

    # 1. State Machine Engine (Section 6)
    p_smc, fvgMinAtr, stateLife = 5, 0.1, 30
    bullState, bullBars = 0, 0
    bearState, bearBars = 0, 0
    trendSMC = 0
    lastHigh, lastLow, prevHigh_smc, prevLow_smc = np.nan, np.nan, np.nan, np.nan

    # 2. Demand & Supply Matrix Configuration (Section 9)
    p_mat = 2
    mergeThresh = 0.3
    maxTests = 4

    s_Boxes = []
    d_Boxes = []

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

        # State Confirmation Trigger
        if (bullState == 4 or (closes[i] > ema13[i] and ema_up[i])) and rsi[i] > 50:
            trendSMC = 1
        elif (bearState == 4 or (closes[i] < ema13[i] and ema_dn[i])) and rsi[i] < 50:
            trendSMC = -1

        if trendSMC == 1 and (closes[i] < ema13[i] and rsi[i] < 45): trendSMC = 0
        if trendSMC == -1 and (closes[i] > ema13[i] and rsi[i] > 55): trendSMC = 0

        # Box Detection Engine
        if i >= 2 * p_mat:
            pm = i - p_mat
            is_pm_hi = all(highs[pm] > highs[pm-k] for k in range(1, p_mat + 1)) and all(highs[pm] > highs[pm+k] for k in range(1, p_mat + 1)):
                if is_pm_hi:
                    topLvl = highs[pm]
                    botLvl = max(opens[pm], closes[pm])
                    isDup = any(b['active'] and abs(b['top'] - topLvl) < (atr[i] * mergeThresh) for b in s_Boxes)
                    if not isDup:
                        s_Boxes.append({'top': topLvl, 'bot': botLvl, 'active': True, 'tests': 0, 'created_bar': i})

            is_pm_lo = all(lows[pm] < lows[pm-k] for k in range(1, p_mat + 1)) and all(lows[pm] < lows[pm+k] for k in range(1, p_mat + 1)):
                if is_pm_lo:
                    topLvl = min(opens[pm], closes[pm])
                    botLvl = lows[pm]
                    isDup = any(b['active'] and abs(b['bot'] - botLvl) < (atr[i] * mergeThresh) for b in d_Boxes)
                    if not isDup:
                        d_Boxes.append({'top': topLvl, 'bot': botLvl, 'active': True, 'tests': 0, 'created_bar': i})

        # Mitigation & Zone Break Engine
        for b in s_Boxes:
            if b['active']:
                if highs[i] > b['bot'] and (i > 0 and highs[i-1] <= b['bot']):
                    b['tests'] += 1
                if highs[i] > b['top'] or b['tests'] >= maxTests:
                    b['active'] = False

        for b in d_Boxes:
            if b['active']:
                if lows[i] < b['top'] and (i > 0 and lows[i-1] >= b['top']):
                    b['tests'] += 1
                if lows[i] < b['bot'] or b['tests'] >= maxTests:
                    b['active'] = False

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

# Helper for Safe MultiIndex Data Extraction
def extract_df(data_dict, ticker):
    try:
        if isinstance(data_dict.columns, pd.MultiIndex):
            if ticker in data_dict.columns.levels[0]:
                return data_dict[ticker].dropna()
        else:
            return data_dict.dropna()
    except Exception:
        pass
    return pd.DataFrame()

# =============================================================================
# LIVE MONITORING SCANNER LOOP (FLICKER-FREE MOBILE ENGINE)
# =============================================================================
session = create_robust_nse_session()
yf_symbols = [f"{s}.NS" for s in WATCHLIST]
cycle = 1

while True:
    try:
        # Download market data cleanly
        data_daily = yf.download(yf_symbols, period="15d", interval="1d", group_by="ticker", progress=False, timeout=10)
        data_3m = yf.download(yf_symbols, period="5d", interval="2m", group_by="ticker", progress=False, timeout=10)
        data_5m = yf.download(yf_symbols, period="5d", interval="5m", group_by="ticker", progress=False, timeout=10)

        full_cards = []
        filtered_cards = []

        for sym in WATCHLIST:
            t_str = f"{sym}.NS"
            try:
                df_d = extract_df(data_daily, t_str)
                df_3 = extract_df(data_3m, t_str)
                df_5 = extract_df(data_5m, t_str)

                if len(df_d) < 2 or len(df_5) < 30: 
                    continue

                ltp = float(df_5.iloc[-1]["Close"])
                if not (PRICE_MIN <= ltp <= PRICE_MAX): 
                    continue

                prev_close = float(df_d.iloc[-2]["Close"])
                p_change = ((ltp - prev_close) / prev_close) * 100

                today_vol = float(df_d.iloc[-1]["Volume"])
                avg_vol_1w = df_d["Volume"].iloc[:-1].mean() if len(df_d) > 1 else today_vol
                vol_chg_pct = ((today_vol - avg_vol_1w) / avg_vol_1w * 100) if avg_vol_1w > 0 else 0.0

                high, low, close = float(df_d.iloc[-1]["High"]), float(df_d.iloc[-1]["Low"]), float(df_d.iloc[-1]["Close"])
                approx_vwap = (high + low + close) / 3.0
                vwap_dist_pct = ((ltp - approx_vwap) / approx_vwap) * 100
                vwap_status = "ABOVE (+)" if vwap_dist_pct > VWAP_BUFFER_PCT else ("BELOW (-)" if vwap_dist_pct < -VWAP_BUFFER_PCT else "AT VWAP")

                # Level-2 COBI Calculation
                bs_ratio, imbalance, cobi_val, has_depth = fetch_live_orderbook_cobi(session, sym, p_change, today_vol)
                imb_vs_avg_vol_pct = ((imbalance / avg_vol_1w) * 100) if avg_vol_1w > 0 else 0.0

                # Strict Pine Execution
                ema3_col, box3_t, is_bull_3m, is_bear_3m = evaluate_pine_indicator(df_3 if len(df_3) >= 30 else df_5)
                ema5_col, box5_t, is_bull_5m, is_bear_5m = evaluate_pine_indicator(df_5)

                tr = pd.concat([df_d["High"] - df_d["Low"], (df_d["High"] - df_d["Close"].shift(1)).abs(), (df_d["Low"] - df_d["Close"].shift(1)).abs()], axis=1).max(axis=1)
                spread = float(tr.iloc[-5:].mean()) * 1.2 if len(tr) >= 5 else 5.0

                # Confluence Decision Matrix
                if (is_bull_3m or is_bull_5m) and cobi_val >= 0.35 and vwap_status == "ABOVE (+)":
                    action = "🚀 INSTITUTIONAL PRO BUY"
                elif (is_bear_3m or is_bear_5m) and cobi_val <= -0.35 and vwap_status == "BELOW (-)":
                    action = "🔻 INSTITUTIONAL PRO SELL"
                elif is_bull_3m and is_bull_5m:
                    action = "🟢 SMC PRO BUY (3m+5m)"
                elif is_bull_3m or is_bull_5m:
                    action = "🟢 SMC PRO BUY"
                elif is_bear_3m and is_bear_5m:
                    action = "🔴 SMC PRO SELL (3m+5m)"
                elif is_bear_3m or is_bear_5m:
                    action = "🔴 SMC PRO SELL"
                elif cobi_val >= 0.40 and p_change > 0.5 and vol_chg_pct > 10:
                    action = "🟢 STRONG BUY"
                elif cobi_val <= -0.40 and p_change < -0.5 and vol_chg_pct > 10:
                    action = "🔴 STRONG SELL"
                elif cobi_val > 0.25 and vwap_status == "ABOVE (+)":
                    action = "🟢 ACCUMULATION"
                elif cobi_val < -0.25 and vwap_status == "BELOW (-)":
                    action = "🔴 DISTRIBUTION"
                else:
                    action = "🟡 SIDEWAYS"

                # Mobile Responsive Card Layout (Under 40 chars width)
                card = (
                    f"┌── {sym} ─────────────\n"
                    f"│ ₹{ltp:.2f} ({p_change:+.2f}%) | Vol:{vol_chg_pct:+.1f}%\n"
                    f"│ 3m:{ema3_col.split()[0]} {box3_t:<6} | 5m:{ema5_col.split()[0]} {box5_t:<6}\n"
                    f"│ COBI:{cobi_val:+.2f} | B/S:{bs_ratio:.2f} | {vwap_status}\n"
                    f"└► {action}"
                )

                full_cards.append(card)
                if any(k in action for k in ["INSTITUTIONAL", "SMC PRO", "STRONG"]):
                    filtered_cards.append(card)

            except Exception:
                continue

        # Atomic Stream Construction (Eliminates screen scrolling & blinking)
        buf = io.StringIO()
        div = "=" * 36
        cur_time = datetime.now().strftime('%H:%M:%S')

        buf.write(f"{div}\n")
        buf.write(f"📊 SMC+COBI QUANT SCANNER\n")
        buf.write(f"🕒 {cur_time} | Cycle: #{cycle} | [₹300-600]\n")
        buf.write(f"{div}\n\n")

        buf.write(f"🎯 HIGH CONVICTION SETUPS ({len(filtered_cards)})\n")
        buf.write(f"{'-' * 36}\n")
        if filtered_cards:
            for c in filtered_cards:
                buf.write(c + "\n\n")
        else:
            buf.write("⚡ No High-Conviction setups active.\n\n")

        buf.write(f"📋 ALL MONITORED STOCKS ({len(full_cards)})\n")
        buf.write(f"{'-' * 36}\n")
        if full_cards:
            for c in full_cards:
                buf.write(c + "\n\n")
        else:
            buf.write("No stocks matching ₹300-600 criteria.\n\n")
        buf.write(f"{div}\n")

        rendered_output = buf.getvalue()
        buf.close()

        # Non-jumping instant refresh
        clear_output(wait=True)
        sys.stdout.write(rendered_output)
        sys.stdout.flush()

        cycle += 1
        time.sleep(15)

    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
        break
    except Exception:
        time.sleep(5)
        continue
