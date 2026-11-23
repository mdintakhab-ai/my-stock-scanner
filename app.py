# =============================================================================
# FALCON NEXT-DAY BEHAVIOUR PREDICTOR V5.5 — VALIDATION-FIRST BUILD
# - Zero Hardcoded / Fallback Shares (100% Dynamic Live NSE Universe)
# - Randomized scan order; deterministic candidate ranking
# - Locked Core Level & Profile Formulas (Zero Decimal Change)
# - 10-Parameter Correlated U/D + Rejection Zone Actionability Filter
# - Mandatory Rejection Glow Filter: Lower (Green) OR Upper (Red) OR Both Must Glow
# - Subtle ~10% Glow Intensity on Numeric Values Only
# =============================================================================

import os
import re
import io
import sys
import time
import math
import zipfile
import warnings
import logging
import subprocess
import urllib.parse
import threading
from io import StringIO
from zipfile import ZipFile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])
    import yfinance as yf

try:
    import feedparser
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "feedparser"])
    import feedparser


warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
for logger_name in ["yfinance", "urllib3", "requests", "peewee"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

IST = ZoneInfo("Asia/Kolkata")
RNG = np.random.default_rng(int(time.time() * 1_000_000) % (2**32 - 1))

# =============================================================================
# CONFIGURATION
# =============================================================================
STEP1_CONFIG = {
    "MIN_PRICE": 300.0,
    "MAX_PRICE": 551.0,
    "MIN_AVG_TURNOVER": 15000000.0,
    "MIN_BETA": 0.10,
    "MAX_BETA": 1.25,
    "MIN_ATR_PCT": 1.20,
    "NUM_BATCHES": 24,
    "CANDIDATE_POOL_SIZE": 140,
    "MIN_CONVICTION": 20.0,
}

TARGET_FINAL_STOCKS = 7
STRICT_MIN_CHECKS = 4
FALLBACK_MIN_CHECKS = 3
MIN_MAX_QTY_STRICT = 300.0
MIN_MAX_QTY_FALLBACK = 150.0

PROFILE_LEN = 50
VALUE_AREA_MULT = 1.0
ATR_LEN = 14
DOWNLOAD_RETRIES = 1

DIRECTION_SCALE = 0.50
GAP_SCALE = 5.0

TREND_WEIGHTS = {
    "MID": 3.0,
    "VAH": 1.0,
    "VAL": 1.0,
    "POC": 3.0,
}

SESSION_START = "09:15"
SESSION_END = "15:30"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "*/*",
}

FALCON_ERRORS = []
NSE_CACHE = {}
YF_DAILY_CACHE = {}
NSE_PREOPEN_MAP = {}  # IEP only: indicative, never treated as actual open
NSE_ORDER_IMBALANCE_MAP = {}
NSE_ACTUAL_OPEN_MAP = {}
NSE_QUOTE_PREV_CLOSE_MAP = {}
NSE_OPEN_SOURCE_MAP = {}
NSE_OPEN_CROSSCHECK_DIFF_MAP = {}
NSE_THREAD_LOCAL = threading.local()
NSE_BHAVCOPY_INDEX = {}


def now_ist():
    return datetime.now(IST)


def is_after_market_open_ist():
    current_t = now_ist().time()
    open_t = pd.to_datetime(SESSION_START).time()
    return current_t >= open_t


def safe_float(value, default=np.nan):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def empty_ohlcv():
    return pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    )


def clean_dataframe(df):
    if df is None or df.empty:
        return empty_ohlcv()

    try:
        data = df.copy()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(col[0]) for col in data.columns]

        data.columns = [str(col).strip().title() for col in data.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required):
            return empty_ohlcv()

        data = data[required].copy()
        for col in required:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["Open", "High", "Low", "Close"])
        data = data[
            (data["Close"] > 0)
            & (data["High"] >= data["Low"])
            & (data["High"] >= data["Open"])
            & (data["High"] >= data["Close"])
            & (data["Low"] <= data["Open"])
            & (data["Low"] <= data["Close"])
        ]
        data["Volume"] = data["Volume"].fillna(0).clip(lower=0)
        data = data[~data.index.duplicated(keep="last")]
        return data.sort_index()

    except Exception as exc:
        FALCON_ERRORS.append(f"clean_dataframe: {exc}")
        return empty_ohlcv()


def convert_to_ist(df):
    if df is None or df.empty:
        return empty_ohlcv()

    data = df.copy()
    try:
        idx = pd.DatetimeIndex(data.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(IST)
        else:
            idx = idx.tz_convert(IST)
        data.index = idx
        data = data[~data.index.isna()]
        return data.sort_index()
    except Exception as exc:
        FALCON_ERRORS.append(f"convert_to_ist: {exc}")
        return empty_ohlcv()


def market_hours_only(df):
    if df is None or df.empty:
        return empty_ohlcv()

    start = pd.to_datetime(SESSION_START).time()
    end = pd.to_datetime(SESSION_END).time()
    mask = [start <= timestamp.time() <= end for timestamp in df.index]
    return df.loc[mask].copy()


def exclude_today(df):
    if df is None or df.empty:
        return empty_ohlcv()

    today = now_ist().date()
    mask = [timestamp.date() < today for timestamp in df.index]
    return df.loc[mask].copy()


def round_profile_value(value):
    number = safe_float(value)
    return round(number, 2) if np.isfinite(number) else np.nan


# =============================================================================
# DYNAMIC LIVE NSE UNIVERSE (ZERO HARDCODED / ZERO FALLBACK SHARES)
# =============================================================================
def get_live_nse_symbols_step1():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
    }
    urls = [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code != 200 or not r.text.strip():
                continue
            df_nse = pd.read_csv(StringIO(r.text))
            df_nse.columns = df_nse.columns.str.strip()
            if "SERIES" in df_nse.columns and "SYMBOL" in df_nse.columns:
                eq_df = df_nse[df_nse["SERIES"].astype(str).str.strip() == "EQ"]
                syms = eq_df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
                if syms:
                    return [f"{s}.NS" for s in syms if s]
            for col_cand in ["Symbol", "SYMBOL", "TckrSymb"]:
                if col_cand in df_nse.columns:
                    syms = df_nse[col_cand].dropna().astype(str).str.strip().unique().tolist()
                    if syms:
                        return [f"{s}.NS" for s in syms if s]
        except Exception as exc:
            FALCON_ERRORS.append(f"NSE symbol fetch ({url}): {exc}")

    raise RuntimeError("Live NSE symbol list fetch failed; zero hardcoded fallback allowed.")


def fetch_batch_1y_daily(symbols_list):
    out = {}
    missing = [s for s in symbols_list if s not in YF_DAILY_CACHE]
    for s in symbols_list:
        if s in YF_DAILY_CACHE:
            out[s] = YF_DAILY_CACHE[s]

    if missing:
        try:
            raw_all = yf.download(
                tickers=missing,
                period="1y",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if raw_all is not None and not raw_all.empty:
                if len(missing) == 1:
                    s = missing[0]
                    cleaned = clean_dataframe(raw_all)
                    YF_DAILY_CACHE[s] = cleaned
                    out[s] = cleaned
                elif isinstance(raw_all.columns, pd.MultiIndex):
                    lvl0 = set(raw_all.columns.get_level_values(0))
                    for s in missing:
                        cleaned = clean_dataframe(raw_all[s]) if s in lvl0 else empty_ohlcv()
                        YF_DAILY_CACHE[s] = cleaned
                        out[s] = cleaned
        except Exception as exc:
            FALCON_ERRORS.append(f"Batch 1y daily download: {exc}")

    for s in missing:
        if s not in out:
            YF_DAILY_CACHE[s] = empty_ohlcv()
            out[s] = empty_ohlcv()
    return out


# =============================================================================
# HTF FRESH DEMAND & SUPPLY ZONES + REJECTION ZONE VALIDATION
# =============================================================================
def detect_fresh_zones_on_htf(df_htf, current_close):
    """Returns list of dicts: {'bot': z_bot, 'top': z_top, 'strength': float}."""
    fresh_demands = []
    fresh_supplies = []
    if df_htf is None or len(df_htf) < 6:
        return fresh_demands, fresh_supplies

    atr_series = (df_htf["High"] - df_htf["Low"]).rolling(5, min_periods=1).mean()

    for i in range(len(df_htf) - 3, 1, -1):
        body = abs(float(df_htf["Close"].iloc[i]) - float(df_htf["Open"].iloc[i]))
        rng = float(df_htf["High"].iloc[i]) - float(df_htf["Low"].iloc[i])
        next_body = abs(float(df_htf["Close"].iloc[i + 1]) - float(df_htf["Open"].iloc[i + 1]))
        if rng <= 0:
            continue

        local_atr = max(float(atr_series.iloc[i + 1]), 1e-6)
        impulse_ratio = next_body / local_atr

        if (body / rng) < 0.48 and next_body > (1.6 * body):
            if float(df_htf["Close"].iloc[i + 1]) > float(df_htf["High"].iloc[i]):
                z_top = max(float(df_htf["Open"].iloc[i]), float(df_htf["Close"].iloc[i]))
                z_bot = float(df_htf["Low"].iloc[i])
                if float(df_htf["Low"].iloc[i + 2:].min()) > z_top and current_close >= z_bot:
                    strength = min(100.0, (next_body / max(body, 1e-6)) * 25.0 + impulse_ratio * 20.0)
                    fresh_demands.append({"bot": z_bot, "top": z_top, "strength": float(strength)})
            elif float(df_htf["Close"].iloc[i + 1]) < float(df_htf["Low"].iloc[i]):
                z_bot = min(float(df_htf["Open"].iloc[i]), float(df_htf["Close"].iloc[i]))
                z_top = float(df_htf["High"].iloc[i])
                if float(df_htf["High"].iloc[i + 2:].max()) < z_bot and current_close <= z_top:
                    strength = min(100.0, (next_body / max(body, 1e-6)) * 25.0 + impulse_ratio * 20.0)
                    fresh_supplies.append({"bot": z_bot, "top": z_top, "strength": float(strength)})

    return fresh_demands, fresh_supplies


def analyze_mtf_htf_zones_step1(df_daily, current_close, atr_14):
    df_w = df_daily.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    df_m = df_daily.resample("ME").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    w_dem, w_sup = detect_fresh_zones_on_htf(df_w, current_close)
    m_dem, m_sup = detect_fresh_zones_on_htf(df_m, current_close)

    all_dem = w_dem + m_dem
    all_sup = w_sup + m_sup

    nearest_dem = all_dem[0] if all_dem else None
    nearest_sup = all_sup[0] if all_sup else None

    hr_up = ((nearest_sup["bot"] - current_close) / atr_14) if (nearest_sup and nearest_sup["bot"] > current_close and atr_14 > 0) else 99.0
    hr_down = ((current_close - nearest_dem["top"]) / atr_14) if (nearest_dem and nearest_dem["top"] < current_close and atr_14 > 0) else 99.0

    zone_signal = 0
    if any(current_close <= z["top"] * 1.03 for z in all_dem):
        zone_signal = 1
    elif any(current_close >= z["bot"] * 0.97 for z in all_sup):
        zone_signal = -1

    return zone_signal, hr_up, hr_down, all_dem, all_sup


def evaluate_rejection_zones_with_correlation(df_daily, lower_rej, upper_rej, atr_14, ud_bias, ud_score, step1_meta=None):
    """
    Checks fresh demand zone near Lower Rejection and fresh supply zone near Upper Rejection,
    correlating zone quality with the 10-parameter directional alignment.
    Returns:
      glow_lower (bool): True if Lower Rejection has a valid actionable demand zone to glow green
      glow_upper (bool): True if Upper Rejection has a valid actionable supply zone to glow red
      zone_correlation_score (float)
    """
    lr = safe_float(lower_rej)
    ur = safe_float(upper_rej)
    if df_daily is None or df_daily.empty or not (np.isfinite(lr) and np.isfinite(ur)):
        return False, False, 0.0

    last_close = safe_float(df_daily["Close"].iloc[-1], lr)
    tol_abs = max(safe_float(atr_14, 0.0) * 0.85, abs(last_close) * 0.012, 0.75)

    df_w = df_daily.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    d_d, s_d = detect_fresh_zones_on_htf(df_daily, max(last_close, ur, lr))
    d_w, s_w = detect_fresh_zones_on_htf(df_w, max(last_close, ur, lr))

    all_demands = d_d + d_w
    all_supplies = s_d + s_w

    dem_score = 0.0
    for z in all_demands:
        bot, top, base_str = z["bot"], z["top"], z["strength"]
        if bot - tol_abs <= lr <= top + tol_abs:
            dist = 0.0
        else:
            dist = min(abs(lr - bot), abs(lr - top))
        if dist <= tol_abs * 2.0:
            proximity_factor = max(0.30, 1.0 - (dist / (tol_abs * 2.0 + 1e-6)))
            dem_score = max(dem_score, base_str * proximity_factor)

    sup_score = 0.0
    for z in all_supplies:
        bot, top, base_str = z["bot"], z["top"], z["strength"]
        if bot - tol_abs <= ur <= top + tol_abs:
            dist = 0.0
        else:
            dist = min(abs(ur - bot), abs(ur - top))
        if dist <= tol_abs * 2.0:
            proximity_factor = max(0.30, 1.0 - (dist / (tol_abs * 2.0 + 1e-6)))
            sup_score = max(sup_score, base_str * proximity_factor)

    # Correlate zone actionability with the 10-parameter score & trend structure
    meta = step1_meta or {}
    corr_factor = 1.0
    if abs(safe_float(ud_score, 0.0)) >= 1.5:
        corr_factor += 0.15
    if str(ud_bias).upper() == "BULLISH" and dem_score > 0:
        corr_factor += 0.10
    elif str(ud_bias).upper() == "BEARISH" and sup_score > 0:
        corr_factor += 0.10
    if bool(meta.get("is_over_extended", False)):
        corr_factor *= 0.85

    eff_dem = dem_score * corr_factor
    eff_sup = sup_score * corr_factor

    # Threshold for actionable zone near rejection level
    min_zone_threshold = 12.0
    glow_lower = eff_dem >= min_zone_threshold
    glow_upper = eff_sup >= min_zone_threshold

    return glow_lower, glow_upper, round(max(eff_dem, eff_sup), 2)


# =============================================================================
# STEP 1 EVALUATION & 10-PARAMETER U/D DIRECTIONAL SCORING
# =============================================================================
def analyze_news_step1(symbol_clean, chg_pct, vol_ratio):
    query = urllib.parse.quote(f"{symbol_clean} share news NSE")
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    bull_words = ["profit", "surge", "jump", "win", "order", "growth", "high", "buy", "upgrade", "approval", "expansion", "deal"]
    bear_words = ["loss", "fall", "drop", "crash", "low", "sell", "downgrade", "penalty", "raid", "fraud", "weak", "decline", "slump"]

    try:
        feed = feedparser.parse(rss_url)
        entries = feed.entries[:2]
        if not entries:
            return 0
        sent_score = 0
        for entry in entries:
            t = entry.title.lower()
            for w in bull_words:
                if w in t:
                    sent_score += 1
            for w in bear_words:
                if w in t:
                    sent_score -= 1
        if sent_score > 0 and chg_pct < 0 and vol_ratio >= 1.3:
            sent_score -= 1
        elif sent_score < 0 and chg_pct > 0 and vol_ratio >= 1.3:
            sent_score += 1
        return max(-3, min(3, sent_score))
    except Exception:
        return 0


def evaluate_stock_from_df_step1(symbol, df, nifty_returns, relaxed=False, fetch_news=False):
    try:
        if df is None or df.empty or len(df) < 60:
            return None

        close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        high = float(df["High"].iloc[-1])
        low = float(df["Low"].iloc[-1])
        open_p = float(df["Open"].iloc[-1])
        vol = float(df["Volume"].iloc[-1])

        if not (STEP1_CONFIG["MIN_PRICE"] <= close <= STEP1_CONFIG["MAX_PRICE"]):
            return None

        avg_turnover = float((df["Close"] * df["Volume"]).rolling(20, min_periods=1).mean().iloc[-1])
        min_turn = STEP1_CONFIG["MIN_AVG_TURNOVER"] * (0.4 if relaxed else 1.0)
        if avg_turnover < min_turn:
            return None

        if nifty_returns is not None and not nifty_returns.empty:
            stock_returns = df["Close"].pct_change().dropna()
            aligned = pd.concat([stock_returns, nifty_returns], axis=1).dropna()
            if len(aligned) >= 40 and aligned.iloc[:, 1].var() > 0:
                cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
            else:
                beta = 0.5
        else:
            beta = 0.5

        if not relaxed:
            if not (STEP1_CONFIG["MIN_BETA"] <= beta <= STEP1_CONFIG["MAX_BETA"]):
                return None

        tr = np.maximum(
            df["High"] - df["Low"],
            np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))),
        )
        atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        atr_pct = (atr_14 / close) * 100.0
        if not relaxed and atr_pct < STEP1_CONFIG["MIN_ATR_PCT"]:
            return None

        day_range = high - low
        if day_range <= 0:
            return None
        clv = (close - low) / day_range

        recent = df.tail(10).copy()
        recent["Range"] = recent["High"] - recent["Low"]
        recent = recent[recent["Range"] > 0]
        recent["CLV"] = (recent["Close"] - recent["Low"]) / recent["Range"]

        green_days = recent[recent["Close"] > recent["Open"]]
        red_days = recent[recent["Close"] < recent["Open"]]
        avg_bull_clv = float(green_days["CLV"].mean()) if len(green_days) > 0 else 0.5
        avg_bear_clv = float(red_days["CLV"].mean()) if len(red_days) > 0 else 0.5

        if not relaxed:
            is_strong_bull = (clv >= 0.65) and (avg_bull_clv >= 0.55) and (close >= open_p)
            is_strong_bear = (clv <= 0.35) and (avg_bear_clv <= 0.45) and (close <= open_p)
            if not (is_strong_bull or is_strong_bear):
                return None
        else:
            is_strong_bull = close >= open_p
            is_strong_bear = close < open_p

        vol_sma_20 = float(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1])
        vol_ratio = (vol / vol_sma_20) if vol_sma_20 > 0 else 1.0

        ema_9 = float(df["Close"].ewm(span=9, adjust=False).mean().iloc[-1])
        ema_20 = float(df["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        ema_50 = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rsi_14 = float((100 - (100 / (1 + (gain / loss.replace(0, np.nan))))).fillna(50).iloc[-1])

        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_bull = float((ema_12 - ema_26).iloc[-1]) > float((ema_12 - ema_26).ewm(span=9, adjust=False).mean().iloc[-1])

        high_52w = float(df["High"].iloc[:-1].max())
        low_52w = float(df["Low"].iloc[:-1].min())
        broke_52w_high = close > high_52w
        broke_52w_low = close < low_52w

        p_h, p_l, p_c = float(df["High"].iloc[-2]), float(df["Low"].iloc[-2]), float(df["Close"].iloc[-2])
        piv = (p_h + p_l + p_c) / 3.0
        r2 = piv + (p_h - p_l)
        s2 = piv - (p_h - p_l)

        zone_signal, hr_up, hr_down, _, _ = analyze_mtf_htf_zones_step1(df, close, atr_14)
        lookback_idx = min(6, len(df))
        move_5d = abs(close - float(df["Close"].iloc[-lookback_idx]))
        is_over_extended = move_5d > (3.5 * atr_14)

        score = 0.0
        if is_strong_bull:
            score += 20.0
            if close > ema_20 and close > ema_50 and (50.0 <= rsi_14 <= 78.0):
                score += 20.0
            if macd_bull:
                score += 10.0
            if vol_ratio >= 1.4:
                score += 15.0
            if close > r2:
                score += 15.0
            if broke_52w_high:
                score += 25.0
            if zone_signal > 0:
                score += 20.0
            if hr_up < 1.5:
                score -= 20.0
        elif is_strong_bear:
            score -= 20.0
            if close < ema_20 and close < ema_50 and (22.0 <= rsi_14 <= 50.0):
                score -= 20.0
            if not macd_bull:
                score -= 10.0
            if vol_ratio >= 1.4:
                score -= 15.0
            if close < s2:
                score -= 15.0
            if broke_52w_low:
                score -= 25.0
            if zone_signal < 0:
                score -= 20.0
            if hr_down < 1.5:
                score += 20.0

        if is_over_extended:
            score = score * 0.80

        if not relaxed and abs(score) < STEP1_CONFIG["MIN_CONVICTION"]:
            return None

        sym_clean = symbol.replace(".NS", "")
        chg_pct = ((close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        n_score = analyze_news_step1(sym_clean, chg_pct, vol_ratio) if (fetch_news and not relaxed) else 0
        if fetch_news and not relaxed:
            score = float(np.clip(score + (n_score * 5.0), -100.0, 100.0))

        if score == 0.0:
            score = 25.0 if close >= open_p else -25.0

        return {
            "Stock": sym_clean,
            "Symbol": symbol,
            "Direction": "BULLISH" if score > 0 else "BEARISH",
            "Conviction": score,
            "Abs_Conviction": abs(score),
            "_step1_meta": {
                "close": close,
                "ema_9": ema_9,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "rsi_14": rsi_14,
                "clv": clv,
                "avg_bull_clv": avg_bull_clv,
                "avg_bear_clv": avg_bear_clv,
                "vol_ratio": vol_ratio,
                "zone_signal": zone_signal,
                "is_over_extended": is_over_extended,
                "news_score": n_score,
                "prior_day_return_pct": (
                    ((float(df.loc[[ts.date() < now_ist().date() for ts in df.index], "Close"].iloc[-1]) -
                      float(df.loc[[ts.date() < now_ist().date() for ts in df.index], "Close"].iloc[-2])) /
                     float(df.loc[[ts.date() < now_ist().date() for ts in df.index], "Close"].iloc[-2])) * 100.0
                    if len(df.loc[[ts.date() < now_ist().date() for ts in df.index], "Close"]) >= 2 and
                       float(df.loc[[ts.date() < now_ist().date() for ts in df.index], "Close"].iloc[-2]) > 0
                    else np.nan
                ),
            },
        }
    except Exception:
        return None


def run_step1_candidate_pool(all_symbols, nifty_returns, relaxed=False):
    shuffled = list(all_symbols)
    RNG.shuffle(shuffled)
    batches = [list(b) for b in np.array_split(shuffled, STEP1_CONFIG["NUM_BATCHES"]) if len(b) > 0]

    def scan_batch_fast(batch_syms):
        data_map = fetch_batch_1y_daily(batch_syms)
        batch_out = []
        for s in batch_syms:
            res = evaluate_stock_from_df_step1(
                s, data_map.get(s), nifty_returns, relaxed=relaxed, fetch_news=False
            )
            if res is not None:
                batch_out.append(res)
        return batch_out

    qualified = []
    with ThreadPoolExecutor(max_workers=min(12, len(batches) or 1)) as executor:
        futures = [executor.submit(scan_batch_fast, batch) for batch in batches]
        for fut in as_completed(futures):
            qualified.extend(fut.result())

    if not qualified:
        return []

    df_q = (
        pd.DataFrame(qualified)
        .drop_duplicates("Stock", keep="first")
    )
    # No random score perturbation: a stock's rank must not change merely because
    # the notebook was rerun. Random shuffling is retained only for workload distribution.
    df_q["_rank_key"] = pd.to_numeric(df_q["Abs_Conviction"], errors="coerce").fillna(-np.inf)
    df_q = df_q.sort_values(
        ["_rank_key", "Stock"], ascending=[False, True], kind="mergesort"
    ).drop(columns=["_rank_key"])

    pool_df = df_q.head(STEP1_CONFIG["CANDIDATE_POOL_SIZE"]).copy()
    records = pool_df.to_dict("records")
    RNG.shuffle(records)
    return records


def evaluate_ten_parameter_ud(item, step1_meta=None):
    """10 original structural votes plus opening-gap, valid imbalance, key-level and extension context.

    Original 10-parameter calculations and their thresholds are retained. Their net score
    is converted to a 10-point scale, then independent opening/session evidence is applied.
    Missing data contributes exactly zero; it is never fabricated as neutral evidence.
    """
    meta = step1_meta or {}
    bull_pts = 0.0
    bear_pts = 0.0

    # 1. Falcon Correlated Trend (original weight and threshold preserved)
    ft = safe_float(item.get("FALCON_CORRELATED_TREND_%"))
    if np.isfinite(ft):
        if ft >= 8.0:
            bull_pts += 1.5
        elif ft <= -8.0:
            bear_pts += 1.5

    # 2. Short-Term vs Long-Term Value Area Position (original)
    st_avg = safe_float(item.get("ST Avg"))
    lt_vah = safe_float(item.get("LT VAH Avg"))
    lt_val = safe_float(item.get("LT VAL"))
    if np.isfinite(st_avg) and np.isfinite(lt_vah) and np.isfinite(lt_val):
        lt_mid_ref = (lt_vah + lt_val) / 2.0
        if st_avg > lt_mid_ref:
            bull_pts += 1.0
        elif st_avg < lt_mid_ref:
            bear_pts += 1.0

    # 3. Target Open vs Support/Down & Reference Price (original)
    t_open = safe_float(item.get("Target Open"))
    s_down = safe_float(item.get("Support/Down"))
    ref_p = safe_float(item.get("Display_Prev_Or_Open"))
    if np.isfinite(t_open) and np.isfinite(s_down):
        mid_os = (t_open + s_down) / 2.0
        if np.isfinite(ref_p):
            if ref_p >= mid_os:
                bull_pts += 1.0
            else:
                bear_pts += 1.0
        elif t_open > s_down:
            bull_pts += 0.5

    # 4. Daily EMA Structure (original)
    close_v = safe_float(meta.get("close"), ref_p)
    ema9 = safe_float(meta.get("ema_9"), safe_float(item.get("Daily EMA9")))
    ema20 = safe_float(meta.get("ema_20"))
    ema50 = safe_float(meta.get("ema_50"))
    if np.isfinite(close_v) and np.isfinite(ema9):
        ema_bull = (1.0 if close_v > ema9 else 0.0) + (0.5 if (np.isfinite(ema20) and close_v > ema20) else 0.0) + (0.5 if (np.isfinite(ema50) and close_v > ema50) else 0.0)
        ema_bear = (1.0 if close_v < ema9 else 0.0) + (0.5 if (np.isfinite(ema20) and close_v < ema20) else 0.0) + (0.5 if (np.isfinite(ema50) and close_v < ema50) else 0.0)
        if ema_bull > ema_bear:
            bull_pts += 1.0
        elif ema_bear > ema_bull:
            bear_pts += 1.0

    # 5. Momentum & RSI (original)
    rsi_v = safe_float(meta.get("rsi_14"), safe_float(item.get("Daily RSI"), 50.0))
    if np.isfinite(rsi_v):
        if rsi_v >= 52.0:
            bull_pts += 1.0
        elif rsi_v <= 48.0:
            bear_pts += 1.0

    # 6. Candle Location Value (original)
    clv_v = safe_float(meta.get("clv"), 0.5)
    if np.isfinite(clv_v):
        if clv_v >= 0.55:
            bull_pts += 1.0
        elif clv_v <= 0.45:
            bear_pts += 1.0

    # 7. Volume & Turnover Conviction (original)
    vol_r = safe_float(meta.get("vol_ratio"), 1.0)
    if np.isfinite(vol_r) and vol_r >= 1.1:
        if bull_pts >= bear_pts:
            bull_pts += 1.0
        else:
            bear_pts += 1.0

    # 8. HTF Fresh Demand/Supply Zone Signal (original)
    z_sig = int(meta.get("zone_signal", 0))
    if z_sig > 0:
        bull_pts += 1.0
    elif z_sig < 0:
        bear_pts += 1.0

    # 9. 5-Day Extension Guard (original half-point adjustment retained)
    over_ext = bool(meta.get("is_over_extended", False))
    if over_ext:
        if bull_pts > bear_pts:
            bull_pts -= 0.5
        elif bear_pts > bull_pts:
            bear_pts -= 0.5

    # 10. News / Catalyst Sentiment (original single application retained)
    n_sc = safe_float(meta.get("news_score"), 0.0)
    if n_sc > 0:
        bull_pts += 1.0
    elif n_sc < 0:
        bear_pts += 1.0

    score = (bull_pts - bear_pts) * 10.0

    # Actual opening gap: official open is preferred; YF fallback must be today's bar only.
    actual_open = safe_float(item.get("Matched_Open_Price"))
    prev_close = safe_float(NSE_QUOTE_PREV_CLOSE_MAP.get(str(item.get("Internal Symbol", "")).upper().replace(".NS", "")))
    if not (np.isfinite(prev_close) and prev_close > 0):
        prev_close = safe_float(item.get("Previous Close"))
    if np.isfinite(actual_open) and actual_open > 0 and np.isfinite(prev_close) and prev_close > 0:
        gap_pct = ((actual_open - prev_close) / prev_close) * 100.0
        if 0.2 < gap_pct < 1.5:
            score += 20.0
        elif gap_pct >= 1.5:
            score += 15.0
        elif -1.5 < gap_pct < -0.2:
            score -= 20.0
        elif gap_pct <= -1.5:
            score -= 15.0
        item["Opening_Gap_%"] = round(gap_pct, 3)
    else:
        item["Opening_Gap_%"] = np.nan

    # Valid NSE pre-open order imbalance only. No data = no score.
    sym_clean = str(item.get("Internal Symbol", "")).upper().replace(".NS", "")
    imbalance_data = NSE_ORDER_IMBALANCE_MAP.get(sym_clean)
    if isinstance(imbalance_data, dict):
        imbalance = safe_float(imbalance_data.get("imbalance"))
        if np.isfinite(imbalance):
            if imbalance > 0.25:
                score += 15.0
            elif imbalance < -0.25:
                score -= 15.0
            item["Order_Imbalance"] = round(imbalance, 4)
        else:
            item["Order_Imbalance"] = np.nan
    else:
        item["Order_Imbalance"] = np.nan

    # Consolidated key-level vote: at most one +/-10 vote to avoid double counting.
    level_price = actual_open if np.isfinite(actual_open) and actual_open > 0 else ref_p
    level_votes = []
    if np.isfinite(level_price):
        r1 = safe_float(item.get("R1"))
        s1 = safe_float(item.get("S1"))
        st_vah = safe_float(item.get("ST VAH"))
        lt_val_level = safe_float(item.get("LT VAL"))
        if np.isfinite(r1) and level_price > r1:
            level_votes.append((1, abs(level_price-r1) / max(abs(level_price), 1.0)))
        if np.isfinite(st_vah) and level_price > st_vah:
            level_votes.append((1, abs(level_price-st_vah) / max(abs(level_price), 1.0)))
        if np.isfinite(s1) and level_price < s1:
            level_votes.append((-1, abs(level_price-s1) / max(abs(level_price), 1.0)))
        if np.isfinite(lt_val_level) and level_price < lt_val_level:
            level_votes.append((-1, abs(level_price-lt_val_level) / max(abs(level_price), 1.0)))
    if level_votes:
        # Only the nearest decisive boundary can vote; equally near opposite signals cancel.
        level_votes.sort(key=lambda x: x[1])
        nearest_distance = level_votes[0][1]
        nearest_signs = {v[0] for v in level_votes if abs(v[1] - nearest_distance) <= 1e-9}
        if len(nearest_signs) == 1:
            score += 10.0 * next(iter(nearest_signs))

    # Prior completed session's one-day extension penalty / bounce-watch adjustment.
    prior_day_return = safe_float(meta.get("prior_day_return_pct"))
    preliminary = score
    if np.isfinite(prior_day_return):
        if prior_day_return >= 8.0 and preliminary > 0:
            score -= 20.0
            item["Extension_Adjustment"] = "GAP-UP TRAP RISK (-20)"
        elif prior_day_return <= -8.0 and preliminary < 0:
            score += 15.0
            item["Extension_Adjustment"] = "BOUNCE WATCH (+15)"
        else:
            item["Extension_Adjustment"] = "NONE"
    else:
        item["Extension_Adjustment"] = "UNAVAILABLE"

    item["UD_Composite_Score"] = round(float(score), 2)
    # A deliberate no-direction band reduces forced bullish/bearish labels.
    if score > 25.0:
        return "BULLISH", float(score)
    if score < -25.0:
        return "BEARISH", float(score)
    return "NEUTRAL", float(score)


# =============================================================================
# FAST NSE BHAVCOPY INDEX & PRE-OPEN CACHE
# =============================================================================
def normalize_column_name(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def find_nse_column(columns, aliases):
    normalized = {normalize_column_name(col): col for col in columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    for alias in aliases:
        alias_key = normalize_column_name(alias)
        if len(alias_key) >= 6:
            for key, original in normalized.items():
                if alias_key in key:
                    return original
    return None


def download_nse_bhavcopy(trading_date):
    date_key = trading_date.strftime("%Y%m%d")
    if date_key in NSE_CACHE:
        return NSE_CACHE[date_key]

    dmy = trading_date.strftime("%d%m%Y")
    urls = [
        f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_key}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{dmy}.csv",
    ]

    for url in urls:
        try:
            response = requests.get(url, headers=NSE_HEADERS, timeout=10)
            if response.status_code != 200:
                continue
            content = response.content
            if url.endswith(".zip"):
                with ZipFile(io.BytesIO(content)) as archive:
                    csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                    if not csv_files:
                        continue
                    with archive.open(csv_files[0]) as file:
                        frame = pd.read_csv(file)
            else:
                frame = pd.read_csv(io.BytesIO(content))

            if frame is not None and not frame.empty:
                NSE_CACHE[date_key] = frame
                return frame
        except Exception as exc:
            FALCON_ERRORS.append(f"Bhavcopy {date_key}: {exc}")

    NSE_CACHE[date_key] = pd.DataFrame()
    return NSE_CACHE[date_key]


def build_fast_bhavcopy_lookup(completed_dates):
    for trading_date in completed_dates[-7:]:
        date_key = trading_date.strftime("%Y%m%d")
        if date_key in NSE_BHAVCOPY_INDEX:
            continue
        bhav = download_nse_bhavcopy(trading_date)
        sym_map = {}
        if bhav is not None and not bhav.empty:
            symbol_col = find_nse_column(bhav.columns, ["TckrSymb", "SYMBOL", "TCKRSYMB", "Security"])
            volume_col = find_nse_column(
                bhav.columns,
                ["TtlTradgVol", "TTLTRADGVOL", "TTLTRDGVOL", "TOTTRDQTY", "TTL_TRD_QNTY", "TOTAL TRADED QUANTITY"]
            )
            trades_col = find_nse_column(
                bhav.columns,
                ["TtlNbOfTxs", "TTLNBOFTXS", "NO_OF_TRADES", "NOOFTRADES", "TOTALTRADES", "TOTTRADES", "TTLTRDNB", "TOTAL TRADES", "TOTAL NUMBER OF TRADES"]
            )
            series_col = find_nse_column(bhav.columns, ["SctySrs", "SERIES"])

            if symbol_col and volume_col and trades_col:
                df_sub = bhav.copy()
                if series_col is not None:
                    eq_sub = df_sub[df_sub[series_col].astype(str).str.upper().str.strip() == "EQ"]
                    if not eq_sub.empty:
                        df_sub = eq_sub
                for _, r in df_sub.iterrows():
                    sym = str(r[symbol_col]).strip().upper()
                    v = safe_float(r[volume_col], np.nan)
                    t = safe_float(r[trades_col], np.nan)
                    if sym and np.isfinite(v) and np.isfinite(t) and v > 0 and t > 0:
                        sym_map[sym] = (v, t)
        NSE_BHAVCOPY_INDEX[date_key] = sym_map


def preload_recent_bhavcopies(nifty_daily_dates):
    if not nifty_daily_dates:
        return
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(download_nse_bhavcopy, nifty_daily_dates[-7:]))
    build_fast_bhavcopy_lookup(nifty_daily_dates[-7:])


def _find_nested_numeric_key(obj, aliases):
    """Find a positive numeric field in nested NSE pre-open JSON without assuming one schema."""
    wanted = {normalize_column_name(x) for x in aliases}
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if normalize_column_name(key) in wanted:
                number = safe_float(value)
                if np.isfinite(number) and number > 0:
                    found.append(number)
            if isinstance(value, (dict, list)):
                nested = _find_nested_numeric_key(value, aliases)
                if nested is not None:
                    found.append(nested)
    elif isinstance(obj, list):
        for value in obj:
            nested = _find_nested_numeric_key(value, aliases)
            if nested is not None:
                found.append(nested)
    return found[0] if found else None


def _find_nested_timestamp(obj):
    """Return a parseable NSE timestamp; absence means snapshot freshness is unverified."""
    aliases = {"timestamp", "lastupdatetime", "updatedat", "updatetime", "tradetime", "marketstatus"}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if normalize_column_name(key) in aliases and isinstance(value, (str, int, float)):
                try:
                    if isinstance(value, (int, float)) and value > 1_000_000_000:
                        stamp = pd.to_datetime(value, unit="s", utc=True).tz_convert(IST)
                    else:
                        stamp = pd.to_datetime(value, errors="coerce")
                        if pd.isna(stamp):
                            continue
                        if stamp.tzinfo is None:
                            stamp = stamp.tz_localize(IST)
                        else:
                            stamp = stamp.tz_convert(IST)
                    if not pd.isna(stamp):
                        return stamp
                except Exception:
                    pass
            if isinstance(value, (dict, list)):
                stamp = _find_nested_timestamp(value)
                if stamp is not None:
                    return stamp
    elif isinstance(obj, list):
        for value in obj:
            stamp = _find_nested_timestamp(value)
            if stamp is not None:
                return stamp
    return None


def preload_nse_preopen_once():
    """Cache IEP and genuine buy/sell quantities only from a timestamped current-session snapshot.

    IEP is stored separately and is NEVER used as the actual exchange open.
    Missing quantities or unverified/stale timestamps are treated as unavailable, not neutral.
    """
    try:
        sess = requests.Session()
        sess.headers.update(NSE_HEADERS)
        sess.get("https://www.nseindia.com/", timeout=4)
        pre_resp = sess.get(
            "https://www.nseindia.com/api/market-data-pre-open?key=ALL",
            timeout=5,
        )
        if pre_resp.status_code != 200:
            return
        payload = pre_resp.json()
        snapshot_time = _find_nested_timestamp(payload)
        snapshot_is_valid = False
        if snapshot_time is not None:
            now_time = now_ist()
            snapshot_is_valid = (
                snapshot_time.date() == now_time.date()
                and pd.to_datetime("09:00").time() <= snapshot_time.time() <= pd.to_datetime("09:15").time()
            )
        for item in payload.get("data", []):
            sym = str(item.get("symbol", "")).upper().strip()
            if not sym:
                continue
            iep = safe_float(item.get("iep"))
            if np.isfinite(iep) and iep > 0:
                NSE_PREOPEN_MAP[sym] = iep

            if not snapshot_is_valid:
                continue
            buy_qty = _find_nested_numeric_key(item, [
                "totalBuyQuantity", "totalBuyQty", "buyQuantity", "buyQty", "totalBuyOrdersQuantity"
            ])
            sell_qty = _find_nested_numeric_key(item, [
                "totalSellQuantity", "totalSellQty", "sellQuantity", "sellQty", "totalSellOrdersQuantity"
            ])
            if buy_qty is not None and sell_qty is not None and (buy_qty + sell_qty) > 0:
                imbalance = (buy_qty - sell_qty) / (buy_qty + sell_qty)
                if np.isfinite(imbalance) and -1.0 <= imbalance <= 1.0:
                    NSE_ORDER_IMBALANCE_MAP[sym] = {
                        "imbalance": float(imbalance),
                        "buy_qty": float(buy_qty),
                        "sell_qty": float(sell_qty),
                        "source": "NSE_PREOPEN",
                    }
    except Exception as exc:
        FALCON_ERRORS.append(f"NSE pre-open cache: {exc}")


def _nse_quote_session():
    sess = getattr(NSE_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(NSE_HEADERS)
        try:
            sess.get("https://www.nseindia.com/", timeout=3)
        except Exception:
            pass
        NSE_THREAD_LOCAL.session = sess
    return sess


def fetch_nse_actual_quote(symbol):
    """Fetch NSE official priceInfo.open and previousClose for one symbol."""
    sym = str(symbol).upper().replace(".NS", "").strip()
    if not sym:
        return sym, np.nan, np.nan, "INVALID_SYMBOL"
    try:
        sess = _nse_quote_session()
        url = f"https://www.nseindia.com/api/quote-equity?symbol={urllib.parse.quote(sym)}"
        headers = dict(NSE_HEADERS)
        headers["Referer"] = f"https://www.nseindia.com/get-quotes/equity?symbol={urllib.parse.quote(sym)}"
        resp = sess.get(url, headers=headers, timeout=3.5)
        if resp.status_code in (401, 403):
            try:
                sess.cookies.clear()
                sess.get("https://www.nseindia.com/", timeout=3)
                resp = sess.get(url, headers=headers, timeout=3.5)
            except Exception:
                pass
        resp.raise_for_status()
        info = resp.json().get("priceInfo", {})
        op = safe_float(info.get("open"))
        pc = safe_float(info.get("previousClose"))
        if not (np.isfinite(op) and op > 0):
            op = np.nan
        if not (np.isfinite(pc) and pc > 0):
            pc = np.nan
        return sym, op, pc, "NSE_OFFICIAL_QUOTE" if np.isfinite(op) else "NSE_OPEN_NOT_AVAILABLE"
    except Exception as exc:
        return sym, np.nan, np.nan, f"NSE_QUOTE_UNAVAILABLE:{type(exc).__name__}"


def preload_nse_actual_quotes(symbols):
    """Parallel official quote lookup; bounded concurrency limits NSE load and runtime."""
    unique = list(dict.fromkeys(str(x).upper().replace(".NS", "") for x in symbols))
    if not is_after_market_open_ist() or not unique:
        return
    workers = min(8, len(unique))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_nse_actual_quote, sym) for sym in unique]
        for fut in as_completed(futures):
            try:
                sym, op, pc, source = fut.result()
                if sym:
                    if np.isfinite(op) and op > 0:
                        NSE_ACTUAL_OPEN_MAP[sym] = float(op)
                    if np.isfinite(pc) and pc > 0:
                        NSE_QUOTE_PREV_CLOSE_MAP[sym] = float(pc)
                    NSE_OPEN_SOURCE_MAP[sym] = source
            except Exception as exc:
                FALCON_ERRORS.append(f"NSE quote worker: {exc}")


def get_nse_stock_statistics(symbol, completed_dates):
    empty = {
        "5D Avg Volume": np.nan,
        "5D Avg Trades": np.nan,
        "MAX SINGLE-ORDER QTY": np.nan,
        "Valid NSE Sessions": 0,
    }

    try:
        if not completed_dates:
            return empty

        sym_clean = symbol.upper().replace(".NS", "")
        rows = []
        for trading_date in completed_dates[-5:]:
            date_key = trading_date.strftime("%Y%m%d")
            if date_key not in NSE_BHAVCOPY_INDEX:
                build_fast_bhavcopy_lookup([trading_date])
            pair = NSE_BHAVCOPY_INDEX.get(date_key, {}).get(sym_clean)
            if pair is not None:
                rows.append({"Volume": pair[0], "Trades": pair[1]})

        if not rows:
            return empty

        stats = pd.DataFrame(rows)
        avg_volume = stats["Volume"].mean()
        avg_trades = stats["Trades"].mean()
        estimated_max_qty = avg_volume * 6.25 / avg_trades if avg_trades > 0 else np.nan

        return {
            "5D Avg Volume": avg_volume,
            "5D Avg Trades": avg_trades,
            "MAX SINGLE-ORDER QTY": estimated_max_qty,
            "Valid NSE Sessions": len(stats),
        }

    except Exception as exc:
        FALCON_ERRORS.append(f"NSE statistics {symbol}: {exc}")
        return empty


# =============================================================================
# LOCKED CORE FORMULAS & PROFILES (ZERO DECIMAL CHANGE)
# =============================================================================
def yahoo_download(symbol, interval, period):
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                threads=False,
            )
            data = clean_dataframe(raw)
            if data.empty:
                continue
            data = convert_to_ist(data)
            if interval != "1d":
                data = market_hours_only(data)
            if not data.empty:
                return data
        except Exception as exc:
            FALCON_ERRORS.append(f"{symbol} {interval} {period}: {exc}")
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(0.15)
    return empty_ohlcv()


def download_fallback(symbol, interval, periods):
    for period in periods:
        data = yahoo_download(symbol, interval, period)
        if not data.empty:
            return data
    return empty_ohlcv()


def resample_ohlcv(df, rule, offset="15min"):
    if df is None or df.empty:
        return empty_ohlcv()

    try:
        data = df.resample(
            rule, origin="start_day", offset=offset, label="left", closed="left"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        data = data.dropna(subset=["Open", "High", "Low", "Close"])
        return clean_dataframe(data)
    except Exception as exc:
        FALCON_ERRORS.append(f"resample {rule}: {exc}")
        return empty_ohlcv()


def get_matched_actual_open_price(symbol, daily_raw, intraday_1m_raw):
    """Resolve today's actual open only; never borrow yesterday's first candle.

    Priority: NSE official quote -> today's first valid 1m bar -> today's daily open.
    NSE IEP is intentionally excluded because it is indicative, not traded opening price.
    """
    sym_clean = str(symbol).upper().replace(".NS", "")
    today = now_ist().date()
    official = safe_float(NSE_ACTUAL_OPEN_MAP.get(sym_clean))

    yf_1m_open = np.nan
    if intraday_1m_raw is not None and not intraday_1m_raw.empty:
        day_1m = intraday_1m_raw[intraday_1m_raw.index.date == today].sort_index()
        if not day_1m.empty:
            # Use first valid regular-session candle, not a stale prior-session candle.
            session_rows = day_1m[[pd.to_datetime(SESSION_START).time() <= ts.time() <= pd.to_datetime(SESSION_END).time() for ts in day_1m.index]]
            if not session_rows.empty:
                candidate = safe_float(session_rows.iloc[0].get("Open"))
                if np.isfinite(candidate) and candidate > 0:
                    yf_1m_open = candidate

    yf_1d_open = np.nan
    if daily_raw is not None and not daily_raw.empty:
        day_1d = daily_raw[daily_raw.index.date == today]
        if not day_1d.empty:
            candidate = safe_float(day_1d.iloc[-1].get("Open"))
            if np.isfinite(candidate) and candidate > 0:
                yf_1d_open = candidate

    if np.isfinite(official) and official > 0:
        yf_cross = yf_1m_open if np.isfinite(yf_1m_open) and yf_1m_open > 0 else yf_1d_open
        if np.isfinite(yf_cross) and yf_cross > 0:
            diff_pct = abs(official - yf_cross) / official * 100.0
            NSE_OPEN_CROSSCHECK_DIFF_MAP[sym_clean] = round(float(diff_pct), 4)
            NSE_OPEN_SOURCE_MAP[sym_clean] = (
                "NSE_OFFICIAL_QUOTE" if diff_pct <= 0.50
                else f"NSE_OFFICIAL_QUOTE_YF_DIFF_{diff_pct:.2f}%"
            )
        else:
            NSE_OPEN_CROSSCHECK_DIFF_MAP[sym_clean] = np.nan
            NSE_OPEN_SOURCE_MAP[sym_clean] = "NSE_OFFICIAL_QUOTE"
        return float(official)

    prev_close = safe_float(NSE_QUOTE_PREV_CLOSE_MAP.get(sym_clean))
    if not (np.isfinite(prev_close) and prev_close > 0):
        if daily_raw is not None and not daily_raw.empty:
            prev_rows = daily_raw[daily_raw.index.date < today]
            if not prev_rows.empty:
                prev_close = safe_float(prev_rows.iloc[-1].get("Close"))

    for candidate, source in [(yf_1m_open, "YF_TODAY_1M"), (yf_1d_open, "YF_TODAY_1D")]:
        if np.isfinite(candidate) and candidate > 0:
            # Guard only obvious bad prints; legitimate exchange gaps are retained.
            if np.isfinite(prev_close) and prev_close > 0:
                gap = abs((candidate - prev_close) / prev_close) * 100.0
                if gap > 25.0:
                    continue
            NSE_OPEN_SOURCE_MAP[sym_clean] = source
            return float(candidate)

    NSE_OPEN_SOURCE_MAP[sym_clean] = "TODAY_OPEN_UNAVAILABLE"
    return np.nan


def fetch_multitf_batch(symbols_list):
    results = {s: {} for s in symbols_list}
    preload_nse_actual_quotes(symbols_list)
    tf_specs = [
        ("1m", "1m", ["5d"]),
        ("5m", "5m", ["5d", "60d"]),
        ("15m", "15m", ["60d", "5d"]),
        ("1H", "60m", ["730d", "60d"]),
        ("Day", "1d", ["2y", "5y"]),
    ]

    for key, interval, periods in tf_specs:
        raw_df = None
        for per in periods:
            try:
                tmp = yf.download(
                    tickers=symbols_list,
                    period=per,
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if tmp is not None and not tmp.empty:
                    raw_df = tmp
                    break
            except Exception:
                continue

        if raw_df is None or raw_df.empty:
            for s in symbols_list:
                results[s][key] = empty_ohlcv()
            continue

        if len(symbols_list) == 1:
            s = symbols_list[0]
            cleaned = clean_dataframe(raw_df)
            if not cleaned.empty:
                cleaned = convert_to_ist(cleaned)
                if interval != "1d":
                    cleaned = market_hours_only(cleaned)
            results[s][key] = cleaned
        elif isinstance(raw_df.columns, pd.MultiIndex):
            lvl0 = set(raw_df.columns.get_level_values(0))
            for s in symbols_list:
                cleaned = clean_dataframe(raw_df[s]) if s in lvl0 else empty_ohlcv()
                if not cleaned.empty:
                    cleaned = convert_to_ist(cleaned)
                    if interval != "1d":
                        cleaned = market_hours_only(cleaned)
                results[s][key] = cleaned
        else:
            for s in symbols_list:
                results[s][key] = empty_ohlcv()

    final_bundles = {}
    for s in symbols_list:
        d_map = results[s]
        raw_1m = d_map.get("1m", empty_ohlcv())
        data_5m = d_map.get("5m", empty_ohlcv())
        data_15m = d_map.get("15m", empty_ohlcv())
        data_1h = d_map.get("1H", empty_ohlcv())
        raw_day = d_map.get("Day", empty_ohlcv())

        if raw_day.empty and s in YF_DAILY_CACHE and not YF_DAILY_CACHE[s].empty:
            raw_day = convert_to_ist(YF_DAILY_CACHE[s])

        matched_open = get_matched_actual_open_price(s, raw_day, raw_1m)
        data_3m = resample_ohlcv(raw_1m, "3min", offset="15min")
        data_4h = resample_ohlcv(data_1h, "4h", offset="15min")

        frames = {
            "1m": exclude_today(raw_1m),
            "3m": exclude_today(data_3m),
            "5m": exclude_today(data_5m),
            "15m": exclude_today(data_15m),
            "1H": exclude_today(data_1h),
            "4H": exclude_today(data_4h),
            "Day": exclude_today(raw_day),
        }
        final_bundles[s] = (frames, matched_open)

    return final_bundles


def calculate_auction_profile(df, profile_len=PROFILE_LEN, value_area_mult=VALUE_AREA_MULT):
    if (
        df is None
        or df.empty
        or not all(col in df.columns for col in ["High", "Low", "Close", "Volume"])
    ):
        return pd.DataFrame(columns=["POC", "VAH", "VAL", "Profile_Deviation", "Weighted_Variance"])

    try:
        data = df.copy()
        data["HLC3"] = (data["High"] + data["Low"] + data["Close"]) / 3.0
        data["Profile_Volume"] = data["Volume"].rolling(window=profile_len, min_periods=1).mean()
        data["Profile_PV"] = (data["HLC3"] * data["Volume"]).rolling(window=profile_len, min_periods=1).mean()
        data["POC"] = np.where(
            (data["Profile_Volume"].notna()) & (data["Profile_Volume"] > 0),
            data["Profile_PV"] / data["Profile_Volume"],
            np.nan,
        )
        data["Squared_Diff"] = (data["HLC3"] - data["POC"]) ** 2
        data["Weighted_Var_Numerator"] = (data["Volume"] * data["Squared_Diff"]).rolling(window=profile_len, min_periods=1).mean()
        data["Weighted_Variance"] = np.where(
            (data["POC"].notna()) & (data["Profile_Volume"] > 0),
            data["Weighted_Var_Numerator"] / data["Profile_Volume"],
            np.nan,
        )
        data["Profile_Deviation"] = np.sqrt(np.maximum(data["Weighted_Variance"], 0.0))
        data["VAH"] = np.where(
            (data["POC"].notna()) & (data["Profile_Deviation"].notna()),
            data["POC"] + data["Profile_Deviation"] * value_area_mult,
            np.nan,
        )
        data["VAL"] = np.where(
            (data["POC"].notna()) & (data["Profile_Deviation"].notna()),
            data["POC"] - data["Profile_Deviation"] * value_area_mult,
            np.nan,
        )
        return data
    except Exception as exc:
        FALCON_ERRORS.append(f"calculate_auction_profile: {exc}")
        return pd.DataFrame()


def extract_last_completed_profile(df, timeframe):
    if df is None or df.empty:
        return None
    profile = calculate_auction_profile(df, profile_len=PROFILE_LEN, value_area_mult=VALUE_AREA_MULT)
    if profile.empty:
        return None

    today = now_ist().date()
    dates = pd.Series(profile.index.date, index=profile.index)
    completed_dates = sorted({d for d in dates.unique() if d < today})
    if not completed_dates:
        return None

    last_date = completed_dates[-1]
    session = profile.loc[dates == last_date]
    if session.empty:
        return None

    last = session.iloc[-1]
    return {
        "Timeframe": timeframe,
        "Timestamp": session.index[-1].strftime("%Y-%m-%d %H:%M"),
        "POC": round_profile_value(last["POC"]),
        "VAH": round_profile_value(last["VAH"]),
        "VAL": round_profile_value(last["VAL"]),
        "Profile Rows": len(profile),
        "Session Rows": len(session),
        "Profile Status": "VALID",
    }


def timeframe_summary(timeframes):
    result = {}
    for timeframe in ["1m", "3m", "5m", "15m", "1H", "4H", "Day"]:
        frame = timeframes.get(timeframe, empty_ohlcv())
        extracted = extract_last_completed_profile(frame, timeframe)
        if extracted is None:
            result[timeframe] = {
                "POC": np.nan, "VAH": np.nan, "VAL": np.nan,
                "Timestamp": None, "Profile Status": "NO DATA",
            }
        else:
            result[timeframe] = extracted
    return result


def verify_profile_values(timeframes, profiles):
    checks = []
    for timeframe in ["1m", "5m", "15m", "1H", "4H", "Day"]:
        frame = timeframes.get(timeframe, empty_ohlcv())
        expected = profiles.get(timeframe, {})
        if frame.empty or not expected:
            checks.append({"Timeframe": timeframe, "Check": "NOT CHECKED", "POC Difference": np.nan, "VAH Difference": np.nan, "VAL Difference": np.nan})
            continue
        second = extract_last_completed_profile(frame.copy(), timeframe)
        if second is None:
            checks.append({"Timeframe": timeframe, "Check": "NO SECOND RESULT", "POC Difference": np.nan, "VAH Difference": np.nan, "VAL Difference": np.nan})
            continue
        differences = {}
        for key in ["POC", "VAH", "VAL"]:
            a = safe_float(expected.get(key))
            b = safe_float(second.get(key))
            differences[key] = round(abs(a - b), 8) if (np.isfinite(a) and np.isfinite(b)) else np.nan
        valid_diffs = [differences[k] for k in differences if np.isfinite(differences[k])]
        passed = len(valid_diffs) == 3 and all(x <= 0.00000001 for x in valid_diffs)
        checks.append({
            "Timeframe": timeframe,
            "Check": "PASS" if passed else "MISMATCH",
            "POC Difference": differences["POC"],
            "VAH Difference": differences["VAH"],
            "VAL Difference": differences["VAL"],
        })

    # Independent repeat calculation for the real 3-minute OHLCV profile.
    frame_3m = timeframes.get("3m", empty_ohlcv())
    p3 = profiles.get("3m", {})
    second_3m = extract_last_completed_profile(frame_3m.copy(), "3m") if not frame_3m.empty else None
    diffs = {}
    for key in ["POC", "VAH", "VAL"]:
        a = safe_float(p3.get(key))
        b = safe_float(second_3m.get(key)) if second_3m else np.nan
        diffs[key] = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan
    valid = [x for x in diffs.values() if np.isfinite(x)]
    checks.append({
        "Timeframe": "3m",
        "Check": "PASS" if (len(valid) == 3 and all(x <= 0.00000001 for x in valid)) else "MISMATCH",
        "POC Difference": diffs["POC"],
        "VAH Difference": diffs["VAH"],
        "VAL Difference": diffs["VAL"],
    })
    return pd.DataFrame(checks)


def calculate_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    data = df.copy()
    try:
        close = data["Close"].astype(float)
        high = data["High"].astype(float)
        low = data["Low"].astype(float)

        data["EMA9"] = close.ewm(span=9, adjust=False).mean()
        data["EMA21"] = close.ewm(span=21, adjust=False).mean()
        data["EMA50"] = close.ewm(span=50, adjust=False).mean()
        data["EMA200"] = close.ewm(span=200, adjust=False).mean()

        previous_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1)
        data["TR"] = tr
        atr = tr.ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=1).mean()
        data["ATR"] = atr

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
        rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100)
        rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
        data["RSI"] = rsi.clip(0, 100)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)

        plus_smoothed = plus_dm.ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=1).mean()
        minus_smoothed = minus_dm.ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=1).mean()

        atr_safe = atr.replace(0, np.nan)
        data["PLUS_DI"] = 100 * plus_smoothed / atr_safe
        data["MINUS_DI"] = 100 * minus_smoothed / atr_safe

        di_sum = (data["PLUS_DI"] + data["MINUS_DI"]).replace(0, np.nan)
        dx = 100 * (data["PLUS_DI"] - data["MINUS_DI"]).abs() / di_sum
        data["ADX"] = dx.ewm(alpha=1 / ATR_LEN, adjust=False, min_periods=1).mean()

        candle_range = (high - low).replace(0, np.nan)
        data["CLV"] = ((close - low) - (high - close)) / candle_range

        mean20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=2).std()
        data["BB_WIDTH"] = 4 * std20 / mean20.replace(0, np.nan)

        movement = close.diff().abs()
        numerator = close.diff(10).abs()
        denominator = movement.rolling(10, min_periods=1).sum()
        data["EFFICIENCY"] = (numerator / denominator.replace(0, np.nan)).clip(0, 1)
        return data
    except Exception as exc:
        FALCON_ERRORS.append(f"calculate_indicators: {exc}")
        return pd.DataFrame()


BEHAVIOUR_NAMES = [
    "Smooth Bullish",
    "Smooth Bearish",
    "Zigzag / Choppy",
    "Reversal",
    "Sideways",
]


def heuristic_share_percent(scores):
    values = np.array([scores[name] for name in BEHAVIOUR_NAMES], dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = exp_values.sum()
    if total <= 0 or not np.isfinite(total):
        return {name: 20.0 for name in BEHAVIOUR_NAMES}
    probabilities = exp_values / total * 100
    return {name: round(float(prob), 2) for name, prob in zip(BEHAVIOUR_NAMES, probabilities)}


def classify_behaviour(df):
    if df is None or len(df) < 5:
        return ({name: 0.0 for name in BEHAVIOUR_NAMES}, "Insufficient Data")
    data = calculate_indicators(df)
    if data.empty or len(data) < 2:
        return ({name: 0.0 for name in BEHAVIOUR_NAMES}, "Insufficient Data")

    last = data.iloc[-1]
    prev = data.iloc[-2]
    close = safe_float(last["Close"])
    atr = safe_float(last["ATR"], 0.0)
    ema9 = safe_float(last["EMA9"])
    ema21 = safe_float(last["EMA21"])
    ema50 = safe_float(last["EMA50"])
    rsi = safe_float(last["RSI"], 50.0)
    adx = safe_float(last["ADX"], 0.0)
    plus_di = safe_float(last["PLUS_DI"], 0.0)
    minus_di = safe_float(last["MINUS_DI"], 0.0)
    clv = safe_float(last["CLV"], 0.0)
    efficiency = safe_float(last["EFFICIENCY"], 0.0)
    bb_width = safe_float(last["BB_WIDTH"], 0.0)

    ema_spread = abs(ema9 - ema21) / close * 100 if close > 0 else 0
    candle_range = max(safe_float(last["High"]) - safe_float(last["Low"]), 0.000001)
    body_ratio = abs(safe_float(last["Close"]) - safe_float(last["Open"])) / candle_range
    momentum = (safe_float(last["Close"]) - safe_float(prev["Close"])) / atr if atr > 0 else 0

    scores = {name: 0.0 for name in BEHAVIOUR_NAMES}
    if ema9 > ema21:
        scores["Smooth Bullish"] += 1.5
    if close > ema50:
        scores["Smooth Bullish"] += 1.0
    if plus_di > minus_di:
        scores["Smooth Bullish"] += 1.0
    if clv > 0.25:
        scores["Smooth Bullish"] += 0.8
    if efficiency > 0.35:
        scores["Smooth Bullish"] += 0.8
    if adx > 20 and momentum > 0:
        scores["Smooth Bullish"] += 0.8

    if ema9 < ema21:
        scores["Smooth Bearish"] += 1.5
    if close < ema50:
        scores["Smooth Bearish"] += 1.0
    if minus_di > plus_di:
        scores["Smooth Bearish"] += 1.0
    if clv < -0.25:
        scores["Smooth Bearish"] += 0.8
    if efficiency > 0.35:
        scores["Smooth Bearish"] += 0.8
    if adx > 20 and momentum < 0:
        scores["Smooth Bearish"] += 0.8

    if efficiency < 0.25:
        scores["Zigzag / Choppy"] += 1.5
    if adx < 18:
        scores["Zigzag / Choppy"] += 0.7
    if bb_width > 0 and ema_spread < 0.25:
        scores["Zigzag / Choppy"] += 1.0
    if body_ratio < 0.35:
        scores["Zigzag / Choppy"] += 0.7

    if rsi < 35 or rsi > 65:
        scores["Reversal"] += 0.8
    if (rsi < 40 and clv > 0.25) or (rsi > 60 and clv < -0.25):
        scores["Reversal"] += 1.5
    if (close > ema9 and ema9 < ema21) or (close < ema9 and ema9 > ema21):
        scores["Reversal"] += 0.8
    if body_ratio > 0.55:
        scores["Reversal"] += 0.5

    if adx < 15:
        scores["Sideways"] += 1.4
    if ema_spread < 0.15:
        scores["Sideways"] += 1.2
    if efficiency < 0.20:
        scores["Sideways"] += 0.8
    if abs(momentum) < 0.25:
        scores["Sideways"] += 0.8

    probabilities = heuristic_share_percent(scores)
    winner = max(probabilities, key=probabilities.get)
    return probabilities, winner


def calculate_levels(previous_daily):
    if previous_daily is None or previous_daily.empty:
        return {}
    try:
        last = previous_daily.iloc[-1]
        high = safe_float(last["High"])
        low = safe_float(last["Low"])
        close = safe_float(last["Close"])
        if not all(np.isfinite(x) for x in [high, low, close]):
            return {}

        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        indicators = calculate_indicators(previous_daily)
        atr = safe_float(indicators.iloc[-1]["ATR"]) if not indicators.empty else np.nan
        bullish_scenario = close + 1.5 * atr if np.isfinite(atr) else np.nan
        bearish_scenario = close - 1.5 * atr if np.isfinite(atr) else np.nan
        bullish_pct = (bullish_scenario - close) / close * 100 if close > 0 and np.isfinite(bullish_scenario) else np.nan
        bearish_pct = (bearish_scenario - close) / close * 100 if close > 0 and np.isfinite(bearish_scenario) else np.nan

        return {
            "Previous Session": str(previous_daily.index[-1].date()),
            "Previous Close": close,
            "R1": r1,
            "R2": r2,
            "S1": s1,
            "S2": s2,
            "Daily ATR": atr,
            "Bullish ATR Scenario": bullish_scenario,
            "Bullish ATR Scenario %": bullish_pct,
            "Bearish ATR Scenario": bearish_scenario,
            "Bearish ATR Scenario %": bearish_pct,
        }
    except Exception as exc:
        FALCON_ERRORS.append(f"calculate_levels: {exc}")
        return {}


def historical_behaviour_report(intraday):
    if intraday is None or intraday.empty:
        return "Insufficient historical data"
    data = exclude_today(intraday)
    if data.empty:
        return "Insufficient completed sessions"

    try:
        data = data.copy()
        data["SessionDate"] = [timestamp.date() for timestamp in data.index]
        session_behaviours = []

        for _, session in data.groupby("SessionDate"):
            if len(session) < 3:
                continue
            open_price = safe_float(session.iloc[0]["Open"])
            close_price = safe_float(session.iloc[-1]["Close"])
            high_price = safe_float(session["High"].max())
            low_price = safe_float(session["Low"].min())
            if open_price <= 0:
                continue
            move_pct = (close_price - open_price) / open_price * 100
            range_pct = (high_price - low_price) / open_price * 100
            path = session["Close"].diff().abs().sum()
            efficiency = abs(close_price - open_price) / path if path > 0 else 0

            if move_pct > 0.8 and efficiency > 0.45:
                behaviour = "Smooth Bullish"
            elif move_pct < -0.8 and efficiency > 0.45:
                behaviour = "Smooth Bearish"
            elif range_pct < 0.8:
                behaviour = "Sideways"
            elif efficiency < 0.25:
                behaviour = "Zigzag / Choppy"
            else:
                behaviour = "Reversal"
            session_behaviours.append(behaviour)

        if not session_behaviours:
            return "Insufficient completed sessions"
        counts = pd.Series(session_behaviours).value_counts(normalize=True) * 100
        return " | ".join(f"{name}: {value:.1f}%" for name, value in counts.items())
    except Exception as exc:
        FALCON_ERRORS.append(f"Historical behaviour: {exc}")
        return "Historical analysis unavailable"


def mean_valid(values):
    valid = [safe_float(value) for value in values if np.isfinite(safe_float(value))]
    return float(np.mean(valid)) if valid else np.nan


def calculate_profile_summary(profiles):
    short_tfs = ["1m", "3m", "5m", "15m"]
    long_tfs = ["1H", "4H", "Day"]
    try:
        st_poc = mean_valid([profiles[tf]["POC"] for tf in short_tfs])
        st_vah = mean_valid([profiles[tf]["VAH"] for tf in short_tfs])
        st_val = mean_valid([profiles[tf]["VAL"] for tf in short_tfs])

        lt_poc = mean_valid([profiles[tf]["POC"] for tf in long_tfs])
        lt_vah = mean_valid([profiles[tf]["VAH"] for tf in long_tfs])
        lt_val = mean_valid([profiles[tf]["VAL"] for tf in long_tfs])

        widths = [
            safe_float(profiles[tf]["VAH"]) - safe_float(profiles[tf]["VAL"])
            for tf in short_tfs + long_tfs
            if np.isfinite(safe_float(profiles[tf]["VAH"])) and np.isfinite(safe_float(profiles[tf]["VAL"]))
        ]
        avg_width = float(np.mean(widths)) if widths else np.nan
        one_min_poc = safe_float(profiles["1m"]["POC"])

        target_open = one_min_poc + 0.382 * avg_width if (np.isfinite(one_min_poc) and np.isfinite(avg_width)) else np.nan
        support_down = one_min_poc - 0.382 * avg_width if (np.isfinite(one_min_poc) and np.isfinite(avg_width)) else np.nan
        st_mid = (st_vah + st_val) / 2 if np.isfinite(st_vah) and np.isfinite(st_val) else np.nan
        lt_mid = (lt_vah + lt_val) / 2 if np.isfinite(lt_vah) and np.isfinite(lt_val) else np.nan

        return {
            "Target Open": target_open,
            "Support/Down": support_down,
            "ST Avg": st_poc,
            "ST POC": st_poc,
            "ST VAH": st_vah,
            "ST VAL": st_val,
            "LT POC": lt_poc,
            "LT VAL": lt_val,
            "LT VAH Avg": lt_vah,
            "ST MID": st_mid,
            "LT MID": lt_mid,
            "Average Profile Width": avg_width,
        }
    except Exception as exc:
        FALCON_ERRORS.append(f"Profile summary: {exc}")
        return {
            "Target Open": np.nan, "Support/Down": np.nan,
            "ST Avg": np.nan, "ST POC": np.nan, "ST VAH": np.nan, "ST VAL": np.nan,
            "LT POC": np.nan, "LT VAL": np.nan, "LT VAH Avg": np.nan,
            "ST MID": np.nan, "LT MID": np.nan, "Average Profile Width": np.nan,
        }


def percentage_shift(short_value, long_value):
    short_value = safe_float(short_value)
    long_value = safe_float(long_value)
    if not np.isfinite(short_value) or not np.isfinite(long_value) or long_value == 0:
        return np.nan
    return ((short_value - long_value) / abs(long_value)) * 100


def calculate_value_overlap(st_vah, st_val, lt_vah, lt_val):
    values = [safe_float(st_vah), safe_float(st_val), safe_float(lt_vah), safe_float(lt_val)]
    if not all(np.isfinite(x) for x in values):
        return np.nan
    st_high, st_low, lt_high, lt_low = values
    intersection = max(0, min(st_high, lt_high) - max(st_low, lt_low))
    union = max(st_high, lt_high) - min(st_low, lt_low)
    if union <= 0:
        return np.nan
    return intersection / union * 100


def calculate_falcon_correlated_trend(summary):
    st_mid = summary.get("ST MID", np.nan)
    lt_mid = summary.get("LT MID", np.nan)
    st_vah = summary.get("ST VAH", np.nan)
    lt_vah = summary.get("LT VAH Avg", np.nan)
    st_val = summary.get("ST VAL", np.nan)
    lt_val = summary.get("LT VAL", np.nan)
    st_poc = summary.get("ST POC", np.nan)
    lt_poc = summary.get("LT POC", np.nan)

    directional_values = {
        "MID": percentage_shift(st_mid, lt_mid),
        "VAH": percentage_shift(st_vah, lt_vah),
        "VAL": percentage_shift(st_val, lt_val),
        "POC": percentage_shift(st_poc, lt_poc),
    }

    weighted_scores = []
    used_weights = []
    for name, value in directional_values.items():
        if np.isfinite(value):
            normalized = np.tanh(value / DIRECTION_SCALE)
            weighted_scores.append(normalized * TREND_WEIGHTS[name])
            used_weights.append(TREND_WEIGHTS[name])

    if not used_weights:
        return np.nan

    directional_score = sum(weighted_scores) / sum(used_weights)
    logic_gap = (st_vah - lt_val) / st_vah * 100 if (np.isfinite(st_vah) and np.isfinite(lt_val) and st_vah != 0) else np.nan
    overlap_pct = calculate_value_overlap(st_vah, st_val, lt_vah, lt_val)
    gap_confidence = 1 / (1 + abs(logic_gap) / GAP_SCALE) if np.isfinite(logic_gap) else 0.5

    if np.isfinite(overlap_pct):
        overlap_fraction = np.clip(overlap_pct / 100, 0, 1)
        overlap_confidence = np.sqrt(1 - overlap_fraction)
    else:
        overlap_confidence = 0.5

    final_score = directional_score * gap_confidence * overlap_confidence * 100
    return float(np.clip(final_score, -100, 100))


def classify_falcon_trend(score):
    score = safe_float(score)
    if not np.isfinite(score):
        return "INSUFFICIENT PROFILE DATA"
    if score >= 35:
        return "STRONG BULLISH"
    if score >= 12:
        return "BULLISH"
    if score <= -35:
        return "STRONG BEARISH"
    if score <= -12:
        return "BEARISH"
    return "NEUTRAL / MIXED"


def calculate_rejection_levels(row: pd.Series) -> tuple[float, float]:
    r1 = float(row["R1"])
    r2 = float(row["R2"])
    s1 = float(row["S1"])
    t_open = float(row["Target_Open"])
    supp = float(row["Support_Down"])
    st_avg = float(row["ST_Avg"])
    st_vah = float(row["ST_VAH"])
    lt_val = float(row["LT_VAL"])
    lt_vah = float(row["LT_VAH_Avg"])
    qty = float(row["Max_Qty"])

    if lt_val > r2 or lt_val > t_open:
        lower_raw = t_open * 1.0035
    elif t_open > r2:
        if lt_vah > r2:
            lower_raw = ((r1 + r2) / 2) * 0.9983
        elif (r2 - s1) / s1 < 0.04 and qty > 400:
            lower_raw = lt_vah * 0.9987
        else:
            if qty >= 200:
                lower_raw = s1 * 1.0020
            elif s1 > lt_vah:
                lower_raw = (s1 + supp) / 2
            else:
                lower_raw = s1
    elif t_open < r1:
        if lt_vah < r1:
            lower_raw = (lt_vah + supp) / 2
        else:
            lower_raw = st_avg * 1.0008
    else:
        if qty >= 500:
            lower_raw = st_vah * 0.9965
        else:
            lower_raw = (st_avg + s1) / 2

    base_upper = (t_open + st_avg) / 2
    if lt_val > t_open:
        upper_val = (t_open + lt_val) / 2
    else:
        upper_val = base_upper

    return round(lower_raw, 2), round(upper_val, 2)


def build_seven_checks_from_levels(
    *,
    lower_rej: float,
    upper_rej: float,
    s1: float,
    r1: float,
    support_down: float,
    target_open: float,
    st_avg: float,
    st_vah: float,
    lt_vah_avg: float,
    live_chart_zone_and_reversal_candle: bool,
) -> dict[str, bool]:
    c1_s1_lr_confluence = (s1 > 0) and (0.0 <= ((lower_rej - s1) / s1) * 100 <= 0.35)
    range_pct = ((upper_rej - lower_rej) / lower_rej) * 100 if lower_rej > 0 else 0.0
    c2_range_width_2_8_to_3_5_pct = 2.80 <= range_pct <= 3.50
    c3_st_avg_gt_lt_vah_and_ur_lt_r1 = (st_avg > lt_vah_avg) and (upper_rej < r1)
    c4_support_down_buffer = (support_down > 0) and (support_down <= lower_rej * 1.01)
    c5_target_open_vs_upper_rej = target_open >= upper_rej
    c6_value_area_ladder = (st_avg >= lt_vah_avg) and (st_vah > 0)
    c7_live_chart_zone_and_reversal_candle = bool(live_chart_zone_and_reversal_candle)

    return {
        "C1_S1_LR_Confluence": c1_s1_lr_confluence,
        "C2_Range_Width_2_8_to_3_5_Pct": c2_range_width_2_8_to_3_5_pct,
        "C3_ST_Avg_GT_LT_VAH_And_UR_LT_R1": c3_st_avg_gt_lt_vah_and_ur_lt_r1,
        "C4_Support_Down_Buffer": c4_support_down_buffer,
        "C5_Target_Open_VS_Upper_Rej": c5_target_open_vs_upper_rej,
        "C6_Value_Area_Ladder": c6_value_area_ladder,
        "C7_Live_Chart_Zone_And_Reversal_Candle": c7_live_chart_zone_and_reversal_candle,
    }


def evaluate_seven_checks(checks: dict[str, bool]) -> dict[str, object]:
    total_checks = len(checks)
    if total_checks == 0:
        raise ValueError("checks खाली नहीं हो सकता।")
    passed_checks_list = [k for k, v in checks.items() if bool(v)]
    failed_checks_list = [k for k, v in checks.items() if not bool(v)]
    passed_count = len(passed_checks_list)
    percentage = round((passed_count / total_checks) * 100, 2)
    return {
        "passed_checks": f"{passed_count}/{total_checks}",
        "passed_count": passed_count,
        "total_checks": total_checks,
        "percentage": percentage,
        "passed_list": passed_checks_list,
        "failed_list": failed_checks_list,
    }


def process_falcon_scanner_table(df: pd.DataFrame, live_confirmed_col: str = "C7_Confirmed") -> pd.DataFrame:
    out = df.copy()
    lower_list, upper_list, passed_str_list, pct_list, passed_cnt_list = [], [], [], [], []

    for _, row in out.iterrows():
        formula_row = pd.Series({
            "R1": row["R1"],
            "R2": row["R2"],
            "S1": row["S1"],
            "Target_Open": row["Target Open"],
            "Support_Down": row["Support/Down"],
            "ST_Avg": row["ST Avg"],
            "ST_VAH": row["ST VAH"],
            "LT_VAL": row["LT VAL"],
            "LT_VAH_Avg": row["LT VAH Avg"],
            "Max_Qty": row["Max Qty"],
        })
        lr, ur = calculate_rejection_levels(formula_row)
        c7_value = row[live_confirmed_col] if live_confirmed_col in out.columns else False
        c7_flag = False if pd.isna(c7_value) else bool(c7_value)

        checks = build_seven_checks_from_levels(
            lower_rej=lr,
            upper_rej=ur,
            s1=float(row["S1"]),
            r1=float(row["R1"]),
            support_down=float(row["Support/Down"]),
            target_open=float(row["Target Open"]),
            st_avg=float(row["ST Avg"]),
            st_vah=float(row["ST VAH"]),
            lt_vah_avg=float(row["LT VAH Avg"]),
            live_chart_zone_and_reversal_candle=c7_flag,
        )
        evaluation = evaluate_seven_checks(checks)
        lower_list.append(lr)
        upper_list.append(ur)
        passed_str_list.append(evaluation["passed_checks"])
        pct_list.append(evaluation["percentage"])
        passed_cnt_list.append(evaluation["passed_count"])

    out["Lower Rejection"] = lower_list
    out["Upper Rejection"] = upper_list
    out["7 Checks Passed"] = passed_str_list
    out["7 Checks Pass %"] = pct_list
    out["_passed_count"] = passed_cnt_list
    return out


def nearest_pivot_column(ref_price, r1, r2, s1, s2):
    p = safe_float(ref_price)
    if not np.isfinite(p):
        return None
    candidates = {
        "R1": safe_float(r1),
        "R2": safe_float(r2),
        "S1": safe_float(s1),
        "S2": safe_float(s2),
    }
    valid = {k: abs(p - v) for k, v in candidates.items() if np.isfinite(v)}
    if not valid:
        return None
    return min(valid, key=valid.get)


def build_stock_item_from_bundle(stock_name, symbol, timeframes, matched_open, nse_stats):
    result = {
        "Stock": stock_name,
        "Internal Symbol": symbol,
        "Internal Status": "PROCESSING",
    }

    try:
        daily = timeframes.get("Day", empty_ohlcv())
        if daily.empty or len(daily) < 5:
            result["Internal Status"] = "INSUFFICIENT DAILY DATA"
            return result

        levels = calculate_levels(daily)
        probabilities, winner = classify_behaviour(daily)

        behaviour_map = {
            "Smooth Bullish": "Bullish",
            "Smooth Bearish": "Bearish",
            "Zigzag / Choppy": "Choppy",
            "Reversal": "Reversal",
            "Sideways": "Sideways",
        }

        profiles = timeframe_summary(timeframes)
        profile_summary = calculate_profile_summary(profiles)
        falcon_trend = calculate_falcon_correlated_trend(profile_summary)
        falcon_trend_bias = classify_falcon_trend(falcon_trend)
        profile_checks = verify_profile_values(timeframes, profiles)
        historical_report = historical_behaviour_report(timeframes.get("5m", empty_ohlcv()))
        daily_indicators = calculate_indicators(daily)

        if daily_indicators.empty:
            result["Internal Status"] = "INDICATOR ERROR"
            return result

        last_daily = daily_indicators.iloc[-1]
        prev_close = safe_float(levels.get("Previous Close"))
        after_open = is_after_market_open_ist()
        ref_price = matched_open if (after_open and np.isfinite(matched_open) and matched_open > 0) else prev_close
        nearest_piv = nearest_pivot_column(
            ref_price,
            levels.get("R1"),
            levels.get("R2"),
            levels.get("S1"),
            levels.get("S2"),
        )

        result.update(levels)
        result.update(profile_summary)
        result.update(nse_stats)

        result.update({
            "Internal Status": "VALID",
            "Matched_Open_Price": matched_open,
            "Opening_Price_Source": NSE_OPEN_SOURCE_MAP.get(str(symbol).upper().replace(".NS", ""), "UNAVAILABLE"),
            "Display_Prev_Or_Open": ref_price,
            "Nearest_Pivot_Col": nearest_piv,
            "Highest Behaviour Name": behaviour_map.get(winner, winner),
            "Highest Behaviour Full Name": winner,
            "Highest Behaviour Heuristic Share %": probabilities.get(winner, np.nan),
            "All Behaviour Heuristic Shares": probabilities,
            "FALCON_CORRELATED_TREND_%": falcon_trend,
            "FALCON TREND BIAS": falcon_trend_bias,
            "Daily RSI": safe_float(last_daily["RSI"]),
            "Daily ADX": safe_float(last_daily["ADX"]),
            "Daily EMA9": safe_float(last_daily["EMA9"]),
            "Daily EMA21": safe_float(last_daily["EMA21"]),
            "Daily ATR": safe_float(last_daily["ATR"]),
            "Historical Behaviour": historical_report,
            "MTF Profiles": profiles,
            "PROFILE CHECKS": profile_checks,
            "PROFILE CHECK STATUS": (
                "PASS"
                if (not profile_checks.empty and (profile_checks["Check"] == "PASS").all())
                else "CHECK REQUIRED"
            ),
        })

        for timeframe, frame in timeframes.items():
            result[f"{timeframe} Available Rows"] = len(frame) if frame is not None else 0
        result["Daily Available Rows"] = len(daily)
        return result

    except Exception as exc:
        FALCON_ERRORS.append(f"{stock_name} analysis: {exc}")
        result["Internal Status"] = "ANALYSIS ERROR"
        return result


# =============================================================================
# MAIN EXECUTION PIPELINE (SHUFFLED + REJECTION ZONE GLOW MANDATORY)
# =============================================================================
all_nse_symbols = get_live_nse_symbols_step1()

nifty_raw_1y = clean_dataframe(yf.download("^NSEI", period="1y", interval="1d", progress=False))
nifty_ret = nifty_raw_1y["Close"].pct_change().dropna() if not nifty_raw_1y.empty else pd.Series(dtype=float)

today_ist = now_ist().date()
completed_nifty_dates = sorted({d for d in nifty_raw_1y.index.date if d < today_ist}) if not nifty_raw_1y.empty else []

preload_recent_bhavcopies(completed_nifty_dates)
preload_nse_preopen_once()

candidates = run_step1_candidate_pool(all_nse_symbols, nifty_ret, relaxed=False)
if len(candidates) < 55:
    existing_stocks = {c["Stock"] for c in candidates}
    extra_cands = run_step1_candidate_pool(all_nse_symbols, nifty_ret, relaxed=True)
    for ec in extra_cands:
        if ec["Stock"] not in existing_stocks:
            candidates.append(ec)
            existing_stocks.add(ec["Stock"])

RNG.shuffle(candidates)

filtered_cands = []
for c in candidates:
    sym = c["Symbol"]
    n_stats = get_nse_stock_statistics(sym, completed_nifty_dates[-5:])
    mq = safe_float(n_stats.get("MAX SINGLE-ORDER QTY"))
    if np.isfinite(mq) and mq >= MIN_MAX_QTY_FALLBACK:
        c["_nse_stats"] = n_stats
        c["_max_qty"] = mq
        filtered_cands.append(c)

if len(filtered_cands) < 35:
    seen_s = {c["Stock"] for c in filtered_cands}
    rem_syms = [s for s in all_nse_symbols if s.replace(".NS", "") not in seen_s]
    RNG.shuffle(rem_syms)
    for idx in range(0, min(len(rem_syms), 175), 35):
        chunk = rem_syms[idx:idx + 35]
        b_map = fetch_batch_1y_daily(chunk)
        for s in chunk:
            st_n = s.replace(".NS", "")
            if st_n in seen_s:
                continue
            df_d = b_map.get(s)
            if df_d is not None and not df_d.empty:
                lc = float(df_d["Close"].iloc[-1])
                if STEP1_CONFIG["MIN_PRICE"] <= lc <= STEP1_CONFIG["MAX_PRICE"]:
                    n_stats = get_nse_stock_statistics(s, completed_nifty_dates[-5:])
                    mq = safe_float(n_stats.get("MAX SINGLE-ORDER QTY"))
                    if np.isfinite(mq) and mq >= MIN_MAX_QTY_FALLBACK:
                        seen_s.add(st_n)
                        filtered_cands.append({
                            "Stock": st_n,
                            "Symbol": s,
                            "Direction": "BULLISH" if lc >= float(df_d["Open"].iloc[-1]) else "BEARISH",
                            "Abs_Conviction": 15.0,
                            "_nse_stats": n_stats,
                            "_max_qty": mq,
                            "_step1_meta": {},
                        })

RNG.shuffle(filtered_cands)
filtered_cands.sort(
    key=lambda x: (
        x["_max_qty"] >= MIN_MAX_QTY_STRICT,
        safe_float(x.get("Abs_Conviction"), -np.inf),
        safe_float(x.get("_max_qty"), -np.inf),
        str(x.get("Stock", "")),
    ),
    reverse=True,
)
pool_to_evaluate = filtered_cands[:42]

valid_glow_items = []
for i in range(0, len(pool_to_evaluate), 14):
    sub_cands = pool_to_evaluate[i:i + 14]
    sub_syms = [c["Symbol"] for c in sub_cands]
    bundles = fetch_multitf_batch(sub_syms)

    for c in sub_cands:
        sym = c["Symbol"]
        frames, m_open = bundles.get(sym, ({}, np.nan))
        item = build_stock_item_from_bundle(c["Stock"], sym, frames, m_open, c["_nse_stats"])
        if not item or item.get("Internal Status") != "VALID":
            continue

        # Hard data-integrity gate: every MTF profile must have real OHLCV-derived
        # POC/VAH/VAL. Missing timeframe data is NO TRADE, never silently averaged.
        mtf_profiles = item.get("MTF Profiles", {})
        required_timeframes = ["1m", "3m", "5m", "15m", "1H", "4H", "Day"]
        profiles_complete = all(
            isinstance(mtf_profiles.get(tf), dict)
            and mtf_profiles[tf].get("Profile Status") == "VALID"
            and all(np.isfinite(safe_float(mtf_profiles[tf].get(k))) for k in ["POC", "VAH", "VAL"])
            for tf in required_timeframes
        )
        if not profiles_complete:
            continue

        profile_check_df = item.get("PROFILE CHECKS")
        if profile_check_df is None or profile_check_df.empty or not (profile_check_df["Check"] == "PASS").all():
            continue

        req_cols = ["R1", "R2", "S1", "S2", "MAX SINGLE-ORDER QTY", "Target Open", "Support/Down", "ST Avg", "ST VAH", "LT VAL", "LT VAH Avg"]
        if not all(np.isfinite(safe_float(item.get(col))) for col in req_cols if col != "S2"):
            continue

        temp_df = pd.DataFrame([{
            "Stock": item["Stock"],
            "R1": item["R1"],
            "R2": item["R2"],
            "S1": item["S1"],
            "S2": item["S2"],
            "Max Qty": item["MAX SINGLE-ORDER QTY"],
            "Target Open": item["Target Open"],
            "Support/Down": item["Support/Down"],
            "ST Avg": item["ST Avg"],
            "ST VAH": item["ST VAH"],
            "LT VAL": item["LT VAL"],
            "LT VAH Avg": item["LT VAH Avg"],
        }])

        proc = process_falcon_scanner_table(temp_df).iloc[0]
        p_cnt = int(proc["_passed_count"])
        mq_val = float(c["_max_qty"])
        lr_val = float(proc["Lower Rejection"])
        ur_val = float(proc["Upper Rejection"])

        item["_step1_dir"] = c["Direction"]
        ud_bias, ud_score = evaluate_ten_parameter_ud(item, step1_meta=c.get("_step1_meta", {}))

        glow_lower, glow_upper, zone_corr_score = evaluate_rejection_zones_with_correlation(
            frames.get("Day", empty_ohlcv()),
            lr_val,
            ur_val,
            item.get("Daily ATR"),
            ud_bias,
            ud_score,
            step1_meta=c.get("_step1_meta", {}),
        )

        # Exclude share if NEITHER Lower Rejection NOR Upper Rejection qualifies to glow
        if not (glow_lower or glow_upper):
            continue

        item["Lower Rejection"] = lr_val
        item["Upper Rejection"] = ur_val
        item["7 Checks Passed"] = proc["7 Checks Passed"]
        item["7 Checks Pass %"] = proc["7 Checks Pass %"]
        item["_passed_count"] = p_cnt
        item["_max_qty"] = mq_val
        item["_step1_conv"] = float(c["Abs_Conviction"])
        item["_ud_bias"] = ud_bias
        item["_ud_score"] = ud_score
        item["_glow_lower"] = glow_lower
        item["_glow_upper"] = glow_upper
        item["_zone_corr_score"] = zone_corr_score
        item["_is_strict"] = (p_cnt >= STRICT_MIN_CHECKS and mq_val >= MIN_MAX_QTY_STRICT)
        valid_glow_items.append(item)

    if len(valid_glow_items) >= TARGET_FINAL_STOCKS * 2:
        break

RNG.shuffle(valid_glow_items)
valid_glow_items.sort(
    key=lambda x: (
        x["_is_strict"],
        x["_passed_count"] >= FALLBACK_MIN_CHECKS,
        x["_zone_corr_score"],
        abs(x["_ud_score"]),
        x["_max_qty"],
    ),
    reverse=True,
)
selected_rows = list(valid_glow_items[:TARGET_FINAL_STOCKS])

DIRECTION_BY_STOCK = {r["Stock"]: r["_ud_bias"] for r in selected_rows}
GLOW_LOWER_BY_STOCK = {r["Stock"]: bool(r["_glow_lower"]) for r in selected_rows}
GLOW_UPPER_BY_STOCK = {r["Stock"]: bool(r["_glow_upper"]) for r in selected_rows}
STOCKS = {r["Stock"]: r["Internal Symbol"] for r in selected_rows}
FALCON_RESULTS = pd.DataFrame(selected_rows)


# =============================================================================
# DASHBOARD PREPARATION ("U/D" HEADER + PREV/OPEN + SUBTLE ~10% NUMERIC GLOW)
# =============================================================================
TRIANGLE_COL = "U/D"
PREV_OPEN_COL = "Open / Prev" if is_after_market_open_ist() else "Prev Close"

DISPLAY_COLUMNS = [
    "Stock",
    TRIANGLE_COL,
    PREV_OPEN_COL,
    "R1",
    "R2",
    "S1",
    "S2",
    "Lower Rejection",
    "Upper Rejection",
    "Max Qty",
    "Target Open",
    "Support/Down",
    "ST Avg",
    "ST VAH",
    "LT VAL",
    "LT VAH Avg",
]


def triangle_symbol_for_stock(stock_label):
    bias = str(DIRECTION_BY_STOCK.get(stock_label, "")).upper()
    if bias == "BULLISH":
        return "▲"
    if bias == "BEARISH":
        return "▼"
    return "—"


if not FALCON_RESULTS.empty:
    dashboard = pd.DataFrame(index=FALCON_RESULTS.index)
    dashboard["Stock"] = FALCON_RESULTS["Stock"].fillna("N/A").astype(str)
    dashboard[TRIANGLE_COL] = dashboard["Stock"].map(triangle_symbol_for_stock)
    dashboard[PREV_OPEN_COL] = FALCON_RESULTS["Display_Prev_Or_Open"]

    for col in ["R1", "R2", "S1", "S2"]:
        dashboard[col] = FALCON_RESULTS[col]

    dashboard["Lower Rejection"] = FALCON_RESULTS["Lower Rejection"]
    dashboard["Upper Rejection"] = FALCON_RESULTS["Upper Rejection"]
    dashboard["Max Qty"] = FALCON_RESULTS["MAX SINGLE-ORDER QTY"]
    dashboard["Target Open"] = FALCON_RESULTS["Target Open"]
    dashboard["Support/Down"] = FALCON_RESULTS["Support/Down"]
    dashboard["ST Avg"] = FALCON_RESULTS["ST Avg"]
    dashboard["ST VAH"] = FALCON_RESULTS["ST VAH"]
    dashboard["LT VAL"] = FALCON_RESULTS["LT VAL"]
    dashboard["LT VAH Avg"] = FALCON_RESULTS["LT VAH Avg"]

    dashboard = dashboard[DISPLAY_COLUMNS].reset_index(drop=True)
else:
    dashboard = pd.DataFrame(columns=DISPLAY_COLUMNS)

NEAREST_PIVOT_BY_STOCK = {
    row["Stock"]: row.get("Nearest_Pivot_Col")
    for _, row in FALCON_RESULTS.iterrows()
} if not FALCON_RESULTS.empty else {}


# =============================================================================
# SUBTLE ~10% GLOW ON NUMERIC VALUE ONLY (NEVER ENTIRE CELL BOX)
# =============================================================================
PRICE_COLUMNS = [
    PREV_OPEN_COL,
    "R1", "R2", "S1", "S2",
    "Lower Rejection", "Upper Rejection",
    "Target Open", "Support/Down",
    "ST Avg", "ST VAH", "LT VAL",
    "LT VAH Avg",
]


def make_price_formatter(col_name):
    def _fmt(value):
        number = safe_float(value)
        if not np.isfinite(number):
            return "—"
        formatted = f"{number:,.2f}"
        if col_name in ["R1", "R2", "S1", "S2", "Lower Rejection", "Upper Rejection"]:
            return f'<span class="falcon-num-token" data-col="{col_name}">{formatted}</span>'
        return formatted
    return _fmt


def format_quantity(value):
    number = safe_float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:,.0f}"


formatters = {col: make_price_formatter(col) for col in PRICE_COLUMNS}
formatters["Max Qty"] = format_quantity


def style_dashboard_cells(df_disp):
    styles = pd.DataFrame("", index=df_disp.index, columns=df_disp.columns)

    for idx, r in df_disp.iterrows():
        stock_name = str(r["Stock"])
        bias = str(DIRECTION_BY_STOCK.get(stock_name, "")).upper()

        styles.loc[idx, "Stock"] = (
            "background-color:#ffffff !important;"
            "color:#202a35 !important;"
            "font-weight:700 !important;"
        )

        if bias == "BULLISH":
            styles.loc[idx, TRIANGLE_COL] = (
                "color:#075b20 !important;"
                "font-size:15px !important;"
                "font-weight:800 !important;"
                "text-align:center !important;"
            )
        elif bias == "BEARISH":
            styles.loc[idx, TRIANGLE_COL] = (
                "color:#a51f1f !important;"
                "font-size:15px !important;"
                "font-weight:800 !important;"
                "text-align:center !important;"
            )
        else:
            styles.loc[idx, TRIANGLE_COL] = (
                "color:#202a35 !important;"
                "font-size:15px !important;"
                "font-weight:700 !important;"
                "text-align:center !important;"
            )

    return styles


column_widths = {
    "Stock": "7.5%",
    TRIANGLE_COL: "3.4%",
    PREV_OPEN_COL: "6.8%",
    "R1": "5.2%",
    "R2": "5.2%",
    "S1": "5.2%",
    "S2": "5.2%",
    "Lower Rejection": "7.4%",
    "Upper Rejection": "7.4%",
    "Max Qty": "5.8%",
    "Target Open": "6.8%",
    "Support/Down": "6.8%",
    "ST Avg": "6.0%",
    "ST VAH": "6.0%",
    "LT VAL": "6.0%",
    "LT VAH Avg": "6.3%",
}

table_styles = [
    {
        "selector": "table",
        "props": [
            ("width", "100%"),
            ("table-layout", "fixed"),
            ("border-collapse", "collapse"),
            ("border-spacing", "0"),
            ("margin", "0"),
            ("font-family", "Arial, sans-serif"),
            ("font-size", "11px"),
            ("line-height", "1.2"),
            ("background-color", "#ffffff"),
        ],
    },
    {
        "selector": "thead th",
        "props": [
            ("background-color", "#172b4d"),
            ("color", "#ffffff"),
            ("font-size", "10.5px"),
            ("font-weight", "700"),
            ("text-align", "center"),
            ("vertical-align", "middle"),
            ("white-space", "normal"),
            ("overflow-wrap", "anywhere"),
            ("padding", "6px 2px"),
            ("border", "1px solid #cbd3df"),
            ("line-height", "1.15"),
        ],
    },
    {
        "selector": "tbody td",
        "props": [
            ("font-size", "11.5px"),
            ("font-weight", "600"),
            ("color", "#202a35"),
            ("text-align", "center"),
            ("vertical-align", "middle"),
            ("white-space", "nowrap"),
            ("padding", "5px 2px"),
            ("height", "30px"),
            ("border", "1px solid #d9dee7"),
            ("line-height", "1.2"),
            ("background-color", "#ffffff"),
        ],
    },
    {
        "selector": "tbody td.col0",
        "props": [
            ("font-weight", "700"),
            ("font-size", "11.5px"),
            ("text-align", "left"),
            ("padding-left", "6px"),
        ],
    },
    {
        "selector": "tbody td.col1",
        "props": [
            ("font-size", "15px"),
            ("text-align", "center"),
            ("padding", "5px 0px"),
        ],
    },
    {
        "selector": "tbody tr:hover td",
        "props": [
            ("border-top", "1px solid #8094b0"),
            ("border-bottom", "1px solid #8094b0"),
        ],
    },
]

for index, column in enumerate(DISPLAY_COLUMNS):
    table_styles.append({
        "selector": f"th.col_heading.col{index}, td.col{index}",
        "props": [
            ("width", column_widths[column]),
            ("max-width", column_widths[column]),
        ],
    })


FALCON_DASHBOARD = (
    dashboard.style
    .hide(axis="index")
    .format(formatters, na_rep="—")
    .apply(style_dashboard_cells, axis=None)
    .set_table_attributes('class="falcon-single-table"')
    .set_table_styles(table_styles, overwrite=False)
)

rendered_html = FALCON_DASHBOARD.to_html()

# Ensure "U/D" header displays cleanly
rendered_html = re.sub(
    r'(<th\b[^>]*class="[^"]*\bcol1\b[^"]*"[^>]*>).*?(</th>)',
    r'\1U/D\2',
    rendered_html,
    count=1,
    flags=re.DOTALL,
)

# Subtle ~10% numeric-only glow styles
GLOW_YELLOW_STYLE = (
    "color:#b45309;font-weight:700;"
    "text-shadow:0 0 1.2px rgba(245,158,11,0.12);"
)
GLOW_GREEN_STYLE = (
    "color:#075b20;font-weight:700;"
    "text-shadow:0 0 1.2px rgba(16,185,129,0.12);"
)
GLOW_RED_STYLE = (
    "color:#a51f1f;font-weight:700;"
    "text-shadow:0 0 1.2px rgba(239,68,68,0.12);"
)

for _, r in dashboard.iterrows():
    st_name = str(r["Stock"])

    nearest_col = NEAREST_PIVOT_BY_STOCK.get(st_name)
    if nearest_col in ["R1", "R2", "S1", "S2"]:
        val_num = safe_float(r[nearest_col])
        if np.isfinite(val_num):
            src_span = f'<span class="falcon-num-token" data-col="{nearest_col}">{val_num:,.2f}</span>'
            dst_span = f'<span style="{GLOW_YELLOW_STYLE}">{val_num:,.2f}</span>'
            rendered_html = rendered_html.replace(src_span, dst_span, 1)

    if GLOW_LOWER_BY_STOCK.get(st_name, False):
        lr_num = safe_float(r["Lower Rejection"])
        if np.isfinite(lr_num):
            src_lr = f'<span class="falcon-num-token" data-col="Lower Rejection">{lr_num:,.2f}</span>'
            dst_lr = f'<span style="{GLOW_GREEN_STYLE}">{lr_num:,.2f}</span>'
            rendered_html = rendered_html.replace(src_lr, dst_lr, 1)

    if GLOW_UPPER_BY_STOCK.get(st_name, False):
        ur_num = safe_float(r["Upper Rejection"])
        if np.isfinite(ur_num):
            src_ur = f'<span class="falcon-num-token" data-col="Upper Rejection">{ur_num:,.2f}</span>'
            dst_ur = f'<span style="{GLOW_RED_STYLE}">{ur_num:,.2f}</span>'
            rendered_html = rendered_html.replace(src_ur, dst_ur, 1)

for col_tk in ["R1", "R2", "S1", "S2", "Lower Rejection", "Upper Rejection"]:
    rendered_html = re.sub(
        rf'<span class="falcon-num-token" data-col="{col_tk}">(.*?)</span>',
        r'\1',
        rendered_html,
    )


# =============================================================================
# V5.5 VALIDATION DISCLOSURE
# =============================================================================
print("\nV5.5 DATA-INTEGRITY RULES: real 3m bars from 1m; 4H aligned to 09:15; ")
print("all 7 timeframe profiles required; deterministic candidate ranking.")
print("Behaviour shares are heuristic score distributions, NOT calibrated probabilities.")
print("No 70% out-of-sample accuracy claim is made: run a dated walk-forward test before live use.")
print(f"Selected candidates: {len(selected_rows)} / {TARGET_FINAL_STOCKS}")
