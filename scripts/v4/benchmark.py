"""
Benchmark rapide — teste chaque amélioration v4 isolément
pour identifier celles qui améliorent réellement le rendement.
"""

import logging
import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import pytz
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

SYMBOLS = {"NAS100": "NAS100.", "US30": "DJ30.", "SPX500": "SP500."}
TIMEFRAME = mt5.TIMEFRAME_M15
H1_TIMEFRAME = mt5.TIMEFRAME_H1
PARIS_TZ = pytz.timezone("Europe/Paris")

BB_LENGTH = 20; BB_STD = 2.0
VOL_SMA_LENGTH = 20; VOL_MULTIPLIER = 1.3
EMA_TREND_LENGTH = 50; H1_EMA_LENGTH = 50
RSI_LENGTH = 14; ATR_LENGTH = 14; ADX_LENGTH = 14

RISK_REWARD = 3
ATR_SL_MULT = 1.5
ATR_TP_MULT = ATR_SL_MULT * RISK_REWARD
ATR_MEDIAN_WINDOW = 100; ATR_CAP_MULT = 1.5
RISK_PCT = 0.02; BACKTEST_MONTHS = 8
INITIAL_CAPITAL = 132

MT5_LOGIN = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER = os.environ["MT5_SERVER"]

logging.basicConfig(level=logging.WARNING)


def fetch(symbol, months, tf):
    utc_to = datetime.now(tz=pytz.utc)
    utc_from = utc_to - timedelta(days=months * 30)
    rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_paris"] = df["time"].dt.tz_convert(PARIS_TZ)
    return df


def indicators(df):
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=BB_STD)
    col_map = {}
    for c in bb.columns:
        if c.startswith("BBU"): col_map[c] = "BBU"
        elif c.startswith("BBL"): col_map[c] = "BBL"
    bb = bb.rename(columns=col_map)
    df = pd.concat([df, bb], axis=1)
    df["vol_sma"] = ta.sma(df["tick_volume"].astype(float), length=VOL_SMA_LENGTH)
    df["ema_trend"] = ta.ema(df["close"], length=EMA_TREND_LENGTH)
    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH)
    atr_med = df["atr"].rolling(ATR_MEDIAN_WINDOW, min_periods=1).median()
    df["atr_capped"] = df["atr"].clip(upper=atr_med * ATR_CAP_MULT)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=ADX_LENGTH)
    df["adx"] = adx_df[f"ADX_{ADX_LENGTH}"]
    return df


def merge_h1(df_m15, df_h1):
    df_h1 = df_h1.copy()
    df_h1["h1_ema50"] = ta.ema(df_h1["close"], length=H1_EMA_LENGTH)
    h1 = df_h1[["time", "close", "h1_ema50"]].copy()
    h1["time_avail"] = h1["time"] + pd.Timedelta(hours=1)
    h1 = h1.rename(columns={"close": "h1_close"}).sort_values("time_avail")
    df_m15 = df_m15.sort_values("time")
    merged = pd.merge_asof(df_m15, h1[["time_avail", "h1_close", "h1_ema50"]],
                           left_on="time", right_on="time_avail", direction="backward")
    return merged.drop(columns=["time_avail"])


def lot_size(equity, atr_c, specs, risk_pct=RISK_PCT, atr_sl_mult=ATR_SL_MULT):
    risk = equity * risk_pct
    sl_dist = atr_sl_mult * atr_c
    sl_val = (sl_dist / specs["tick_size"]) * specs["tick_value"]
    if sl_val <= 0: return specs["vol_min"]
    lot = risk / sl_val
    step = specs["vol_step"]
    lot = round(lot / step) * step
    return max(specs["vol_min"], min(specs["vol_max"], round(lot, 6)))


def is_session(dt_paris, start_h, start_m, end_h, end_m):
    start = dt_paris.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = dt_paris.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= dt_paris <= end


def run(symbols_data, cfg):
    """Run backtest with given config dict."""
    session_start_h = cfg.get("session_start_h", 15)
    session_start_m = cfg.get("session_start_m", 30)
    session_end_h = cfg.get("session_end_h", 22)
    session_end_m = cfg.get("session_end_m", 0)
    use_h1 = cfg.get("use_h1", False)
    adx_threshold = cfg.get("adx_threshold", 0)
    trail_levels = cfg.get("trail_levels", [])  # [(threshold_r, lock_r), ...]
    atr_tp_mult = cfg.get("atr_tp_mult", ATR_TP_MULT)
    atr_sl_mult = cfg.get("atr_sl_mult", ATR_SL_MULT)
    risk_pct = cfg.get("risk_pct", RISK_PCT)
    trail_tight_r = cfg.get("trail_tight_r", atr_tp_mult * 0.5 / atr_sl_mult)  # default = TP/2 en R
    trail_tight_dist = cfg.get("trail_tight_dist", atr_sl_mult)

    equity = INITIAL_CAPITAL
    trades = []
    positions = {}
    pending = {}
    n_bars = min(len(c["df"]) for c in symbols_data.values())
    warmup = 100

    for i in range(warmup, n_bars):
        for symbol, c in symbols_data.items():
            row = c["df"].iloc[i]
            specs = c["specs"]
            point = specs["point"]

            close = row["close"]; op = row["open"]; hi = row["high"]; lo = row["low"]
            bbu = row.get("BBU"); bbl = row.get("BBL")
            tv = row["tick_volume"]; vsma = row.get("vol_sma")
            ema = row.get("ema_trend"); rsi = row.get("rsi")
            atr_c = row.get("atr_capped"); adx_v = row.get("adx")
            h1c = row.get("h1_close"); h1e = row.get("h1_ema50")
            t_paris = row["time_paris"]

            check_vals = [bbu, bbl, vsma, ema, rsi, atr_c, adx_v]
            if use_h1:
                check_vals += [h1c, h1e]
            if any(pd.isna(v) for v in check_vals):
                pending.pop(symbol, None); continue
            if atr_c == 0:
                pending.pop(symbol, None); continue

            # Entry
            if symbol in pending and symbol not in positions:
                sig = pending.pop(symbol)
                ep = op; ac = sig["atr_capped"]
                lt = lot_size(equity, ac, specs, risk_pct, atr_sl_mult)
                if sig["d"] == "long":
                    positions[symbol] = {"d": "long", "ep": ep, "sl": ep - atr_sl_mult*ac,
                        "tp": ep + atr_tp_mult*ac, "et": t_paris, "ae": ac, "bp": ep, "lt": lt}
                else:
                    positions[symbol] = {"d": "short", "ep": ep, "sl": ep + atr_sl_mult*ac,
                        "tp": ep - atr_tp_mult*ac, "et": t_paris, "ae": ac, "bp": ep, "lt": lt}

            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = atr_sl_mult * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                if p["d"] == "long":
                    if lo <= p["sl"]:
                        pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    if hi >= p["tp"]:
                        pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    if hi > p["bp"]: p["bp"] = hi
                    pr = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] - trail_tight_dist * ae
                        if ns > p["sl"]: p["sl"] = ns
                    elif trail_levels:
                        for tr, lr in reversed(trail_levels):
                            if pr >= tr:
                                ns = p["ep"] + lr * one_r
                                if ns > p["sl"]: p["sl"] = ns
                                break
                else:
                    if hi >= p["sl"]:
                        pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    if lo <= p["tp"]:
                        pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    if lo < p["bp"]: p["bp"] = lo
                    pr = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] + trail_tight_dist * ae
                        if ns < p["sl"]: p["sl"] = ns
                    elif trail_levels:
                        for tr, lr in reversed(trail_levels):
                            if pr >= tr:
                                ns = p["ep"] - lr * one_r
                                if ns < p["sl"]: p["sl"] = ns
                                break
                continue

            if not is_session(t_paris, session_start_h, session_start_m, session_end_h, session_end_m):
                continue

            vt = VOL_MULTIPLIER * vsma
            h1_long_ok = (not use_h1) or (h1c > h1e)
            h1_short_ok = (not use_h1) or (h1c < h1e)
            adx_ok = adx_v > adx_threshold

            if close > bbu and close > ema and tv > vt and 50 < rsi < 70 and adx_ok and h1_long_ok:
                pending[symbol] = {"d": "long", "atr_capped": atr_c}
            elif close < bbl and close < ema and tv > vt and 30 < rsi < 50 and adx_ok and h1_short_ok:
                pending[symbol] = {"d": "short", "atr_capped": atr_c}
            else:
                pending.pop(symbol, None)

    # Close remaining
    for symbol, p in positions.items():
        c = symbols_data[symbol]
        specs = c["specs"]; point = specs["point"]; lt = p["lt"]
        upp = (specs["tick_value"] / specs["tick_size"]) * point * lt
        last = c["df"].iloc[-1]
        if p["d"] == "long":
            pnl = round((last["close"] - p["ep"]) / point * upp, 2)
        else:
            pnl = round((p["ep"] - last["close"]) / point * upp, 2)
        equity += pnl; trades.append(pnl)

    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) * 100 if trades else 0
    ret = equity - INITIAL_CAPITAL
    ret_pct = ret / INITIAL_CAPITAL * 100

    peak = INITIAL_CAPITAL; max_dd = 0
    eq = INITIAL_CAPITAL
    for t in trades:
        eq += t
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        "trades": len(trades), "wins": wins, "wr": round(wr, 1),
        "ret": round(ret, 2), "ret_pct": round(ret_pct, 1),
        "equity": round(equity, 2), "max_dd": round(max_dd, 1),
    }


def main():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 error: {mt5.last_error()}")
        return

    symbols_data = {}
    for label, symbol in SYMBOLS.items():
        if not mt5.symbol_select(symbol, True): continue
        df_m15 = fetch(symbol, BACKTEST_MONTHS, TIMEFRAME)
        df_h1 = fetch(symbol, BACKTEST_MONTHS, H1_TIMEFRAME)
        if df_m15 is None or df_h1 is None: continue
        df_m15 = indicators(df_m15)
        df_m15 = merge_h1(df_m15, df_h1)
        info = mt5.symbol_info(symbol)
        specs = {"point": info.point, "tick_value": info.trade_tick_value,
                 "tick_size": info.trade_tick_size, "vol_min": info.volume_min,
                 "vol_max": info.volume_max, "vol_step": info.volume_step}
        symbols_data[symbol] = {"df": df_m15, "specs": specs}

    mt5.shutdown()

    if not symbols_data:
        print("No data"); return

    # ─── TESTS ───
    S17_21 = {"session_start_h": 17, "session_start_m": 0, "session_end_h": 21, "session_end_m": 0}
    SL125 = {"atr_sl_mult": 1.25, "atr_tp_mult": 1.25 * 3}

    tests = {
        # ─── References ───
        "v3 BASE": {
            "session_start_h": 15, "session_start_m": 30,
            "session_end_h": 22, "session_end_m": 0,
        },
        "17h-21h (best session)": {**S17_21},
        "17-21 SL=1.25 (best combo)": {**S17_21, **SL125},

        # ─── SL fine-tuning around 1.25 ───
        "17-21 SL=1.1": {**S17_21, "atr_sl_mult": 1.1, "atr_tp_mult": 1.1 * 3},
        "17-21 SL=1.15": {**S17_21, "atr_sl_mult": 1.15, "atr_tp_mult": 1.15 * 3},
        "17-21 SL=1.2": {**S17_21, "atr_sl_mult": 1.2, "atr_tp_mult": 1.2 * 3},
        "17-21 SL=1.3": {**S17_21, "atr_sl_mult": 1.3, "atr_tp_mult": 1.3 * 3},
        "17-21 SL=1.35": {**S17_21, "atr_sl_mult": 1.35, "atr_tp_mult": 1.35 * 3},

        # ─── SL=1.25 + trailing improvements ───
        "SL1.25 tight_dist=1.0": {**S17_21, **SL125, "trail_tight_dist": 1.0},
        "SL1.25 tight_dist=0.75": {**S17_21, **SL125, "trail_tight_dist": 0.75},
        "SL1.25 tight_dist=0.5": {**S17_21, **SL125, "trail_tight_dist": 0.5},
        "SL1.25 trail_r=1.0": {**S17_21, **SL125, "trail_tight_r": 1.0},
        "SL1.25 trail_r=2.0": {**S17_21, **SL125, "trail_tight_r": 2.0},
        "SL1.25 BE@1R": {**S17_21, **SL125, "trail_levels": [(1.0, 0.0)]},

        # ─── SL=1.25 + risk ───
        "SL1.25 risk=2.5%": {**S17_21, **SL125, "risk_pct": 0.025},
        "SL1.25 risk=3%": {**S17_21, **SL125, "risk_pct": 0.03},

        # ─── SL=1.25 + best combos ───
        "SL1.25+tight0.75+risk2.5%": {**S17_21, **SL125, "trail_tight_dist": 0.75, "risk_pct": 0.025},
        "SL1.25+tight0.75+risk3%": {**S17_21, **SL125, "trail_tight_dist": 0.75, "risk_pct": 0.03},
        "SL1.25+tight1.0+risk2.5%": {**S17_21, **SL125, "trail_tight_dist": 1.0, "risk_pct": 0.025},
        "SL1.25+tight1.0+risk3%": {**S17_21, **SL125, "trail_tight_dist": 1.0, "risk_pct": 0.03},
        "SL1.25+R4:1": {**S17_21, "atr_sl_mult": 1.25, "atr_tp_mult": 1.25 * 4},
        "SL1.25+R4:1+tight0.75": {**S17_21, "atr_sl_mult": 1.25, "atr_tp_mult": 1.25 * 4, "trail_tight_dist": 0.75},

        # ─── Compare: tight_dist alone vs SL=1.25 ───
        "17-21 tight_dist=0.75": {**S17_21, "trail_tight_dist": 0.75},
        "17-21 tight0.75+risk2.5%": {**S17_21, "trail_tight_dist": 0.75, "risk_pct": 0.025},
    }

    print(f"{'Test':<36} | {'Trades':>6} | {'WR':>6} | {'Ret $':>9} | {'Ret %':>8} | {'DD %':>5}")
    print("-" * 85)
    for name, cfg in tests.items():
        r = run(symbols_data, cfg)
        print(f"{name:<36} | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['ret']:>+9.2f} | {r['ret_pct']:>+7.1f}% | {r['max_dd']:>5.1f}%")


if __name__ == "__main__":
    main()
