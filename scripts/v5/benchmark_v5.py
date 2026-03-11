"""
Benchmark v5 — teste de nouvelles améliorations par-dessus la base v4.
Base v4 : SL=1.25 ATR, trail_dist=0.75, session 17h-21h.
"""

import logging
import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import pytz
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

SYMBOLS = {"NAS100": "NAS100.", "US30": "DJ30.", "SPX500": "SP500."}
TIMEFRAME = mt5.TIMEFRAME_M15
PARIS_TZ = pytz.timezone("Europe/Paris")

# ─── Base v4 defaults ───
BB_LENGTH = 20; BB_STD_DEFAULT = 2.0
VOL_SMA_LENGTH = 20; VOL_MULT_DEFAULT = 1.3
EMA_TREND_LENGTH = 50
RSI_LENGTH = 14; ATR_LENGTH = 14

RISK_REWARD = 4  # v5
ATR_SL_MULT = 1.25
ATR_TP_MULT = ATR_SL_MULT * RISK_REWARD  # 5.0
ATR_MEDIAN_WINDOW = 100; ATR_CAP_MULT = 1.5
RISK_PCT = 0.025; BACKTEST_MONTHS = 8  # v5: 2.5%
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


def indicators(df, bb_std=BB_STD_DEFAULT):
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=bb_std)
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
    # ATR percentile (rolling 100 bars)
    df["atr_pct"] = df["atr"].rolling(100, min_periods=50).rank(pct=True)
    # Momentum : close vs close[N]
    df["mom_3"] = df["close"] - df["close"].shift(3)
    df["mom_5"] = df["close"] - df["close"].shift(5)
    return df


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
    session_start_h = cfg.get("session_start_h", 17)
    session_start_m = cfg.get("session_start_m", 0)
    session_end_h = cfg.get("session_end_h", 21)
    session_end_m = cfg.get("session_end_m", 0)

    atr_sl_mult = cfg.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = cfg.get("atr_tp_mult", ATR_TP_MULT)
    risk_pct = cfg.get("risk_pct", RISK_PCT)
    trail_tight_r = cfg.get("trail_tight_r", atr_tp_mult * 0.5 / atr_sl_mult)
    trail_tight_dist = cfg.get("trail_tight_dist", 0.5)  # v5 default
    vol_mult = cfg.get("vol_mult", VOL_MULT_DEFAULT)

    # Filters
    allowed_days = cfg.get("allowed_days", None)  # e.g. [0,1,2,3,4] Mon-Fri
    atr_pct_min = cfg.get("atr_pct_min", 0.0)  # min ATR percentile
    atr_pct_max = cfg.get("atr_pct_max", 1.0)  # max ATR percentile
    mom_filter = cfg.get("mom_filter", 0)  # 0=off, 3=mom_3, 5=mom_5
    rsi_long_range = cfg.get("rsi_long", (50, 70))
    rsi_short_range = cfg.get("rsi_short", (30, 50))

    # Partial TP
    partial_tp_r = cfg.get("partial_tp_r", 0)  # 0=off, e.g. 1.5 = close 50% at 1.5R
    partial_pct = cfg.get("partial_pct", 0.5)  # fraction to close at partial TP

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
            atr_c = row.get("atr_capped")
            atr_pct_val = row.get("atr_pct")
            mom3 = row.get("mom_3"); mom5 = row.get("mom_5")
            t_paris = row["time_paris"]

            check_vals = [bbu, bbl, vsma, ema, rsi, atr_c]
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
                        "tp": ep + atr_tp_mult*ac, "et": t_paris, "ae": ac, "bp": ep,
                        "lt": lt, "lt_orig": lt, "partial_done": False}
                else:
                    positions[symbol] = {"d": "short", "ep": ep, "sl": ep + atr_sl_mult*ac,
                        "tp": ep - atr_tp_mult*ac, "et": t_paris, "ae": ac, "bp": ep,
                        "lt": lt, "lt_orig": lt, "partial_done": False}

            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = atr_sl_mult * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                # Ordre réaliste : TP → update trailing → SL
                if p["d"] == "long":
                    # Partial TP
                    if partial_tp_r > 0 and not p["partial_done"]:
                        partial_level = p["ep"] + partial_tp_r * one_r
                        if hi >= partial_level:
                            closed_lt = round(p["lt_orig"] * partial_pct, 6)
                            pnl_partial = round((partial_level - p["ep"]) / point *
                                (specs["tick_value"] / specs["tick_size"]) * point * closed_lt, 2)
                            equity += pnl_partial
                            p["lt"] -= closed_lt
                            p["partial_done"] = True
                            lt = p["lt"]
                            upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                    # 1. TP check
                    if hi >= p["tp"]:
                        pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    # 2. Update trailing (prix monte au high puis retrace)
                    if hi > p["bp"]: p["bp"] = hi
                    pr = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] - trail_tight_dist * ae
                        if ns > p["sl"]: p["sl"] = ns
                    # 3. SL check avec trailing resserré
                    if lo <= p["sl"]:
                        pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                else:
                    # Partial TP
                    if partial_tp_r > 0 and not p["partial_done"]:
                        partial_level = p["ep"] - partial_tp_r * one_r
                        if lo <= partial_level:
                            closed_lt = round(p["lt_orig"] * partial_pct, 6)
                            pnl_partial = round((p["ep"] - partial_level) / point *
                                (specs["tick_value"] / specs["tick_size"]) * point * closed_lt, 2)
                            equity += pnl_partial
                            p["lt"] -= closed_lt
                            p["partial_done"] = True
                            lt = p["lt"]
                            upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                    # 1. TP check
                    if lo <= p["tp"]:
                        pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                    # 2. Update trailing (prix descend au low puis retrace)
                    if lo < p["bp"]: p["bp"] = lo
                    pr = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] + trail_tight_dist * ae
                        if ns < p["sl"]: p["sl"] = ns
                    # 3. SL check avec trailing resserré
                    if hi >= p["sl"]:
                        pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl); del positions[symbol]; continue
                continue

            if not is_session(t_paris, session_start_h, session_start_m, session_end_h, session_end_m):
                continue

            # Day filter
            if allowed_days is not None and t_paris.weekday() not in allowed_days:
                continue

            # ATR percentile filter
            if not pd.isna(atr_pct_val):
                if atr_pct_val < atr_pct_min or atr_pct_val > atr_pct_max:
                    continue

            vt = vol_mult * vsma

            # Momentum filter
            mom_long_ok = True
            mom_short_ok = True
            if mom_filter == 3 and not pd.isna(mom3):
                mom_long_ok = mom3 > 0
                mom_short_ok = mom3 < 0
            elif mom_filter == 5 and not pd.isna(mom5):
                mom_long_ok = mom5 > 0
                mom_short_ok = mom5 < 0

            rsi_l_min, rsi_l_max = rsi_long_range
            rsi_s_min, rsi_s_max = rsi_short_range

            if (close > bbu and close > ema and tv > vt
                    and rsi_l_min < rsi < rsi_l_max and mom_long_ok):
                pending[symbol] = {"d": "long", "atr_capped": atr_c}
            elif (close < bbl and close < ema and tv > vt
                    and rsi_s_min < rsi < rsi_s_max and mom_short_ok):
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

    # Load data with default BB std
    symbols_data_default = {}
    for label, symbol in SYMBOLS.items():
        if not mt5.symbol_select(symbol, True): continue
        df_m15 = fetch(symbol, BACKTEST_MONTHS, TIMEFRAME)
        if df_m15 is None: continue
        df_m15 = indicators(df_m15, bb_std=BB_STD_DEFAULT)
        info = mt5.symbol_info(symbol)
        specs = {"point": info.point, "tick_value": info.trade_tick_value,
                 "tick_size": info.trade_tick_size, "vol_min": info.volume_min,
                 "vol_max": info.volume_max, "vol_step": info.volume_step}
        symbols_data_default[symbol] = {"df": df_m15, "specs": specs}

    # Load data with BB std=1.5
    symbols_data_bb15 = {}
    for label, symbol in SYMBOLS.items():
        if not mt5.symbol_select(symbol, True): continue
        df_m15 = fetch(symbol, BACKTEST_MONTHS, TIMEFRAME)
        if df_m15 is None: continue
        df_m15 = indicators(df_m15, bb_std=1.5)
        info = mt5.symbol_info(symbol)
        specs = {"point": info.point, "tick_value": info.trade_tick_value,
                 "tick_size": info.trade_tick_size, "vol_min": info.volume_min,
                 "vol_max": info.volume_max, "vol_step": info.volume_step}
        symbols_data_bb15[symbol] = {"df": df_m15, "specs": specs}

    mt5.shutdown()

    if not symbols_data_default:
        print("No data"); return

    # ─── BASE v4 shortcut ───
    BASE = {}  # v4 defaults already in run()

    tests = {
        # ─── Reference ───
        "v4 BASE (reference)": {**BASE},

        # ─── Day of week filters ───
        "No Monday": {**BASE, "allowed_days": [1, 2, 3, 4]},
        "No Friday": {**BASE, "allowed_days": [0, 1, 2, 3]},
        "No Mon+Fri": {**BASE, "allowed_days": [1, 2, 3]},
        "Tue-Thu only": {**BASE, "allowed_days": [1, 2, 3]},
        "Mon-Wed only": {**BASE, "allowed_days": [0, 1, 2]},
        "Wed-Fri only": {**BASE, "allowed_days": [2, 3, 4]},

        # ─── Session micro-optimization ───
        "Session 17:15-20:45": {**BASE, "session_start_h": 17, "session_start_m": 15,
                                 "session_end_h": 20, "session_end_m": 45},
        "Session 17:30-21:00": {**BASE, "session_start_h": 17, "session_start_m": 30},
        "Session 16:30-21:00": {**BASE, "session_start_h": 16, "session_start_m": 30},
        "Session 17:00-20:00": {**BASE, "session_end_h": 20, "session_end_m": 0},
        "Session 17:00-20:30": {**BASE, "session_end_h": 20, "session_end_m": 30},

        # ─── ATR percentile (volatility regime) ───
        "ATR pct > 30%": {**BASE, "atr_pct_min": 0.30},
        "ATR pct > 40%": {**BASE, "atr_pct_min": 0.40},
        "ATR pct > 50%": {**BASE, "atr_pct_min": 0.50},
        "ATR pct 25-85%": {**BASE, "atr_pct_min": 0.25, "atr_pct_max": 0.85},

        # ─── Momentum filter ───
        "Momentum 3 bars": {**BASE, "mom_filter": 3},
        "Momentum 5 bars": {**BASE, "mom_filter": 5},

        # ─── Volume multiplier ───
        "Vol mult 1.5x": {**BASE, "vol_mult": 1.5},
        "Vol mult 1.8x": {**BASE, "vol_mult": 1.8},
        "Vol mult 1.0x": {**BASE, "vol_mult": 1.0},

        # ─── Risk per trade ───
        "Risk 2.5%": {**BASE, "risk_pct": 0.025},
        "Risk 3%": {**BASE, "risk_pct": 0.03},

        # ─── R:R variations ───
        "R:R 3.5:1": {**BASE, "atr_tp_mult": ATR_SL_MULT * 3.5},
        "R:R 4:1": {**BASE, "atr_tp_mult": ATR_SL_MULT * 4},

        # ─── RSI range variations ───
        "RSI long 45-75": {**BASE, "rsi_long": (45, 75), "rsi_short": (25, 55)},
        "RSI long 55-70": {**BASE, "rsi_long": (55, 70), "rsi_short": (30, 45)},

        # ─── Partial TP ───
        "Partial 50% @ 1.5R": {**BASE, "partial_tp_r": 1.5},
        "Partial 50% @ 1.0R": {**BASE, "partial_tp_r": 1.0},
        "Partial 50% @ 2.0R": {**BASE, "partial_tp_r": 2.0},

        # ─── Trail variations ───
        "Trail dist 0.5": {**BASE, "trail_tight_dist": 0.5},
        "Trail dist 1.0": {**BASE, "trail_tight_dist": 1.0},
        "Trail R=1.0": {**BASE, "trail_tight_r": 1.0},
        "Trail R=2.0": {**BASE, "trail_tight_r": 2.0},
    }

    print(f"{'Test':<36} | {'Trades':>6} | {'WR':>6} | {'Ret $':>9} | {'Ret %':>8} | {'DD %':>5}")
    print("-" * 85)

    # Run default BB tests
    for name, cfg in tests.items():
        r = run(symbols_data_default, cfg)
        print(f"{name:<36} | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['ret']:>+9.2f} | {r['ret_pct']:>+7.1f}% | {r['max_dd']:>5.1f}%")

    # BB std=1.5 test
    print("-" * 85)
    r = run(symbols_data_bb15, BASE)
    print(f"{'BB std=1.5':<36} | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['ret']:>+9.2f} | {r['ret_pct']:>+7.1f}% | {r['max_dd']:>5.1f}%")

    # ─── COMBOS of best individual results (run after reviewing above) ───
    print(f"\n{'='*85}")
    print("COMBOS")
    print(f"{'='*85}")
    combos = {
        "Risk2.5% + trail0.5": {**BASE, "risk_pct": 0.025, "trail_tight_dist": 0.5},
        "Risk2.5% + R4:1": {**BASE, "risk_pct": 0.025, "atr_tp_mult": ATR_SL_MULT * 4},
        "Risk2.5% + partial@1.5R": {**BASE, "risk_pct": 0.025, "partial_tp_r": 1.5},
        "Risk3% + trail0.5": {**BASE, "risk_pct": 0.03, "trail_tight_dist": 0.5},
        "Risk2.5%+trail0.5+mom3": {**BASE, "risk_pct": 0.025, "trail_tight_dist": 0.5, "mom_filter": 3},
        "Risk2.5%+trail0.5+noMon": {**BASE, "risk_pct": 0.025, "trail_tight_dist": 0.5,
                                     "allowed_days": [1, 2, 3, 4]},
        "Risk2.5%+trail0.5+ATR>30": {**BASE, "risk_pct": 0.025, "trail_tight_dist": 0.5,
                                      "atr_pct_min": 0.30},
        "Risk2.5%+R4:1+trail0.5": {**BASE, "risk_pct": 0.025, "atr_tp_mult": ATR_SL_MULT * 4,
                                    "trail_tight_dist": 0.5},
        "Risk2.5%+partial1.5R+trail0.5": {**BASE, "risk_pct": 0.025, "partial_tp_r": 1.5,
                                           "trail_tight_dist": 0.5},
    }
    print(f"{'Test':<36} | {'Trades':>6} | {'WR':>6} | {'Ret $':>9} | {'Ret %':>8} | {'DD %':>5}")
    print("-" * 85)
    for name, cfg in combos.items():
        r = run(symbols_data_default, cfg)
        print(f"{name:<36} | {r['trades']:>6} | {r['wr']:>5.1f}% | {r['ret']:>+9.2f} | {r['ret_pct']:>+7.1f}% | {r['max_dd']:>5.1f}%")


if __name__ == "__main__":
    main()
