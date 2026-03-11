"""
Benchmark v6 Round 2 — teste les combos manquants avec SL 0.75 et trail 0.2.
Les 2 plus gros boosters individuels du round 1 n'ont pas été testés en combo.
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

BB_LENGTH = 20; BB_STD = 2.0
VOL_SMA_LENGTH = 20; VOL_MULT = 1.3
EMA_TREND_LENGTH = 50
RSI_LENGTH = 14; ATR_LENGTH = 14

RISK_REWARD = 4
ATR_SL_MULT = 0.75  # v6
ATR_TP_MULT = ATR_SL_MULT * RISK_REWARD  # 3.0
ATR_MEDIAN_WINDOW = 100; ATR_CAP_MULT = 1.5
RISK_PCT = 0.03; BACKTEST_MONTHS = 8  # v6: 3%
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
    df = df.copy()
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
    return df


def lot_size(equity, atr_c, specs, risk_pct, atr_sl_mult):
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
    atr_sl_mult = cfg.get("atr_sl_mult", ATR_SL_MULT)
    risk_reward = cfg.get("risk_reward", RISK_REWARD)
    atr_tp_mult = cfg.get("atr_tp_mult", atr_sl_mult * risk_reward)
    risk_pct = cfg.get("risk_pct", RISK_PCT)
    trail_tight_r = cfg.get("trail_tight_r", 1.5)  # v6: 1.5R
    trail_tight_dist = cfg.get("trail_tight_dist", 0.3)  # v6: 0.3 ATR
    vol_mult = cfg.get("vol_mult", VOL_MULT)
    breakeven_r = cfg.get("breakeven_r", 0)

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
                    positions[symbol] = {
                        "d": "long", "ep": ep, "sl": ep - atr_sl_mult * ac,
                        "tp": ep + atr_tp_mult * ac, "et": t_paris, "ae": ac,
                        "bp": ep, "lt": lt}
                else:
                    positions[symbol] = {
                        "d": "short", "ep": ep, "sl": ep + atr_sl_mult * ac,
                        "tp": ep - atr_tp_mult * ac, "et": t_paris, "ae": ac,
                        "bp": ep, "lt": lt}

            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = atr_sl_mult * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                if p["d"] == "long":
                    # 1. TP
                    if hi >= p["tp"]:
                        pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                    # 2. Update best price
                    if hi > p["bp"]: p["bp"] = hi
                    # 3. Breakeven
                    if breakeven_r > 0:
                        profit_r = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                        if profit_r >= breakeven_r and p["sl"] < p["ep"]:
                            p["sl"] = p["ep"]
                    # 4. Trailing
                    pr = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] - trail_tight_dist * ae
                        if ns > p["sl"]: p["sl"] = ns
                    # 5. SL
                    if lo <= p["sl"]:
                        pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                else:
                    # 1. TP
                    if lo <= p["tp"]:
                        pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                    # 2. Update best price
                    if lo < p["bp"]: p["bp"] = lo
                    # 3. Breakeven
                    if breakeven_r > 0:
                        profit_r = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                        if profit_r >= breakeven_r and p["sl"] > p["ep"]:
                            p["sl"] = p["ep"]
                    # 4. Trailing
                    pr = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                    if pr >= trail_tight_r:
                        ns = p["bp"] + trail_tight_dist * ae
                        if ns < p["sl"]: p["sl"] = ns
                    # 5. SL
                    if hi >= p["sl"]:
                        pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                continue

            if not is_session(t_paris, 17, 0, 21, 0):
                continue

            vt = vol_mult * vsma
            if (close > bbu and close > ema and tv > vt and 50 < rsi < 70):
                pending[symbol] = {"d": "long", "atr_capped": atr_c}
            elif (close < bbl and close < ema and tv > vt and 30 < rsi < 50):
                pending[symbol] = {"d": "short", "atr_capped": atr_c}
            else:
                pending.pop(symbol, None)

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
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak = INITIAL_CAPITAL; max_dd = 0; eq = INITIAL_CAPITAL
    for t in trades:
        eq += t
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        "trades": len(trades), "wins": wins, "wr": round(wr, 1),
        "ret_pct": round(ret_pct, 1), "equity": round(equity, 2),
        "max_dd": round(max_dd, 1),
    }


def pr(name, r):
    print(f"  {name:<45} │ {r['trades']:>5} │ {r['wr']:>5.1f}% │"
          f" {r['ret_pct']:>+9.1f}% │ {r['max_dd']:>5.1f}% │ {r['equity']:>10.0f}$")


def main():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 error: {mt5.last_error()}")
        return

    raw_dfs = {}
    specs_cache = {}
    for label, symbol in SYMBOLS.items():
        if not mt5.symbol_select(symbol, True): continue
        df = fetch(symbol, BACKTEST_MONTHS, TIMEFRAME)
        if df is None: continue
        info = mt5.symbol_info(symbol)
        specs_cache[symbol] = {
            "point": info.point, "tick_value": info.trade_tick_value,
            "tick_size": info.trade_tick_size, "vol_min": info.volume_min,
            "vol_max": info.volume_max, "vol_step": info.volume_step}
        df = indicators(df)
        raw_dfs[symbol] = {"df": df, "specs": specs_cache[symbol]}

    mt5.shutdown()
    if not raw_dfs:
        print("No data"); return

    header = (f"  {'Test':<45} │ {'Trades':>5} │ {'WR':>6} │"
              f" {'Ret %':>9} │ {'DD %':>5} │ {'Equity':>10}")
    sep = "  " + "─" * 95

    # ══════════════════════════════════════════════════════
    #  REFERENCE
    # ══════════════════════════════════════════════════════
    print("=" * 99)
    print("  ROUND 2 — SL 0.75 & Trail 0.2 combos")
    print("=" * 99)
    print(header); print(sep)

    # Référence v5
    pr("v5 BASE", run(raw_dfs, {}))
    # Rappel du best round 1
    pr("R1 best: R6+r3%+t0.3+act1.5R", run(raw_dfs, {
        "risk_reward": 6, "risk_pct": 0.03,
        "trail_tight_dist": 0.3, "trail_tight_r": 1.5}))

    # ── SL 0.75 seul (rappel) ──
    print(sep)
    print("  --- SL 0.75 combos ---")
    print(sep)
    tests_sl075 = {
        "SL0.75 (rappel)":
            {"atr_sl_mult": 0.75},
        "SL0.75 + risk3%":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03},
        "SL0.75 + risk3.5%":
            {"atr_sl_mult": 0.75, "risk_pct": 0.035},
        "SL0.75 + trail0.3":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.3},
        "SL0.75 + trail0.2":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2},
        "SL0.75 + act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_r": 1.5},
        "SL0.75 + act1.0R":
            {"atr_sl_mult": 0.75, "trail_tight_r": 1.0},
        "SL0.75 + R5:1":
            {"atr_sl_mult": 0.75, "risk_reward": 5},
        "SL0.75 + R6:1":
            {"atr_sl_mult": 0.75, "risk_reward": 6},
        "SL0.75+r3%+trail0.3":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03, "trail_tight_dist": 0.3},
        "SL0.75+r3%+trail0.2":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03, "trail_tight_dist": 0.2},
        "SL0.75+r3%+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03, "trail_tight_r": 1.5},
        "SL0.75+r3%+trail0.3+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+r3%+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03,
             "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "SL0.75+r3.5%+trail0.3+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.035,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+r3.5%+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.035,
             "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "SL0.75+R5:1+r3%+trail0.3+act1.5R":
            {"atr_sl_mult": 0.75, "risk_reward": 5, "risk_pct": 0.03,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+R5:1+r3%+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "risk_reward": 5, "risk_pct": 0.03,
             "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "SL0.75+R6:1+r3%+trail0.3+act1.5R":
            {"atr_sl_mult": 0.75, "risk_reward": 6, "risk_pct": 0.03,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+R6:1+r3%+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "risk_reward": 6, "risk_pct": 0.03,
             "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
    }
    for name, cfg in tests_sl075.items():
        pr(name, run(raw_dfs, cfg))

    # ── Trail 0.2 combos (SL 1.25 par défaut) ──
    print(sep)
    print("  --- Trail 0.2 combos (SL 1.25 défaut) ---")
    print(sep)
    tests_t02 = {
        "Trail0.2 (rappel)":
            {"trail_tight_dist": 0.2},
        "Trail0.2 + risk3%":
            {"trail_tight_dist": 0.2, "risk_pct": 0.03},
        "Trail0.2 + risk3.5%":
            {"trail_tight_dist": 0.2, "risk_pct": 0.035},
        "Trail0.2 + act1.5R":
            {"trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "Trail0.2 + act1.0R":
            {"trail_tight_dist": 0.2, "trail_tight_r": 1.0},
        "Trail0.2 + R5:1":
            {"trail_tight_dist": 0.2, "risk_reward": 5},
        "Trail0.2 + R6:1":
            {"trail_tight_dist": 0.2, "risk_reward": 6},
        "Trail0.2+r3%+act1.5R":
            {"trail_tight_dist": 0.2, "risk_pct": 0.03, "trail_tight_r": 1.5},
        "Trail0.2+r3%+R5:1":
            {"trail_tight_dist": 0.2, "risk_pct": 0.03, "risk_reward": 5},
        "Trail0.2+r3%+R5:1+act1.5R":
            {"trail_tight_dist": 0.2, "risk_pct": 0.03,
             "risk_reward": 5, "trail_tight_r": 1.5},
        "Trail0.2+r3%+R6:1+act1.5R":
            {"trail_tight_dist": 0.2, "risk_pct": 0.03,
             "risk_reward": 6, "trail_tight_r": 1.5},
        "Trail0.2+r3.5%+R5:1+act1.5R":
            {"trail_tight_dist": 0.2, "risk_pct": 0.035,
             "risk_reward": 5, "trail_tight_r": 1.5},
        "Trail0.2+r3.5%+R6:1+act1.5R":
            {"trail_tight_dist": 0.2, "risk_pct": 0.035,
             "risk_reward": 6, "trail_tight_r": 1.5},
    }
    for name, cfg in tests_t02.items():
        pr(name, run(raw_dfs, cfg))

    # ── MEGA COMBOS : SL 0.75 + trail 0.2 ensemble ──
    print(sep)
    print("  --- MEGA COMBOS : SL 0.75 + Trail 0.2 ---")
    print(sep)
    tests_mega = {
        "SL0.75+trail0.2":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2},
        "SL0.75+trail0.2+r3%":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2, "risk_pct": 0.03},
        "SL0.75+trail0.2+r3.5%":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2, "risk_pct": 0.035},
        "SL0.75+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+act1.0R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2, "trail_tight_r": 1.0},
        "SL0.75+trail0.2+r3%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_pct": 0.03, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+r3%+act1.0R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_pct": 0.03, "trail_tight_r": 1.0},
        "SL0.75+trail0.2+r3.5%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_pct": 0.035, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+R5:1+r3%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_reward": 5, "risk_pct": 0.03, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+R6:1+r3%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_reward": 6, "risk_pct": 0.03, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+R5:1+r3.5%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_reward": 5, "risk_pct": 0.035, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+R6:1+r3.5%+act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_reward": 6, "risk_pct": 0.035, "trail_tight_r": 1.5},
    }
    for name, cfg in tests_mega.items():
        pr(name, run(raw_dfs, cfg))

    print(f"\n{'=' * 99}")
    print("  Round 2 terminé.")
    print(f"{'=' * 99}")


if __name__ == "__main__":
    main()
