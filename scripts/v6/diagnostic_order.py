"""
Diagnostic : compare l'ordre intra-bougie optimiste vs pessimiste.
- Optimiste : TP → update trailing → SL  (le backtest actuel)
- Pessimiste : TP → SL (ancien SL) → update trailing (pour la prochaine bougie)

Si les résultats s'effondrent en pessimiste, c'est que l'ordre favorable
fait tout le travail = faux edge.
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


def run(symbols_data, cfg, order="optimistic"):
    """
    order = "optimistic"  : TP → update trailing → SL  (backtest actuel)
    order = "pessimistic" : TP → SL (ancien) → update trailing (pour next bar)
    """
    atr_sl_mult = cfg.get("atr_sl_mult", ATR_SL_MULT)
    risk_reward = cfg.get("risk_reward", RISK_REWARD)
    atr_tp_mult = cfg.get("atr_tp_mult", atr_sl_mult * risk_reward)
    risk_pct = cfg.get("risk_pct", RISK_PCT)
    trail_tight_r = cfg.get("trail_tight_r", 1.5)  # v6: 1.5R
    trail_tight_dist = cfg.get("trail_tight_dist", 0.3)  # v6: 0.3 ATR
    vol_mult = cfg.get("vol_mult", VOL_MULT)

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

            if symbol in pending and symbol not in positions:
                sig = pending.pop(symbol)
                ep = op; ac = sig["atr_capped"]
                lt = lot_size(equity, ac, specs, risk_pct, atr_sl_mult)
                if sig["d"] == "long":
                    positions[symbol] = {
                        "d": "long", "ep": ep, "sl": ep - atr_sl_mult * ac,
                        "tp": ep + atr_tp_mult * ac, "ae": ac, "bp": ep, "lt": lt}
                else:
                    positions[symbol] = {
                        "d": "short", "ep": ep, "sl": ep + atr_sl_mult * ac,
                        "tp": ep - atr_tp_mult * ac, "ae": ac, "bp": ep, "lt": lt}

            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = atr_sl_mult * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                if order == "optimistic":
                    # TP → Trail → SL (favorable)
                    if p["d"] == "long":
                        if hi >= p["tp"]:
                            pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        if hi > p["bp"]: p["bp"] = hi
                        pr_val = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                        if pr_val >= trail_tight_r:
                            ns = p["bp"] - trail_tight_dist * ae
                            if ns > p["sl"]: p["sl"] = ns
                        if lo <= p["sl"]:
                            pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                    else:
                        if lo <= p["tp"]:
                            pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        if lo < p["bp"]: p["bp"] = lo
                        pr_val = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                        if pr_val >= trail_tight_r:
                            ns = p["bp"] + trail_tight_dist * ae
                            if ns < p["sl"]: p["sl"] = ns
                        if hi >= p["sl"]:
                            pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue

                else:
                    # TP → SL (ancien) → Trail (pessimiste)
                    if p["d"] == "long":
                        if hi >= p["tp"]:
                            pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        # SL check AVANT trailing update
                        if lo <= p["sl"]:
                            pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        # Trailing update (pour la prochaine bougie)
                        if hi > p["bp"]: p["bp"] = hi
                        pr_val = (p["bp"] - p["ep"]) / one_r if one_r > 0 else 0
                        if pr_val >= trail_tight_r:
                            ns = p["bp"] - trail_tight_dist * ae
                            if ns > p["sl"]: p["sl"] = ns
                    else:
                        if lo <= p["tp"]:
                            pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        if hi >= p["sl"]:
                            pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                            equity += pnl; trades.append(pnl)
                            del positions[symbol]; continue
                        if lo < p["bp"]: p["bp"] = lo
                        pr_val = (p["ep"] - p["bp"]) / one_r if one_r > 0 else 0
                        if pr_val >= trail_tight_r:
                            ns = p["bp"] + trail_tight_dist * ae
                            if ns < p["sl"]: p["sl"] = ns
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
    print(f"  {name:<48} │ {r['trades']:>5} │ {r['wr']:>5.1f}% │"
          f" {r['ret_pct']:>+10.1f}% │ {r['max_dd']:>5.1f}% │ {r['equity']:>10.0f}$")


def main():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 error: {mt5.last_error()}")
        return

    data = {}
    for label, symbol in SYMBOLS.items():
        if not mt5.symbol_select(symbol, True): continue
        df = fetch(symbol, BACKTEST_MONTHS, TIMEFRAME)
        if df is None: continue
        info = mt5.symbol_info(symbol)
        specs = {"point": info.point, "tick_value": info.trade_tick_value,
                 "tick_size": info.trade_tick_size, "vol_min": info.volume_min,
                 "vol_max": info.volume_max, "vol_step": info.volume_step}
        df = indicators(df)
        data[symbol] = {"df": df, "specs": specs}

    mt5.shutdown()
    if not data:
        print("No data"); return

    header = (f"  {'Test':<48} │ {'Trades':>5} │ {'WR':>6} │"
              f" {'Ret %':>10} │ {'DD %':>5} │ {'Equity':>10}")
    sep = "  " + "─" * 100

    configs = {
        "v5 BASE (SL1.25 trail0.5 act2R)":
            {},
        "SL0.75 + trail0.3 + act1.5R":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+r3%+trail0.3+act1.5R (recommandé)":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "SL0.75+r3%+trail0.2+act1.5R":
            {"atr_sl_mult": 0.75, "risk_pct": 0.03,
             "trail_tight_dist": 0.2, "trail_tight_r": 1.5},
        "SL0.75+trail0.2+r3%+act1.0R (mega)":
            {"atr_sl_mult": 0.75, "trail_tight_dist": 0.2,
             "risk_pct": 0.03, "trail_tight_r": 1.0},
        "R6:1+r3%+trail0.3+act1.5R (R1 best)":
            {"risk_reward": 6, "risk_pct": 0.03,
             "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
    }

    print("=" * 104)
    print("  DIAGNOSTIC : Ordre OPTIMISTE (TP → Trail → SL)")
    print("  = backtest actuel, favorable au trailing")
    print("=" * 104)
    print(header); print(sep)
    for name, cfg in configs.items():
        r = run(data, cfg, order="optimistic")
        pr(name, r)

    print(f"\n{'=' * 104}")
    print("  DIAGNOSTIC : Ordre PESSIMISTE (TP → SL → Trail)")
    print("  = worst case, SL vérifié AVANT le trailing update")
    print("=" * 104)
    print(header); print(sep)
    for name, cfg in configs.items():
        r = run(data, cfg, order="pessimistic")
        pr(name, r)

    print(f"\n{'=' * 104}")
    print("  Si les résultats s'effondrent en pessimiste → le trailing")
    print("  favorable fait tout le travail = faux edge en backtest.")
    print("  La réalité est ENTRE les deux ordres.")
    print(f"{'=' * 104}")


if __name__ == "__main__":
    main()
