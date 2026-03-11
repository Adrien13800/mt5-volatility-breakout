"""
Benchmark v6 — nouvelles améliorations par-dessus la base v5.
Base v5 : R:R 4:1, SL=1.25 ATR, trail_dist=0.5, risk=2.5%, session 17h-21h.

Axes testés :
  1. R:R agressif (5:1, 6:1)
  2. SL serré (1.0, 0.75 ATR)
  3. Risk agressif (3%, 3.5%, 4%)
  4. BB length (14, 30)
  5. EMA length (20, 100, 200)
  6. Breakeven stop (SL → entry à 1R)
  7. Time exit (fermer après N bougies)
  8. Trail activate plus tôt (1R, 1.5R)
  9. Trail dist serré (0.3, 0.4)
  10. ATR cap mult (1.0, 2.0)
  11. Max positions simultanées (1, 2)
  12. Combos des meilleurs
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

# ─── Base v5 defaults ───
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


# ════════════════════════════════════════════════════════════════
#  DATA
# ════════════════════════════════════════════════════════════════

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


def indicators(df, bb_std=BB_STD, bb_length=BB_LENGTH,
               ema_length=EMA_TREND_LENGTH, atr_cap_mult=ATR_CAP_MULT):
    df = df.copy()
    bb = ta.bbands(df["close"], length=bb_length, std=bb_std)
    col_map = {}
    for c in bb.columns:
        if c.startswith("BBU"): col_map[c] = "BBU"
        elif c.startswith("BBL"): col_map[c] = "BBL"
    bb = bb.rename(columns=col_map)
    df = pd.concat([df, bb], axis=1)
    df["vol_sma"] = ta.sma(df["tick_volume"].astype(float), length=VOL_SMA_LENGTH)
    df["ema_trend"] = ta.ema(df["close"], length=ema_length)
    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH)
    atr_med = df["atr"].rolling(ATR_MEDIAN_WINDOW, min_periods=1).median()
    df["atr_capped"] = df["atr"].clip(upper=atr_med * atr_cap_mult)
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


# ════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def run(symbols_data, cfg):
    """Run backtest with given config dict."""
    session_start_h = cfg.get("session_start_h", 17)
    session_start_m = cfg.get("session_start_m", 0)
    session_end_h = cfg.get("session_end_h", 21)
    session_end_m = cfg.get("session_end_m", 0)

    atr_sl_mult = cfg.get("atr_sl_mult", ATR_SL_MULT)
    risk_reward = cfg.get("risk_reward", RISK_REWARD)
    atr_tp_mult = cfg.get("atr_tp_mult", atr_sl_mult * risk_reward)
    risk_pct = cfg.get("risk_pct", RISK_PCT)
    trail_tight_r = cfg.get("trail_tight_r", 1.5)  # v6: 1.5R
    trail_tight_dist = cfg.get("trail_tight_dist", 0.3)  # v6: 0.3 ATR
    vol_mult = cfg.get("vol_mult", VOL_MULT)

    # New v6 params
    breakeven_r = cfg.get("breakeven_r", 0)        # 0=off, move SL to entry at N R
    time_exit_bars = cfg.get("time_exit_bars", 0)   # 0=off, close after N bars
    max_positions = cfg.get("max_positions", 0)      # 0=unlimited

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

            # ── Entry ──
            if symbol in pending and symbol not in positions:
                # Max positions check
                if max_positions > 0 and len(positions) >= max_positions:
                    pending.pop(symbol, None)
                else:
                    sig = pending.pop(symbol)
                    ep = op; ac = sig["atr_capped"]
                    lt = lot_size(equity, ac, specs, risk_pct, atr_sl_mult)
                    if sig["d"] == "long":
                        positions[symbol] = {
                            "d": "long", "ep": ep, "sl": ep - atr_sl_mult * ac,
                            "tp": ep + atr_tp_mult * ac, "et": t_paris, "ae": ac,
                            "bp": ep, "lt": lt, "entry_bar": i}
                    else:
                        positions[symbol] = {
                            "d": "short", "ep": ep, "sl": ep + atr_sl_mult * ac,
                            "tp": ep - atr_tp_mult * ac, "et": t_paris, "ae": ac,
                            "bp": ep, "lt": lt, "entry_bar": i}

            # ── Position management ──
            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = atr_sl_mult * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                # Time exit check
                if time_exit_bars > 0 and (i - p["entry_bar"]) >= time_exit_bars:
                    if p["d"] == "long":
                        pnl = round((close - p["ep"]) / point * upp, 2)
                    else:
                        pnl = round((p["ep"] - close) / point * upp, 2)
                    equity += pnl; trades.append(pnl)
                    del positions[symbol]; continue

                # Ordre réaliste : TP → breakeven → trailing → SL
                if p["d"] == "long":
                    # 1. TP check
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
                    # 5. SL check
                    if lo <= p["sl"]:
                        pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                else:
                    # 1. TP check
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
                    # 5. SL check
                    if hi >= p["sl"]:
                        pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                continue

            # ── Signal generation (only outside position) ──
            if not is_session(t_paris, session_start_h, session_start_m,
                              session_end_h, session_end_m):
                continue

            vt = vol_mult * vsma

            if (close > bbu and close > ema and tv > vt
                    and 50 < rsi < 70):
                pending[symbol] = {"d": "long", "atr_capped": atr_c}
            elif (close < bbl and close < ema and tv > vt
                    and 30 < rsi < 50):
                pending[symbol] = {"d": "short", "atr_capped": atr_c}
            else:
                pending.pop(symbol, None)

    # Close remaining positions
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

    peak = INITIAL_CAPITAL; max_dd = 0; eq = INITIAL_CAPITAL
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


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def build_dataset(raw_dfs, specs_cache, bb_std=BB_STD, bb_length=BB_LENGTH,
                  ema_length=EMA_TREND_LENGTH, atr_cap_mult=ATR_CAP_MULT):
    """Build dataset with specific indicator parameters."""
    data = {}
    for symbol, raw_df in raw_dfs.items():
        df = indicators(raw_df, bb_std=bb_std, bb_length=bb_length,
                        ema_length=ema_length, atr_cap_mult=atr_cap_mult)
        data[symbol] = {"df": df, "specs": specs_cache[symbol]}
    return data


def print_result(name, r):
    print(f"  {name:<40} │ {r['trades']:>5} │ {r['wr']:>5.1f}% │"
          f" {r['ret_pct']:>+8.1f}% │ {r['max_dd']:>5.1f}% │ {r['equity']:>10.0f}$")


def main():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"MT5 error: {mt5.last_error()}")
        return

    # ── Fetch raw data ──
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
        raw_dfs[symbol] = df

    mt5.shutdown()

    if not raw_dfs:
        print("No data"); return

    # ── Build datasets with different indicator params ──
    print("Building datasets...")
    ds_default = build_dataset(raw_dfs, specs_cache)
    ds_bb14 = build_dataset(raw_dfs, specs_cache, bb_length=14)
    ds_bb30 = build_dataset(raw_dfs, specs_cache, bb_length=30)
    ds_ema20 = build_dataset(raw_dfs, specs_cache, ema_length=20)
    ds_ema100 = build_dataset(raw_dfs, specs_cache, ema_length=100)
    ds_ema200 = build_dataset(raw_dfs, specs_cache, ema_length=200)
    ds_cap10 = build_dataset(raw_dfs, specs_cache, atr_cap_mult=1.0)
    ds_cap20 = build_dataset(raw_dfs, specs_cache, atr_cap_mult=2.0)
    print("Done.\n")

    BASE = {}  # v5 defaults already in run()

    header = (f"  {'Test':<40} │ {'Trades':>5} │ {'WR':>6} │"
              f" {'Ret %':>8} │ {'DD %':>5} │ {'Equity':>10}")
    sep = "  " + "─" * 90

    # ══════════════════════════════════════════════════════
    #  REFERENCE
    # ══════════════════════════════════════════════════════
    print("=" * 94)
    print("  REFERENCE v5")
    print("=" * 94)
    print(header); print(sep)
    r = run(ds_default, BASE)
    print_result("v5 BASE (R4:1 SL1.25 trail0.5 risk2.5%)", r)

    # ══════════════════════════════════════════════════════
    #  1. R:R AGRESSIF
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  1. RISK:REWARD RATIO")
    print("=" * 94)
    print(header); print(sep)
    for rr in [5, 6, 7, 8]:
        cfg = {**BASE, "risk_reward": rr}
        r = run(ds_default, cfg)
        print_result(f"R:R {rr}:1", r)

    # ══════════════════════════════════════════════════════
    #  2. SL MULTIPLIER
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  2. SL MULTIPLIER (ATR)")
    print("=" * 94)
    print(header); print(sep)
    for sl in [0.75, 1.0, 1.5, 1.75]:
        cfg = {**BASE, "atr_sl_mult": sl}
        r = run(ds_default, cfg)
        print_result(f"SL = {sl} ATR", r)

    # ══════════════════════════════════════════════════════
    #  3. RISK PER TRADE
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  3. RISK PER TRADE")
    print("=" * 94)
    print(header); print(sep)
    for risk in [0.03, 0.035, 0.04, 0.05]:
        cfg = {**BASE, "risk_pct": risk}
        r = run(ds_default, cfg)
        print_result(f"Risk {risk*100:.1f}%", r)

    # ══════════════════════════════════════════════════════
    #  4. BOLLINGER BAND LENGTH
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  4. BOLLINGER BAND LENGTH")
    print("=" * 94)
    print(header); print(sep)
    r = run(ds_bb14, BASE)
    print_result("BB length = 14", r)
    r = run(ds_bb30, BASE)
    print_result("BB length = 30", r)

    # ══════════════════════════════════════════════════════
    #  5. EMA TREND LENGTH
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  5. EMA TREND LENGTH")
    print("=" * 94)
    print(header); print(sep)
    for ds, name in [(ds_ema20, "EMA 20"), (ds_ema100, "EMA 100"), (ds_ema200, "EMA 200")]:
        r = run(ds, BASE)
        print_result(name, r)

    # ══════════════════════════════════════════════════════
    #  6. BREAKEVEN STOP
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  6. BREAKEVEN STOP (SL -> entry at N R)")
    print("=" * 94)
    print(header); print(sep)
    for be in [0.5, 1.0, 1.5]:
        cfg = {**BASE, "breakeven_r": be}
        r = run(ds_default, cfg)
        print_result(f"Breakeven @ {be}R", r)

    # ══════════════════════════════════════════════════════
    #  7. TIME EXIT
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  7. TIME EXIT (close after N bars)")
    print("=" * 94)
    print(header); print(sep)
    for bars in [20, 40, 60, 80]:
        cfg = {**BASE, "time_exit_bars": bars}
        r = run(ds_default, cfg)
        print_result(f"Time exit {bars} bars ({bars*15/60:.0f}h)", r)

    # ══════════════════════════════════════════════════════
    #  8. TRAIL ACTIVATE EARLIER
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  8. TRAIL ACTIVATION THRESHOLD (R)")
    print("=" * 94)
    print(header); print(sep)
    for tr in [1.0, 1.5, 2.5, 3.0]:
        cfg = {**BASE, "trail_tight_r": tr}
        r = run(ds_default, cfg)
        print_result(f"Trail activate @ {tr}R", r)

    # ══════════════════════════════════════════════════════
    #  9. TRAIL DISTANCE
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  9. TRAIL DISTANCE (ATR)")
    print("=" * 94)
    print(header); print(sep)
    for td in [0.2, 0.3, 0.4, 0.6, 0.75]:
        cfg = {**BASE, "trail_tight_dist": td}
        r = run(ds_default, cfg)
        print_result(f"Trail dist = {td} ATR", r)

    # ══════════════════════════════════════════════════════
    #  10. ATR CAP MULTIPLIER
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  10. ATR CAP MULTIPLIER")
    print("=" * 94)
    print(header); print(sep)
    r = run(ds_cap10, BASE)
    print_result("ATR cap = 1.0x median", r)
    r = run(ds_cap20, BASE)
    print_result("ATR cap = 2.0x median", r)

    # ══════════════════════════════════════════════════════
    #  11. MAX POSITIONS SIMULTANÉES
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  11. MAX POSITIONS")
    print("=" * 94)
    print(header); print(sep)
    for mp in [1, 2]:
        cfg = {**BASE, "max_positions": mp}
        r = run(ds_default, cfg)
        print_result(f"Max {mp} position(s)", r)

    # ══════════════════════════════════════════════════════
    #  12. COMBOS
    # ══════════════════════════════════════════════════════
    print(f"\n{'=' * 94}")
    print("  12. COMBOS (meilleurs individuels)")
    print("=" * 94)
    print(header); print(sep)

    combos = {
        # R:R + risk
        "R5:1 + risk3%": {"risk_reward": 5, "risk_pct": 0.03},
        "R5:1 + risk3.5%": {"risk_reward": 5, "risk_pct": 0.035},
        "R6:1 + risk3%": {"risk_reward": 6, "risk_pct": 0.03},
        "R6:1 + risk3.5%": {"risk_reward": 6, "risk_pct": 0.035},

        # R:R + trail
        "R5:1 + trail_dist0.3": {"risk_reward": 5, "trail_tight_dist": 0.3},
        "R5:1 + trail_dist0.4": {"risk_reward": 5, "trail_tight_dist": 0.4},
        "R5:1 + trail_act1.5R": {"risk_reward": 5, "trail_tight_r": 1.5},
        "R6:1 + trail_dist0.3": {"risk_reward": 6, "trail_tight_dist": 0.3},

        # SL + risk
        "SL1.0 + risk3%": {"atr_sl_mult": 1.0, "risk_pct": 0.03},
        "SL1.0 + risk3.5%": {"atr_sl_mult": 1.0, "risk_pct": 0.035},

        # Breakeven combos
        "BE@1R + trail_act1.5R": {"breakeven_r": 1.0, "trail_tight_r": 1.5},
        "BE@1R + trail_dist0.3": {"breakeven_r": 1.0, "trail_tight_dist": 0.3},
        "BE@0.5R + trail_act1R": {"breakeven_r": 0.5, "trail_tight_r": 1.0},

        # Triple combos
        "R5:1+risk3%+trail0.3": {"risk_reward": 5, "risk_pct": 0.03,
                                  "trail_tight_dist": 0.3},
        "R5:1+risk3%+trail_act1.5R": {"risk_reward": 5, "risk_pct": 0.03,
                                       "trail_tight_r": 1.5},
        "R5:1+risk3.5%+trail0.4": {"risk_reward": 5, "risk_pct": 0.035,
                                    "trail_tight_dist": 0.4},
        "R6:1+risk3%+trail0.3": {"risk_reward": 6, "risk_pct": 0.03,
                                  "trail_tight_dist": 0.3},
        "R5:1+risk3%+BE@1R": {"risk_reward": 5, "risk_pct": 0.03,
                               "breakeven_r": 1.0},
        "R5:1+risk3%+trail0.3+BE@1R": {"risk_reward": 5, "risk_pct": 0.03,
                                        "trail_tight_dist": 0.3, "breakeven_r": 1.0},
        "R6:1+risk3%+trail0.3+BE@1R": {"risk_reward": 6, "risk_pct": 0.03,
                                        "trail_tight_dist": 0.3, "breakeven_r": 1.0},
        "R5:1+risk3.5%+trail0.3+BE@1R": {"risk_reward": 5, "risk_pct": 0.035,
                                          "trail_tight_dist": 0.3, "breakeven_r": 1.0},

        # Quad combos
        "R5:1+risk3%+trail0.3+act1.5R": {"risk_reward": 5, "risk_pct": 0.03,
                                          "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "R5:1+risk3%+trail0.4+act1.5R": {"risk_reward": 5, "risk_pct": 0.03,
                                          "trail_tight_dist": 0.4, "trail_tight_r": 1.5},
        "R6:1+risk3%+trail0.3+act1.5R": {"risk_reward": 6, "risk_pct": 0.03,
                                          "trail_tight_dist": 0.3, "trail_tight_r": 1.5},
        "R5:1+r3.5%+t0.3+act1.5R+BE1R": {"risk_reward": 5, "risk_pct": 0.035,
                                           "trail_tight_dist": 0.3, "trail_tight_r": 1.5,
                                           "breakeven_r": 1.0},
    }

    for name, cfg in combos.items():
        r = run(ds_default, {**BASE, **cfg})
        print_result(name, r)

    print(f"\n{'=' * 94}")
    print("  Benchmark v6 terminé.")
    print(f"{'=' * 94}")


if __name__ == "__main__":
    main()
