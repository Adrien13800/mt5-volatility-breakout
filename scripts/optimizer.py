"""
Strategy Optimizer — Recherche exhaustive sur 10 000+ combinaisons.
Varie : timeframe, Bollinger Bands, EMA, RSI, volume, ATR, SL/TP,
        trailing, sessions horaires, jours de trading.
Execution parallele via multiprocessing (N-1 workers CPU).

Usage:  python optimizer.py
Output: optimizer_results.csv  (toutes les combinaisons)
        optimizer_top50.csv    (top 50 par rendement filtre freq.)
"""

import os
import random
import sys
import time as _time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count, freeze_support

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MT5_LOGIN = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER = os.environ["MT5_SERVER"]

PARIS_TZ = pytz.timezone("Europe/Paris")
INITIAL_CAPITAL = 132
BACKTEST_MONTHS = 8
N_COMBINATIONS = 10_000

SYMBOLS = {"NAS100": "NAS100.", "US30": "DJ30.", "SPX500": "SP500."}

TIMEFRAMES = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
}

DAYS_MAP = {
    "Mon-Fri":      {0, 1, 2, 3, 4},
    "Mon-Thu":      {0, 1, 2, 3},
    "Tue-Fri":      {1, 2, 3, 4},
    "Tue-Thu":      {1, 2, 3},
    "Mon-Wed-Fri":  {0, 2, 4},
}

# ═══════════════════════════════════════════════════════════════════════
#  PARAMETER SPACE  (~14 milliards de combinaisons possibles)
#  10 000 echantillons aleatoires depuis cet espace.
# ═══════════════════════════════════════════════════════════════════════

PARAM_SPACE = {
    # --- Timeframe ---
    "timeframe":        ["M5", "M15", "M30", "H1"],

    # --- Bollinger Bands ---
    "bb_length":        [10, 14, 20, 30],
    "bb_std":           [1.5, 1.8, 2.0, 2.5],

    # --- Volume filter (0 = desactive) ---
    "vol_mult":         [0.0, 0.8, 1.0, 1.1, 1.3],

    # --- EMA trend (0 = desactive) ---
    "ema_length":       [0, 20, 30, 50, 100],

    # --- RSI ---
    "rsi_length":       [10, 14, 20],
    "rsi_long_lo":      [0, 35, 40, 45, 50],
    "rsi_long_hi":      [65, 70, 75, 80, 100],
    "rsi_short_lo":     [0, 20, 25, 30],
    "rsi_short_hi":     [50, 55, 60, 65, 100],

    # --- ATR ---
    "atr_length":       [10, 14, 20],
    "atr_sl_mult":      [0.5, 0.75, 1.0, 1.25, 1.5],
    "atr_cap_mult":     [1.0, 1.5, 2.0],

    # --- Risk / Reward ---
    "risk_reward":      [2, 3, 4, 5, 6],
    "risk_pct":         [0.02, 0.025, 0.03, 0.035, 0.04],

    # --- Trailing stop ---
    "trail_activate_r": [1.0, 1.5, 2.0, 2.5],
    "trail_dist":       [0.2, 0.3, 0.4, 0.5, 0.75],

    # --- Session (heures Paris) ---
    "session_start_h":  [9, 14, 15, 16, 17],
    "session_end_h":    [20, 21, 22, 23],

    # --- Jours de trading ---
    "trading_days":     ["Mon-Fri", "Mon-Thu", "Tue-Fri", "Tue-Thu", "Mon-Wed-Fri"],
}

# Config v6 actuelle (reference)
V6_REFERENCE = {
    "timeframe": "M15", "bb_length": 20, "bb_std": 2.0, "vol_mult": 1.3,
    "ema_length": 50, "rsi_length": 14, "rsi_long_lo": 50, "rsi_long_hi": 70,
    "rsi_short_lo": 30, "rsi_short_hi": 50, "atr_length": 14, "atr_sl_mult": 0.75,
    "atr_cap_mult": 1.5, "risk_reward": 4, "risk_pct": 0.03,
    "trail_activate_r": 1.5, "trail_dist": 0.3,
    "session_start_h": 17, "session_end_h": 21, "trading_days": "Mon-Fri",
}


# ═══════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

def fetch_all_data():
    """Fetch raw OHLCV for every symbol x every timeframe from MT5."""
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"  MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    utc_to = datetime.now(tz=pytz.utc)
    utc_from = utc_to - timedelta(days=BACKTEST_MONTHS * 30)
    all_data = {}
    specs_cache = {}

    for tf_name, tf_val in TIMEFRAMES.items():
        all_data[tf_name] = {}
        for label, symbol in SYMBOLS.items():
            if not mt5.symbol_select(symbol, True):
                print(f"  x {symbol} {tf_name}: symbol not available")
                continue

            rates = mt5.copy_rates_range(symbol, tf_val, utc_from, utc_to)
            if rates is None or len(rates) == 0:
                print(f"  x {symbol} {tf_name}: no data")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["time_paris"] = df["time"].dt.tz_convert(PARIS_TZ)

            if symbol not in specs_cache:
                info = mt5.symbol_info(symbol)
                specs_cache[symbol] = {
                    "point": info.point,
                    "tick_value": info.trade_tick_value,
                    "tick_size": info.trade_tick_size,
                    "vol_min": info.volume_min,
                    "vol_max": info.volume_max,
                    "vol_step": info.volume_step,
                }

            all_data[tf_name][symbol] = {"df": df, "specs": specs_cache[symbol]}
            print(f"  + {symbol:<8} {tf_name:<4} : {len(df):>7,} candles")

    mt5.shutdown()
    return all_data


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════

def compute_indicators(df, cfg):
    """Compute all technical indicators with config-specific parameters."""
    df = df.copy()

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=cfg["bb_length"], std=cfg["bb_std"])
    col_map = {}
    for c in bb.columns:
        if c.startswith("BBU"):
            col_map[c] = "BBU"
        elif c.startswith("BBL"):
            col_map[c] = "BBL"
    bb = bb.rename(columns=col_map)
    df = pd.concat([df, bb], axis=1)

    # Volume SMA (fixed at 20)
    df["vol_sma"] = ta.sma(df["tick_volume"].astype(float), length=20)

    # EMA (0 = disabled)
    if cfg["ema_length"] > 0:
        df["ema_trend"] = ta.ema(df["close"], length=cfg["ema_length"])
    else:
        df["ema_trend"] = np.nan

    # RSI
    df["rsi"] = ta.rsi(df["close"], length=cfg["rsi_length"])

    # ATR + capping
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=cfg["atr_length"])
    atr_med = df["atr"].rolling(100, min_periods=1).median()
    df["atr_capped"] = df["atr"].clip(upper=atr_med * cfg["atr_cap_mult"])

    return df


# ═══════════════════════════════════════════════════════════════════════
#  LOT SIZING
# ═══════════════════════════════════════════════════════════════════════

def lot_size(equity, atr_c, specs, risk_pct, atr_sl_mult):
    risk = equity * risk_pct
    sl_dist = atr_sl_mult * atr_c
    sl_val = (sl_dist / specs["tick_size"]) * specs["tick_value"]
    if sl_val <= 0:
        return specs["vol_min"]
    lot = risk / sl_val
    step = specs["vol_step"]
    lot = round(lot / step) * step
    return max(specs["vol_min"], min(specs["vol_max"], round(lot, 6)))


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(cfg, symbols_data):
    """Run a single backtest. Returns metrics dict or None if too few trades."""
    sl_m = cfg["atr_sl_mult"]
    rr = cfg["risk_reward"]
    tp_m = sl_m * rr
    rpct = cfg["risk_pct"]
    tr_r = cfg["trail_activate_r"]
    tr_d = cfg["trail_dist"]
    vm = cfg["vol_mult"]
    rll = cfg["rsi_long_lo"];  rlh = cfg["rsi_long_hi"]
    rsl = cfg["rsi_short_lo"]; rsh = cfg["rsi_short_hi"]
    s_h = cfg["session_start_h"]
    e_h = cfg["session_end_h"]
    use_ema = cfg["ema_length"] > 0
    days = DAYS_MAP[cfg["trading_days"]]

    equity = INITIAL_CAPITAL
    trades = []
    positions = {}   # symbol -> position dict
    pending = {}     # symbol -> pending signal

    n_bars = min(len(d["df"]) for d in symbols_data.values())
    warmup = max(100, cfg["bb_length"] + 20, cfg["ema_length"] + 20)
    if n_bars <= warmup + 50:
        return None

    for i in range(warmup, n_bars):
        for symbol, sdata in symbols_data.items():
            row = sdata["df"].iloc[i]
            specs = sdata["specs"]
            point = specs["point"]

            close = row["close"]; op = row["open"]
            hi = row["high"];     lo = row["low"]
            bbu = row.get("BBU"); bbl = row.get("BBL")
            tv = row["tick_volume"]; vsma = row.get("vol_sma")
            ema = row.get("ema_trend")
            rsi = row.get("rsi")
            atr_c = row.get("atr_capped")
            t_p = row["time_paris"]

            # Validate indicators
            if any(pd.isna(v) for v in [bbu, bbl, vsma, rsi, atr_c]):
                pending.pop(symbol, None)
                continue
            if use_ema and pd.isna(ema):
                pending.pop(symbol, None)
                continue
            if atr_c == 0:
                pending.pop(symbol, None)
                continue

            # ── Deferred entry (signal bar[i-1] -> entry at open bar[i]) ──
            if symbol in pending and symbol not in positions:
                sig = pending.pop(symbol)
                ep = op
                ac = sig["ac"]
                lt = lot_size(equity, ac, specs, rpct, sl_m)
                if sig["d"] == "long":
                    positions[symbol] = {
                        "d": "long", "ep": ep,
                        "sl": ep - sl_m * ac, "tp": ep + tp_m * ac,
                        "ae": ac, "bp": ep, "lt": lt,
                    }
                else:
                    positions[symbol] = {
                        "d": "short", "ep": ep,
                        "sl": ep + sl_m * ac, "tp": ep - tp_m * ac,
                        "ae": ac, "bp": ep, "lt": lt,
                    }

            # ── Position management ──
            if symbol in positions:
                p = positions[symbol]
                ae = p["ae"]; lt = p["lt"]
                one_r = sl_m * ae
                upp = (specs["tick_value"] / specs["tick_size"]) * point * lt

                if p["d"] == "long":
                    # TP hit
                    if hi >= p["tp"]:
                        pnl = round((p["tp"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                    # Update best price
                    if hi > p["bp"]:
                        p["bp"] = hi
                    # Trailing
                    if one_r > 0:
                        pr = (p["bp"] - p["ep"]) / one_r
                        if pr >= tr_r:
                            ns = p["bp"] - tr_d * ae
                            if ns > p["sl"]:
                                p["sl"] = ns
                    # SL hit
                    if lo <= p["sl"]:
                        pnl = round((p["sl"] - p["ep"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                else:
                    # TP hit
                    if lo <= p["tp"]:
                        pnl = round((p["ep"] - p["tp"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                    # Update best price
                    if lo < p["bp"]:
                        p["bp"] = lo
                    # Trailing
                    if one_r > 0:
                        pr = (p["ep"] - p["bp"]) / one_r
                        if pr >= tr_r:
                            ns = p["bp"] + tr_d * ae
                            if ns < p["sl"]:
                                p["sl"] = ns
                    # SL hit
                    if hi >= p["sl"]:
                        pnl = round((p["ep"] - p["sl"]) / point * upp, 2)
                        equity += pnl; trades.append(pnl)
                        del positions[symbol]; continue
                continue  # Skip signal generation while in position

            # ── Signal generation (only outside position) ──
            # Day filter
            if t_p.weekday() not in days:
                continue
            # Session filter
            if not (s_h <= t_p.hour < e_h):
                continue

            # Conditions
            vol_ok = (tv > vm * vsma) if vm > 0 else True
            ema_l = (close > ema) if use_ema else True
            ema_s = (close < ema) if use_ema else True

            if close > bbu and ema_l and vol_ok and rll < rsi < rlh:
                pending[symbol] = {"d": "long", "ac": atr_c}
            elif close < bbl and ema_s and vol_ok and rsl < rsi < rsh:
                pending[symbol] = {"d": "short", "ac": atr_c}
            else:
                pending.pop(symbol, None)

    # Close remaining positions at last close
    for symbol, p in list(positions.items()):
        sdata = symbols_data[symbol]
        specs = sdata["specs"]; point = specs["point"]; lt = p["lt"]
        upp = (specs["tick_value"] / specs["tick_size"]) * point * lt
        last = sdata["df"].iloc[-1]
        if p["d"] == "long":
            pnl = round((last["close"] - p["ep"]) / point * upp, 2)
        else:
            pnl = round((p["ep"] - last["close"]) / point * upp, 2)
        equity += pnl; trades.append(pnl)

    # ── Metrics ──
    n_trades = len(trades)
    if n_trades < 5:
        return None

    wins = sum(1 for t in trades if t > 0)
    wr = wins / n_trades * 100
    ret_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Max drawdown
    peak = INITIAL_CAPITAL; max_dd = 0; eq = INITIAL_CAPITAL
    for t in trades:
        eq += t
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    g_win = sum(t for t in trades if t > 0)
    g_loss = abs(sum(t for t in trades if t < 0))
    pf = round(g_win / g_loss, 2) if g_loss > 0 else 999.99

    # Trades per day (calendar)
    first_t = symbols_data[next(iter(symbols_data))]["df"].iloc[warmup]["time_paris"]
    last_t = symbols_data[next(iter(symbols_data))]["df"].iloc[-1]["time_paris"]
    total_days = max(1, (last_t - first_t).days)
    tpd = n_trades / total_days

    # Average trade
    avg_t = sum(trades) / n_trades

    return {
        "trades": n_trades,
        "wins": wins,
        "wr": round(wr, 1),
        "ret_pct": round(ret_pct, 1),
        "equity": round(equity, 2),
        "max_dd": round(max_dd, 1),
        "trades_per_day": round(tpd, 2),
        "profit_factor": min(pf, 999.99),
        "avg_trade": round(avg_t, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING
# ═══════════════════════════════════════════════════════════════════════

_worker_data = None


def _init_worker(data):
    """Called once per worker process — stores shared data."""
    global _worker_data
    _worker_data = data


def _worker_task(args):
    """Single backtest task executed by a worker."""
    idx, cfg = args
    global _worker_data

    tf_data = _worker_data.get(cfg["timeframe"])
    if not tf_data:
        return idx, cfg, None

    try:
        symbols_data = {}
        for symbol, sdata in tf_data.items():
            df = compute_indicators(sdata["df"], cfg)
            symbols_data[symbol] = {"df": df, "specs": sdata["specs"]}

        if not symbols_data:
            return idx, cfg, None

        result = run_backtest(cfg, symbols_data)
    except Exception as e:
        result = None

    return idx, cfg, result


# ═══════════════════════════════════════════════════════════════════════
#  COMBINATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def generate_combinations(n):
    """Generate n unique random parameter combinations."""
    combos = []
    seen = set()
    max_attempts = n * 20

    for _ in range(max_attempts):
        if len(combos) >= n:
            break

        cfg = {k: random.choice(v) for k, v in PARAM_SPACE.items()}

        # Validity constraints
        if cfg["rsi_long_lo"] >= cfg["rsi_long_hi"]:
            continue
        if cfg["rsi_short_lo"] >= cfg["rsi_short_hi"]:
            continue
        if cfg["session_start_h"] >= cfg["session_end_h"]:
            continue

        # Deduplicate
        key = str(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        combos.append(cfg)

    return combos


# ═══════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def _print_table(df_top, max_rows=50):
    """Pretty-print a table of top results."""
    for i, row in enumerate(df_top.head(max_rows).itertuples(), 1):
        print(
            f"\n  #{i:<3}"
            f" Ret:{row.ret_pct:>+10.1f}%"
            f"  WR:{row.wr:>5.1f}%"
            f"  Trades:{row.trades:>5} ({row.trades_per_day:.1f}/j)"
            f"  DD:{row.max_dd:>5.1f}%"
            f"  PF:{row.profit_factor:>6.2f}"
            f"  Equity:${row.equity:>14,.0f}"
        )
        print(
            f"       TF={row.timeframe:<3}"
            f"  BB({row.bb_length},{row.bb_std})"
            f"  Vol={'OFF' if row.vol_mult == 0 else row.vol_mult}"
            f"  EMA={'OFF' if row.ema_length == 0 else row.ema_length}"
            f"  RSI({row.rsi_length}) L=[{row.rsi_long_lo}-{row.rsi_long_hi}]"
            f" S=[{row.rsi_short_lo}-{row.rsi_short_hi}]"
        )
        print(
            f"       SL={row.atr_sl_mult}ATR"
            f"  RR={row.risk_reward}:1"
            f"  Risk={row.risk_pct * 100:.1f}%"
            f"  Trail={row.trail_dist}@{row.trail_activate_r}R"
            f"  Session={row.session_start_h}h-{row.session_end_h}h"
            f"  Days={row.trading_days}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 100)
    print("  STRATEGY OPTIMIZER")
    print(f"  {N_COMBINATIONS:,} combinaisons | {len(SYMBOLS)} symboles | "
          f"{len(TIMEFRAMES)} timeframes | {BACKTEST_MONTHS} mois")
    print("=" * 100)

    # ── 1. Fetch data ──
    print("\n[1/4] Recuperation des donnees MT5 pour tous les timeframes...")
    all_data = fetch_all_data()
    total_candles = sum(
        len(sdata["df"])
        for tf_data in all_data.values()
        for sdata in tf_data.values()
    )
    print(f"  Total: {total_candles:,} candles chargees")

    # ── 2. Generate combinations ──
    print(f"\n[2/4] Generation de {N_COMBINATIONS:,} combinaisons aleatoires...")
    random.seed(42)  # Reproductible
    combos = generate_combinations(N_COMBINATIONS)
    # Insert v6 reference as idx=0
    combos.insert(0, V6_REFERENCE.copy())
    print(f"  {len(combos):,} combinaisons uniques "
          f"(1 ref v6 + {len(combos) - 1:,} aleatoires)")

    # ── 3. Run backtests in parallel ──
    n_workers = max(2, cpu_count() - 1)
    print(f"\n[3/4] Lancement des backtests sur {n_workers} workers...")

    tasks = list(enumerate(combos))
    results = []
    t0 = _time.time()

    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(all_data,)) as pool:
        for done, (idx, cfg, res) in enumerate(
            pool.imap_unordered(_worker_task, tasks, chunksize=25), 1
        ):
            if res is not None:
                results.append({"idx": idx, **cfg, **res})

            if done % 500 == 0 or done == len(tasks):
                elapsed = _time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                valid_pct = len(results) / done * 100
                print(
                    f"  [{done:>6}/{len(tasks)}]"
                    f"  {rate:>5.0f} tests/s"
                    f"  ETA {eta:>5.0f}s"
                    f"  |  valides: {len(results)} ({valid_pct:.0f}%)"
                )

    elapsed = _time.time() - t0
    print(f"\n  Termine en {elapsed:.1f}s "
          f"({len(tasks) / elapsed:.0f} tests/s)")
    print(f"  Resultats valides: {len(results):,} / {len(tasks):,}")

    if not results:
        print("\n  Aucun resultat valide !")
        return

    # ── 4. Analyze & save ──
    print(f"\n[4/4] Analyse des resultats...")

    df = pd.DataFrame(results)

    # Save all results
    out_dir = os.path.dirname(__file__)
    full_path = os.path.join(out_dir, "optimizer_results.csv")
    df.to_csv(full_path, index=False)
    print(f"  Resultats complets: {full_path}")

    # ── Reference v6 ──
    v6 = df[df["idx"] == 0]
    if not v6.empty:
        v = v6.iloc[0]
        print(f"\n{'=' * 100}")
        print(f"  REFERENCE — Config v6 actuelle (en production)")
        print(f"{'=' * 100}")
        print(
            f"  Trades: {v['trades']:<6}"
            f"  WR: {v['wr']:.1f}%"
            f"  Ret: {v['ret_pct']:+.1f}%"
            f"  DD: {v['max_dd']:.1f}%"
            f"  Trades/j: {v['trades_per_day']:.2f}"
            f"  PF: {v['profit_factor']:.2f}"
            f"  Equity: ${v['equity']:,.0f}"
        )

    # ── Filter: >= 0.8 trades/day ──
    df_freq = df[df["trades_per_day"] >= 0.8].copy()

    # ── TOP 50 by return % (with frequency >= 0.8/day) ──
    print(f"\n{'=' * 100}")
    print(f"  TOP 50 — Meilleur rendement"
          f"  (filtre: >= 0.8 trades/jour, {len(df_freq)} configs)")
    print(f"{'=' * 100}")

    if len(df_freq) > 0:
        top_ret = df_freq.nlargest(50, "ret_pct")
        _print_table(top_ret, 50)

        # Save top 50
        top_path = os.path.join(out_dir, "optimizer_top50.csv")
        top_ret.to_csv(top_path, index=False)
        print(f"\n  Top 50 sauvegarde: {top_path}")
    else:
        print("  Aucune config avec >= 0.8 trades/jour !")
        top_ret = pd.DataFrame()

    # ── TOP 30 balanced score ──
    #  score = rendement × min(freq, 3) / max(drawdown, 1)
    if len(df_freq) > 0:
        df_freq["score"] = (
            df_freq["ret_pct"].clip(lower=0)
            * df_freq["trades_per_day"].clip(upper=3)
            / df_freq["max_dd"].clip(lower=1)
        )

        print(f"\n{'=' * 100}")
        print(f"  TOP 30 — Score equilibre"
              f"  (rendement x frequence / drawdown)")
        print(f"{'=' * 100}")
        top_bal = df_freq.nlargest(30, "score")
        _print_table(top_bal, 30)

    # ── TOP 20 low drawdown + profitable ──
    df_profitable = df[(df["ret_pct"] > 0) & (df["trades_per_day"] >= 0.5)].copy()
    if len(df_profitable) > 0:
        print(f"\n{'=' * 100}")
        print(f"  TOP 20 — Drawdown le plus faible"
              f"  (rentable + >= 0.5 trades/jour, {len(df_profitable)} configs)")
        print(f"{'=' * 100}")
        top_dd = df_profitable.nsmallest(20, "max_dd")
        _print_table(top_dd, 20)

    # ── SUMMARY ──
    print(f"\n{'=' * 100}")
    print(f"  RESUME GLOBAL")
    print(f"{'=' * 100}")
    print(f"  Configs testees:           {len(df):>7,}")
    print(f"  Configs rentables:         {len(df[df['ret_pct'] > 0]):>7,}"
          f"  ({len(df[df['ret_pct'] > 0]) / len(df) * 100:.0f}%)")
    print(f"  Configs >= 0.5 trade/j:    {len(df[df['trades_per_day'] >= 0.5]):>7,}")
    print(f"  Configs >= 0.8 trade/j:    {len(df_freq):>7,}")
    print(f"  Configs >= 1.0 trade/j:    "
          f"{len(df[df['trades_per_day'] >= 1.0]):>7,}")
    print(f"  Configs >= 2.0 trades/j:   "
          f"{len(df[df['trades_per_day'] >= 2.0]):>7,}")
    print(f"  ---")
    print(f"  Meilleur rendement:        {df['ret_pct'].max():>+12.1f}%")
    print(f"  Rendement median:          {df['ret_pct'].median():>+12.1f}%")
    print(f"  Pire drawdown:             {df['max_dd'].max():>10.1f}%")
    print(f"  Moy. trades/jour:          {df['trades_per_day'].mean():>10.2f}")
    print(f"  Max trades/jour:           {df['trades_per_day'].max():>10.2f}")

    # ── Best per timeframe ──
    print(f"\n  --- Meilleur rendement par timeframe (>= 0.8 t/j) ---")
    if len(df_freq) > 0:
        for tf in ["M5", "M15", "M30", "H1"]:
            tf_df = df_freq[df_freq["timeframe"] == tf]
            if tf_df.empty:
                print(f"  {tf:<4}: aucune config qualifiee")
            else:
                best = tf_df.loc[tf_df["ret_pct"].idxmax()]
                print(f"  {tf:<4}: Ret {best['ret_pct']:>+10.1f}%"
                      f"  WR {best['wr']:.1f}%"
                      f"  {best['trades']} trades ({best['trades_per_day']:.1f}/j)"
                      f"  DD {best['max_dd']:.1f}%")

    print(f"\n{'=' * 100}")
    print(f"  Optimisation terminee.")
    print(f"  Fichiers: optimizer_results.csv | optimizer_top50.csv")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    freeze_support()
    main()
