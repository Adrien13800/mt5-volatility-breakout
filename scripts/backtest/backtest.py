"""
Backtest — Breakout de Volatilité v3
Lot sizing dynamique (2% risque/trade), multi-symbole bar-par-bar,
EMA trend filter, ATR-based SL/TP capped, RSI, trailing progressif.
"""

import logging
import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import pytz
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── CONFIGURATION ───────────────────────────

SYMBOLS = {
    "NAS100": "NAS100.",
    "US30": "DJ30.",
    "SPX500": "SP500.",
}

TIMEFRAME = mt5.TIMEFRAME_M15
BB_LENGTH = 20
BB_STD = 2.0
VOL_SMA_LENGTH = 20
VOL_MULTIPLIER = 1.3

EMA_TREND_LENGTH = 50
RSI_LENGTH = 14
ATR_LENGTH = 14

RISK_REWARD = 3

ATR_SL_MULT = 1.5
ATR_TP_MULT = ATR_SL_MULT * RISK_REWARD
ATR_TRAIL_ACTIVATE = ATR_TP_MULT * 0.5
ATR_TRAIL_DIST = ATR_SL_MULT
ATR_MEDIAN_WINDOW = 100
ATR_CAP_MULT = 1.5

RISK_PCT = 0.02
BACKTEST_MONTHS = 8

SESSION_START_H = 15
SESSION_START_M = 30
SESSION_END_H = 22
SESSION_END_M = 0

PARIS_TZ = pytz.timezone("Europe/Paris")

MT5_LOGIN = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER = os.environ["MT5_SERVER"]

INITIAL_CAPITAL = 132

# ─────────────────────────── LOGGING ─────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("Backtest")

# ─────────────────────────── MT5 ─────────────────────────────────────


def connect_mt5() -> bool:
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error("Échec d'initialisation MT5 : %s", mt5.last_error())
        return False
    log.info("MT5 connecté pour backtest — version %s", mt5.version())
    return True


def fetch_historical(symbol: str, months: int) -> pd.DataFrame | None:
    utc_to = datetime.now(tz=pytz.utc)
    utc_from = utc_to - timedelta(days=months * 30)

    rates = mt5.copy_rates_range(symbol, TIMEFRAME, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        log.warning("Aucune donnée historique pour %s.", symbol)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_paris"] = df["time"].dt.tz_convert(PARIS_TZ)
    return df


def get_symbol_specs(symbol: str) -> dict:
    info = mt5.symbol_info(symbol)
    if info is None:
        log.warning("Specs indisponibles pour %s, valeurs par défaut.", symbol)
        return {
            "point": 0.01,
            "tick_value": 1.0,
            "tick_size": 0.01,
            "vol_min": 0.01,
            "vol_max": 100.0,
            "vol_step": 0.01,
        }
    return {
        "point": info.point,
        "tick_value": info.trade_tick_value,
        "tick_size": info.trade_tick_size,
        "vol_min": info.volume_min,
        "vol_max": info.volume_max,
        "vol_step": info.volume_step,
    }


# ─────────────────────────── INDICATEURS ─────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=BB_STD)
    col_map = {}
    for c in bb.columns:
        if c.startswith("BBU"):
            col_map[c] = "BBU"
        elif c.startswith("BBM"):
            col_map[c] = "BBM"
        elif c.startswith("BBL"):
            col_map[c] = "BBL"
    bb = bb.rename(columns=col_map)
    df = pd.concat([df, bb], axis=1)

    df["vol_sma"] = ta.sma(df["tick_volume"].astype(float), length=VOL_SMA_LENGTH)
    df["ema_trend"] = ta.ema(df["close"], length=EMA_TREND_LENGTH)
    df["rsi"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH)
    atr_median = df["atr"].rolling(ATR_MEDIAN_WINDOW, min_periods=1).median()
    df["atr_capped"] = df["atr"].clip(upper=atr_median * ATR_CAP_MULT)
    return df


# ─────────────────────────── FILTRE SESSION ──────────────────────────


def is_in_session(dt_paris: datetime) -> bool:
    start = dt_paris.replace(
        hour=SESSION_START_H, minute=SESSION_START_M, second=0, microsecond=0
    )
    end = dt_paris.replace(
        hour=SESSION_END_H, minute=SESSION_END_M, second=0, microsecond=0
    )
    return start <= dt_paris <= end


# ─────────────────────────── LOT SIZING ──────────────────────────────


def calculate_lot_size(equity: float, atr_capped: float, specs: dict) -> float:
    risk_amount = equity * RISK_PCT
    sl_distance = ATR_SL_MULT * atr_capped
    sl_value_per_lot = (sl_distance / specs["tick_size"]) * specs["tick_value"]
    if sl_value_per_lot <= 0:
        return specs["vol_min"]
    lot = risk_amount / sl_value_per_lot
    step = specs["vol_step"]
    lot = round(lot / step) * step
    lot = max(specs["vol_min"], min(specs["vol_max"], lot))
    return round(lot, 6)


# ─────────────────────────── MOTEUR DE BACKTEST ──────────────────────


def run_backtest(symbols_data: dict) -> list[dict]:
    equity = INITIAL_CAPITAL
    trades: list[dict] = []
    positions: dict[str, dict] = {}
    pending_signals: dict[str, dict] = {}

    n_bars = min(len(cfg["df"]) for cfg in symbols_data.values())
    warmup = max(
        BB_LENGTH + VOL_SMA_LENGTH,
        EMA_TREND_LENGTH,
        RSI_LENGTH,
        ATR_LENGTH,
        ATR_MEDIAN_WINDOW,
    )

    for i in range(warmup, n_bars):
        for symbol, cfg in symbols_data.items():
            row = cfg["df"].iloc[i]
            specs = cfg["specs"]
            point = specs["point"]

            close = row["close"]
            open_price = row["open"]
            high = row["high"]
            low = row["low"]
            bb_upper = row.get("BBU")
            bb_lower = row.get("BBL")
            tick_vol = row["tick_volume"]
            vol_sma = row.get("vol_sma")
            ema = row.get("ema_trend")
            rsi = row.get("rsi")
            atr_capped = row.get("atr_capped")
            t_paris = row["time_paris"]

            if any(
                pd.isna(v) for v in (bb_upper, bb_lower, vol_sma, ema, rsi, atr_capped)
            ):
                pending_signals.pop(symbol, None)
                continue

            if atr_capped == 0:
                pending_signals.pop(symbol, None)
                continue

            # ── Entrée différée ──
            if symbol in pending_signals and symbol not in positions:
                sig = pending_signals.pop(symbol)
                entry_price = open_price
                atr_c = sig["atr_capped"]
                lot = calculate_lot_size(equity, atr_c, specs)

                if sig["direction"] == "long":
                    positions[symbol] = {
                        "direction": "long",
                        "entry_price": entry_price,
                        "sl": entry_price - ATR_SL_MULT * atr_c,
                        "tp": entry_price + ATR_TP_MULT * atr_c,
                        "entry_time": t_paris,
                        "atr_entry": atr_c,
                        "best_price": entry_price,
                        "lot_size": lot,
                    }
                else:
                    positions[symbol] = {
                        "direction": "short",
                        "entry_price": entry_price,
                        "sl": entry_price + ATR_SL_MULT * atr_c,
                        "tp": entry_price - ATR_TP_MULT * atr_c,
                        "entry_time": t_paris,
                        "atr_entry": atr_c,
                        "best_price": entry_price,
                        "lot_size": lot,
                    }

            # ── Gestion position ouverte ──
            if symbol in positions:
                pos = positions[symbol]
                direction = pos["direction"]
                atr_e = pos["atr_entry"]
                lot = pos["lot_size"]
                usd_per_pt = (specs["tick_value"] / specs["tick_size"]) * point * lot

                if direction == "long":
                    if low <= pos["sl"]:
                        pnl_pts = (pos["sl"] - pos["entry_price"]) / point
                        pnl_usd = round(pnl_pts * usd_per_pt, 2)
                        equity += pnl_usd
                        trades.append(
                            _rec(
                                symbol,
                                pos,
                                t_paris,
                                pos["sl"],
                                "SL",
                                pnl_pts,
                                pnl_usd,
                                lot,
                                equity,
                            )
                        )
                        del positions[symbol]
                        continue

                    if high >= pos["tp"]:
                        pnl_pts = (pos["tp"] - pos["entry_price"]) / point
                        pnl_usd = round(pnl_pts * usd_per_pt, 2)
                        equity += pnl_usd
                        trades.append(
                            _rec(
                                symbol,
                                pos,
                                t_paris,
                                pos["tp"],
                                "TP",
                                pnl_pts,
                                pnl_usd,
                                lot,
                                equity,
                            )
                        )
                        del positions[symbol]
                        continue

                    if high > pos["best_price"]:
                        pos["best_price"] = high
                    profit_dist = pos["best_price"] - pos["entry_price"]
                    if profit_dist >= ATR_TRAIL_ACTIVATE * atr_e:
                        new_sl = pos["best_price"] - ATR_TRAIL_DIST * atr_e
                        if new_sl > pos["sl"]:
                            pos["sl"] = new_sl

                else:
                    if high >= pos["sl"]:
                        pnl_pts = (pos["entry_price"] - pos["sl"]) / point
                        pnl_usd = round(pnl_pts * usd_per_pt, 2)
                        equity += pnl_usd
                        trades.append(
                            _rec(
                                symbol,
                                pos,
                                t_paris,
                                pos["sl"],
                                "SL",
                                pnl_pts,
                                pnl_usd,
                                lot,
                                equity,
                            )
                        )
                        del positions[symbol]
                        continue

                    if low <= pos["tp"]:
                        pnl_pts = (pos["entry_price"] - pos["tp"]) / point
                        pnl_usd = round(pnl_pts * usd_per_pt, 2)
                        equity += pnl_usd
                        trades.append(
                            _rec(
                                symbol,
                                pos,
                                t_paris,
                                pos["tp"],
                                "TP",
                                pnl_pts,
                                pnl_usd,
                                lot,
                                equity,
                            )
                        )
                        del positions[symbol]
                        continue

                    if low < pos["best_price"]:
                        pos["best_price"] = low
                    profit_dist = pos["entry_price"] - pos["best_price"]
                    if profit_dist >= ATR_TRAIL_ACTIVATE * atr_e:
                        new_sl = pos["best_price"] + ATR_TRAIL_DIST * atr_e
                        if new_sl < pos["sl"]:
                            pos["sl"] = new_sl

                continue

            # ── Filtre session US ──
            if not is_in_session(t_paris):
                continue

            vol_threshold = VOL_MULTIPLIER * vol_sma

            # ── Signal Long ──
            if (
                close > bb_upper
                and close > ema
                and tick_vol > vol_threshold
                and 50 < rsi < 70
            ):
                pending_signals[symbol] = {
                    "direction": "long",
                    "atr_capped": atr_capped,
                }

            # ── Signal Short ──
            elif (
                close < bb_lower
                and close < ema
                and tick_vol > vol_threshold
                and 30 < rsi < 50
            ):
                pending_signals[symbol] = {
                    "direction": "short",
                    "atr_capped": atr_capped,
                }
            else:
                pending_signals.pop(symbol, None)

    # Fermer les positions restantes
    for symbol, pos in list(positions.items()):
        cfg = symbols_data[symbol]
        specs = cfg["specs"]
        point = specs["point"]
        lot = pos["lot_size"]
        usd_per_pt = (specs["tick_value"] / specs["tick_size"]) * point * lot
        last = cfg["df"].iloc[-1]

        if pos["direction"] == "long":
            pnl_pts = (last["close"] - pos["entry_price"]) / point
        else:
            pnl_pts = (pos["entry_price"] - last["close"]) / point
        pnl_usd = round(pnl_pts * usd_per_pt, 2)
        equity += pnl_usd
        trades.append(
            _rec(
                symbol,
                pos,
                last["time_paris"],
                last["close"],
                "END",
                pnl_pts,
                pnl_usd,
                lot,
                equity,
            )
        )

    return trades


def _rec(symbol, pos, exit_time, exit_price, reason, pnl_pts, pnl_usd, lot, equity):
    return {
        "symbol": symbol,
        "direction": pos["direction"],
        "entry_time": pos["entry_time"],
        "entry_price": pos["entry_price"],
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": reason,
        "lot_size": lot,
        "pnl_points": round(pnl_pts, 1),
        "pnl_usd": pnl_usd,
        "equity": round(equity, 2),
    }


# ─────────────────────────── RAPPORT ─────────────────────────────────


def print_report(all_trades: list[dict]):
    if not all_trades:
        log.info("Aucun trade sur la période.")
        return

    df = pd.DataFrame(all_trades)

    separator = "=" * 80
    log.info(separator)
    log.info(
        "RAPPORT DE BACKTEST v3 — %d mois | Capital : %.2f $ | Risque : %.0f%%/trade | R = %d",
        BACKTEST_MONTHS,
        INITIAL_CAPITAL,
        RISK_PCT * 100,
        RISK_REWARD,
    )
    log.info(separator)

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].copy()
        n = len(sym_df)
        wins = sym_df[sym_df["pnl_usd"] > 0]
        losses = sym_df[sym_df["pnl_usd"] <= 0]
        total_pnl = sym_df["pnl_usd"].sum()
        win_rate = len(wins) / n * 100 if n > 0 else 0
        avg_win = wins["pnl_usd"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_usd"].mean() if len(losses) > 0 else 0
        best = sym_df["pnl_usd"].max()
        worst = sym_df["pnl_usd"].min()
        avg_lot = sym_df["lot_size"].mean()

        n_long = len(sym_df[sym_df["direction"] == "long"])
        n_short = len(sym_df[sym_df["direction"] == "short"])
        by_reason = sym_df["exit_reason"].value_counts().to_dict()

        log.info("-" * 80)
        log.info("  %s", symbol)
        log.info("-" * 80)
        log.info("  Trades          : %d  (Long: %d | Short: %d)", n, n_long, n_short)
        log.info("  Gagnants        : %d  |  Perdants : %d", len(wins), len(losses))
        log.info("  Win rate        : %.1f%%", win_rate)
        log.info("  PnL total       : %.2f $", total_pnl)
        log.info(
            "  Gain moyen      : %.2f $  |  Perte moyenne : %.2f $", avg_win, avg_loss
        )
        log.info("  Meilleur trade  : %.2f $  |  Pire trade    : %.2f $", best, worst)
        log.info("  Lot moyen       : %.3f", avg_lot)
        log.info("  Sorties         : %s", by_reason)

    log.info(separator)

    final_equity = df["equity"].iloc[-1]
    total_return = final_equity - INITIAL_CAPITAL
    total_return_pct = (total_return / INITIAL_CAPITAL) * 100

    peak_equity = INITIAL_CAPITAL
    max_dd_usd = 0
    max_dd_pct = 0
    for eq in df["equity"]:
        if eq > peak_equity:
            peak_equity = eq
        dd = peak_equity - eq
        if dd > max_dd_usd:
            max_dd_usd = dd
        dd_pct = (dd / peak_equity * 100) if peak_equity > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    total_trades = len(df)
    total_wins = len(df[df["pnl_usd"] > 0])
    global_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    log.info("  GLOBAL")
    log.info("  Capital initial : %.2f $", INITIAL_CAPITAL)
    log.info("  Capital final   : %.2f $", final_equity)
    log.info("  Rendement       : %.2f $ (%.1f%%)", total_return, total_return_pct)
    log.info("  Max drawdown    : %.2f $ (%.1f%%)", max_dd_usd, max_dd_pct)
    log.info("  Total trades    : %d", total_trades)
    log.info("  Win rate global : %.1f%%", global_wr)
    log.info(separator)

    csv_path = f"backtest_{RISK_REWARD}R.csv"
    df.to_csv(csv_path, index=False)
    log.info("Détail des trades exporté dans : %s", csv_path)


# ─────────────────────────── MAIN ────────────────────────────────────


def main():
    if not connect_mt5():
        return

    try:
        symbols_data = {}

        for label, symbol in SYMBOLS.items():
            log.info(
                "Chargement %s (%s) sur %d mois...", label, symbol, BACKTEST_MONTHS
            )

            if not mt5.symbol_select(symbol, True):
                log.warning("Symbole %s indisponible, ignoré.", symbol)
                continue

            df = fetch_historical(symbol, BACKTEST_MONTHS)
            if df is None:
                continue

            log.info(
                "  %d bougies M15 chargées (%s → %s).",
                len(df),
                df["time_paris"].iloc[0].strftime("%Y-%m-%d"),
                df["time_paris"].iloc[-1].strftime("%Y-%m-%d"),
            )

            df = compute_indicators(df)
            specs = get_symbol_specs(symbol)
            symbols_data[symbol] = {"df": df, "specs": specs}

        if not symbols_data:
            log.error("Aucun symbole chargé.")
            return

        trades = run_backtest(symbols_data)
        log.info("%d trades au total.", len(trades))
        print_report(trades)

    except Exception as exc:
        log.exception("Erreur pendant le backtest : %s", exc)
    finally:
        mt5.shutdown()
        log.info("MT5 déconnecté.")


if __name__ == "__main__":
    main()
