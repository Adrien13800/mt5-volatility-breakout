"""
Breakout de Volatilité v3 — Bot MT5 automatisé
EMA trend filter, ATR-based SL/TP (capped), RSI confirmation,
Long + Short, trailing progressif.
"""

import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

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

MAGIC_NUMBER = 847291
RISK_PCT = 0.02

SESSION_START_H = 15
SESSION_START_M = 30
SESSION_END_H = 22
SESSION_END_M = 0

SCAN_INTERVAL = 15

MT5_LOGIN = int(os.environ["MT5_LOGIN"])
MT5_PASSWORD = os.environ["MT5_PASSWORD"]
MT5_SERVER = os.environ["MT5_SERVER"]

PARIS_TZ = pytz.timezone("Europe/Paris")

# ─────────────────────────── LOGGING ─────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# --- ANSI colors for terminal ---
class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"

class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG:    _C.DIM,
        logging.INFO:     _C.CYAN,
        logging.WARNING:  _C.YELLOW,
        logging.ERROR:    _C.RED + _C.BOLD,
        logging.CRITICAL: _C.BG_RED + _C.WHITE + _C.BOLD,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, _C.RESET)
        ts = self.formatTime(record, self.datefmt)
        level = record.levelname.ljust(8)
        msg = record.getMessage()
        return f"{_C.DIM}{ts}{_C.RESET} {color}{level}{_C.RESET} {msg}"

# Console handler (colored)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

logging.basicConfig(level=logging.DEBUG, handlers=[_console])
log = logging.getLogger("BreakoutBot")

# General log file (all messages)
_file_all = RotatingFileHandler(
    "breakout_bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
_file_all.setLevel(logging.DEBUG)
_file_all.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
log.addHandler(_file_all)

# Trade-only log file
_trade_formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt=LOG_DATEFMT)
_file_trades = RotatingFileHandler(
    "trades.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
_file_trades.setFormatter(_trade_formatter)
_file_trades.setLevel(logging.INFO)
trade_log = logging.getLogger("BreakoutBot.trades")
trade_log.addHandler(_file_trades)
trade_log.propagate = False  # don't duplicate into main log

# Error-only log file
_file_errors = RotatingFileHandler(
    "errors.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
)
_file_errors.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
_file_errors.setLevel(logging.ERROR)
log.addHandler(_file_errors)

# ─────────────────────────── STATE ───────────────────────────────────

tracked_positions: dict[str, dict] = {}

# ─────────────────────────── MT5 HELPERS ─────────────────────────────


def connect_mt5() -> bool:
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error("Échec d'initialisation MT5 : %s", mt5.last_error())
        return False

    account = mt5.account_info()
    if account is None:
        log.error("MT5 initialisé mais account_info() retourne None.")
        return False

    log.info(
        "MT5 connecté — compte #%d (%s) | solde=%.2f %s | serveur=%s",
        account.login,
        account.name,
        account.balance,
        account.currency,
        account.server,
    )
    return True


def ensure_connected() -> bool:
    account = mt5.account_info()
    if account is not None:
        return True
    log.warning("Connexion MT5 perdue, tentative de reconnexion...")
    mt5.shutdown()
    for attempt in range(3):
        if connect_mt5():
            return True
        time.sleep(5 * (attempt + 1))
    log.critical("Impossible de reconnecter MT5 après 3 tentatives.")
    return False


def disconnect_mt5():
    mt5.shutdown()
    log.info("MT5 déconnecté proprement.")


def is_symbol_available(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        log.warning("Symbole %s introuvable sur le broker.", symbol)
        return False
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            log.warning("Impossible d'activer le symbole %s.", symbol)
            return False
    return True


# ─────────────────────────── DONNÉES ─────────────────────────────────


def get_candles(symbol: str, count: int = 200) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, count)
    if rates is None or len(rates) == 0:
        log.warning("Aucune donnée reçue pour %s.", symbol)
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    bb = ta.bbands(df["close"], length=BB_LENGTH, std=BB_STD)
    if bb is None:
        return None
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
    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_LENGTH)
    if atr is None:
        return None
    df["atr"] = atr
    atr_median = df["atr"].rolling(ATR_MEDIAN_WINDOW, min_periods=1).median()
    df["atr_capped"] = df["atr"].clip(upper=atr_median * ATR_CAP_MULT)
    return df


# ─────────────────────────── SESSION ─────────────────────────────────


def in_trading_session() -> bool:
    now = datetime.now(PARIS_TZ)
    if now.weekday() >= 5:
        return False
    start = now.replace(
        hour=SESSION_START_H, minute=SESSION_START_M, second=0, microsecond=0
    )
    end = now.replace(hour=SESSION_END_H, minute=SESSION_END_M, second=0, microsecond=0)
    return start <= now <= end


# ─────────────────────────── POSITIONS ───────────────────────────────


def get_my_position(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    matches = [p for p in positions if p.magic == MAGIC_NUMBER]
    if len(matches) > 1:
        log.warning(
            "%s — %d positions avec magic %d détectées, gestion de la première.",
            symbol, len(matches), MAGIC_NUMBER,
        )
    return matches[0] if matches else None


def calculate_lot_size(symbol: str, sl_dist: float) -> float | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    account = mt5.account_info()
    if account is None:
        return None

    risk_amount = account.equity * RISK_PCT
    sl_value_per_lot = (sl_dist / info.trade_tick_size) * info.trade_tick_value
    if sl_value_per_lot <= 0:
        return info.volume_min

    lot = risk_amount / sl_value_per_lot
    step = info.volume_step
    lot = round(lot / step) * step
    lot = max(info.volume_min, min(info.volume_max, lot))
    return round(lot, 6)


def open_order(symbol: str, direction: str, sl_dist: float, tp_dist: float) -> float | None:
    """Ouvre un ordre avec lot sizing dynamique. Retourne le prix de fill ou None."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Tick indisponible pour %s, ordre annulé.", symbol)
        return None

    lot_size = calculate_lot_size(symbol, sl_dist)
    if lot_size is None:
        log.error("Impossible de calculer le lot pour %s.", symbol)
        return None

    if direction == "long":
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        sl = round(price - sl_dist, info.digits)
        tp = round(price + tp_dist, info.digits)
        comment = "Breakout Bot — Long"
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        sl = round(price + sl_dist, info.digits)
        tp = round(price - tp_dist, info.digits)
        comment = "Breakout Bot — Short"

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        comment = result.comment if result else "no response"
        log.error(
            "Ordre %s échoué sur %s | retcode=%s | %s",
            direction.upper(), symbol, retcode, comment,
        )
        return None

    fill_price = result.price if result.price else price
    tag = f"{_C.BG_GREEN}{_C.WHITE} TRADE " if direction == "long" else f"{_C.BG_RED}{_C.WHITE} TRADE "
    log.info(
        "%s %s %s @ %.5f | lot=%.3f | SL=%.5f | TP=%.5f | ticket=%d%s",
        tag, direction.upper(), symbol, fill_price, lot_size, sl, tp, result.order, _C.RESET,
    )
    trade_log.info(
        "OPEN %s %s @ %.5f | lot=%.3f | SL=%.5f | TP=%.5f | ticket=%d",
        direction.upper(), symbol, fill_price, lot_size, sl, tp, result.order,
    )
    return fill_price


def modify_sl(symbol: str, ticket: int, new_sl: float, current_tp: float) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        return False

    new_sl = round(new_sl, info.digits)
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": current_tp,
        "magic": MAGIC_NUMBER,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Modification SL échouée ticket %d : %s", ticket, result)
        return False

    log.info(
        "%s TRAIL %s ticket=%d | nouveau SL=%.5f%s",
        _C.MAGENTA, symbol, ticket, new_sl, _C.RESET,
    )
    trade_log.info("TRAIL %s ticket=%d | nouveau SL=%.5f", symbol, ticket, new_sl)
    return True


# ─────────────────────────── LOGIQUE STRATÉGIE ───────────────────────


def process_symbol(symbol: str):
    if not is_symbol_available(symbol):
        return

    df = get_candles(symbol)
    if df is None:
        return

    df = compute_indicators(df)
    if df is None:
        log.warning("%s — échec calcul indicateurs.", symbol)
        return

    if len(df) < 2:
        log.warning("%s — pas assez de bougies.", symbol)
        return

    prev = df.iloc[-2]

    close = prev["close"]
    bb_upper = prev.get("BBU")
    bb_lower = prev.get("BBL")
    tick_vol = prev["tick_volume"]
    vol_sma = prev.get("vol_sma")
    ema = prev.get("ema_trend")
    rsi = prev.get("rsi")
    atr_capped = prev.get("atr_capped")

    if any(pd.isna(v) for v in (bb_upper, bb_lower, vol_sma, ema, rsi, atr_capped)):
        log.warning("%s — indicateurs non calculés (données insuffisantes).", symbol)
        return

    if atr_capped == 0:
        return

    vol_threshold = VOL_MULTIPLIER * vol_sma
    pos = get_my_position(symbol)

    # ── Log indicateurs à chaque scan (fichier uniquement) ──
    candle_time = prev["time"]
    log.debug(
        "%s | candle=%s | close=%.2f | BB=[%.2f, %.2f] | EMA=%.2f | RSI=%.1f | ATR=%.2f | vol=%d/%.0f",
        symbol, candle_time, close, bb_lower, bb_upper, ema, rsi, atr_capped,
        int(tick_vol), vol_threshold,
    )

    # ── Trailing stop sur position ouverte ──
    if pos is not None:
        log.info(
            "%s — position ouverte détectée (ticket=%d, type=%s), signal ignoré.",
            symbol, pos.ticket, "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT",
        )
        tracking = tracked_positions.get(symbol)
        if tracking is None or tracking.get("ticket") != pos.ticket:
            tracked_positions[symbol] = {
                "ticket": pos.ticket,
                "best_price": pos.price_open,
                "atr_entry": atr_capped,
            }
            tracking = tracked_positions[symbol]

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log.warning("Tick indisponible pour %s, trailing ignoré.", symbol)
            return

        atr_e = tracking["atr_entry"]
        is_long = pos.type == mt5.ORDER_TYPE_BUY

        if is_long:
            current_price = tick.bid
            if current_price > tracking["best_price"]:
                tracking["best_price"] = current_price

            profit_dist = tracking["best_price"] - pos.price_open
            if profit_dist >= ATR_TRAIL_ACTIVATE * atr_e:
                new_sl = tracking["best_price"] - ATR_TRAIL_DIST * atr_e
                if new_sl > pos.sl:
                    modify_sl(symbol, pos.ticket, new_sl, pos.tp)
        else:
            current_price = tick.ask
            if current_price < tracking["best_price"]:
                tracking["best_price"] = current_price

            profit_dist = pos.price_open - tracking["best_price"]
            if profit_dist >= ATR_TRAIL_ACTIVATE * atr_e:
                new_sl = tracking["best_price"] + ATR_TRAIL_DIST * atr_e
                if new_sl < pos.sl:
                    modify_sl(symbol, pos.ticket, new_sl, pos.tp)

        return

    tracked_positions.pop(symbol, None)

    # ── Filtre session américaine ──
    if not in_trading_session():
        return

    vol_icon = f"{_C.GREEN}✓{_C.RESET}" if tick_vol > vol_threshold else f"{_C.RED}✗{_C.RESET}"
    rsi_color = _C.GREEN if 50 < rsi < 70 else (_C.RED if 30 < rsi < 50 else _C.YELLOW)
    print(
        f"  {_C.BOLD}{symbol:<10}{_C.RESET}"
        f"  close {_C.WHITE}{close:>10.2f}{_C.RESET}"
        f"  │ BB [{bb_lower:.2f} — {bb_upper:.2f}]"
        f"  │ EMA {ema:.2f}"
        f"  │ RSI {rsi_color}{rsi:5.1f}{_C.RESET}"
        f"  │ ATR {atr_capped:.2f}"
        f"  │ vol {int(tick_vol):>5}/{int(vol_threshold):<5} {vol_icon}"
    )

    # ── Conditions individuelles ──
    bb_long = close > bb_upper
    bb_short = close < bb_lower
    ema_long = close > ema
    ema_short = close < ema
    vol_ok = tick_vol > vol_threshold
    rsi_long = 50 < rsi < 70
    rsi_short = 30 < rsi < 50

    # ── Signal Long ──
    if bb_long and ema_long and vol_ok and rsi_long:
        print(f"  {_C.BG_GREEN}{_C.WHITE}{_C.BOLD} ▲ LONG  {_C.RESET} {symbol} — breakout BB + EMA + RSI + volume")
        trade_log.info("SIGNAL LONG %s | close=%.2f BB_up=%.2f EMA=%.2f RSI=%.1f vol=%d", symbol, close, bb_upper, ema, rsi, int(tick_vol))
        sl_dist = ATR_SL_MULT * atr_capped
        tp_dist = ATR_TP_MULT * atr_capped
        fill_price = open_order(symbol, "long", sl_dist, tp_dist)
        if fill_price is not None:
            new_pos = get_my_position(symbol)
            tracked_positions[symbol] = {
                "ticket": new_pos.ticket if new_pos else 0,
                "best_price": fill_price,
                "atr_entry": atr_capped,
            }

    # ── Signal Short ──
    elif bb_short and ema_short and vol_ok and rsi_short:
        print(f"  {_C.BG_RED}{_C.WHITE}{_C.BOLD} ▼ SHORT {_C.RESET} {symbol} — breakdown BB + EMA + RSI + volume")
        trade_log.info("SIGNAL SHORT %s | close=%.2f BB_lo=%.2f EMA=%.2f RSI=%.1f vol=%d", symbol, close, bb_lower, ema, rsi, int(tick_vol))
        sl_dist = ATR_SL_MULT * atr_capped
        tp_dist = ATR_TP_MULT * atr_capped
        fill_price = open_order(symbol, "short", sl_dist, tp_dist)
        if fill_price is not None:
            new_pos = get_my_position(symbol)
            tracked_positions[symbol] = {
                "ticket": new_pos.ticket if new_pos else 0,
                "best_price": fill_price,
                "atr_entry": atr_capped,
            }

    # ── Aucun signal — détail des conditions ──
    else:
        failed = []
        if not bb_long and not bb_short:
            failed.append("BB")
        if not vol_ok:
            failed.append(f"Vol({int(tick_vol)}/{int(vol_threshold)})")
        if not rsi_long and not rsi_short:
            failed.append(f"RSI({rsi:.1f})")
        if bb_long and not ema_long:
            failed.append("EMA↓")
        if bb_short and not ema_short:
            failed.append("EMA↑")
        missing = ", ".join(failed) if failed else "aucune condition proche"
        print(f"  {_C.DIM}  ─ {symbol:<10} pas de signal : {missing}{_C.RESET}")

        log.info(
            "NO_SIGNAL %s | candle=%s | close=%.2f | BB_lo=%.2f BB_up=%.2f | EMA=%.2f | RSI=%.1f "
            "| vol=%d/%.0f | conditions: BB_L=%s BB_S=%s EMA_L=%s EMA_S=%s VOL=%s RSI_L=%s RSI_S=%s",
            symbol, candle_time, close, bb_lower, bb_upper, ema, rsi,
            int(tick_vol), vol_threshold,
            bb_long, bb_short, ema_long, ema_short, vol_ok, rsi_long, rsi_short,
        )


def log_account_status():
    account = mt5.account_info()
    if account is None:
        return
    pnl = account.profit
    pnl_color = _C.GREEN if pnl >= 0 else _C.RED
    print(
        f"\n{_C.BOLD}{_C.YELLOW}{'─' * 70}\n"
        f"  COMPTE  │  Solde: {account.balance:>10.2f} $  │  Equity: {account.equity:>10.2f} $\n"
        f"          │  Marge libre: {account.margin_free:>10.2f} $  │  "
        f"P&L: {pnl_color}{pnl:>+10.2f} ${_C.RESET}\n"
        f"{_C.YELLOW}{'─' * 70}{_C.RESET}"
    )
    log.info(
        "COMPTE | Solde=%.2f $ | Equity=%.2f $ | Marge libre=%.2f $ | P&L flottant=%.2f $",
        account.balance, account.equity, account.margin_free, pnl,
    )


# ─────────────────────────── BOUCLE PRINCIPALE ───────────────────────


def main():
    if not connect_mt5():
        return

    print(
        f"\n{_C.BOLD}{_C.GREEN}"
        f"╔══════════════════════════════════════════════════════════╗\n"
        f"║         BREAKOUT BOT v3 — Volatilité MT5               ║\n"
        f"╚══════════════════════════════════════════════════════════╝{_C.RESET}\n"
        f"  Symboles : {', '.join(SYMBOLS.values())}\n"
        f"  Timeframe: M15  │  Risk/Reward: {RISK_REWARD}  │  Scan: {SCAN_INTERVAL}s\n"
        f"  Session  : {SESSION_START_H}:{SESSION_START_M:02d} — {SESSION_END_H}:{SESSION_END_M:02d} (Paris)\n"
    )
    log.info(
        "Démarrage du bot v3 — Symboles : %s | TF : M15 | R = %d | Scan toutes les %ds",
        list(SYMBOLS.values()), RISK_REWARD, SCAN_INTERVAL,
    )

    scan_count = 0

    try:
        while True:
            scan_start = time.monotonic()

            try:
                if not ensure_connected():
                    log.error("MT5 indisponible, attente avant retry...")
                    time.sleep(30)
                    continue

                now_paris = datetime.now(PARIS_TZ).strftime("%H:%M:%S")
                session_ok = in_trading_session()
                session_tag = f"{_C.GREEN}● ACTIVE{_C.RESET}" if session_ok else f"{_C.RED}○ FERMÉE{_C.RESET}"
                print(
                    f"\n{_C.BOLD}{_C.CYAN}{'═' * 70}\n"
                    f"  SCAN  {now_paris} (Paris)   │   Session US : {session_tag}\n"
                    f"{_C.CYAN}{'═' * 70}{_C.RESET}"
                )

                for symbol in SYMBOLS.values():
                    try:
                        process_symbol(symbol)
                    except Exception as exc:
                        log.exception("Erreur sur %s : %s", symbol, exc)

                scan_count += 1
                if scan_count % 20 == 0:
                    log_account_status()

            except Exception as exc:
                log.exception("Erreur globale dans la boucle : %s", exc)

            elapsed = time.monotonic() - scan_start
            time.sleep(max(0, SCAN_INTERVAL - elapsed))

    except KeyboardInterrupt:
        log.info("Arrêt demandé par l'utilisateur (Ctrl+C).")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
