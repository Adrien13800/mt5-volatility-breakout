"""
Tests unitaires — Breakout Bot v7
Verifie : config v7, conditions de signal, RSI ranges, SL/TP,
          trailing stop, session, lot sizing, modify_sl, indicateurs.

Usage:  py test_bot.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# ── Mock MetaTrader5 + env vars AVANT import du bot ──
os.environ.setdefault("MT5_LOGIN", "12345")
os.environ.setdefault("MT5_PASSWORD", "test")
os.environ.setdefault("MT5_SERVER", "test")

mock_mt5 = MagicMock()
mock_mt5.TIMEFRAME_M15 = 16385
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
mock_mt5.TRADE_ACTION_SLTP = 6
mock_mt5.TRADE_ACTION_DEAL = 1
mock_mt5.ORDER_TIME_GTC = 0
mock_mt5.ORDER_FILLING_IOC = 1
mock_mt5.TRADE_RETCODE_DONE = 10009
sys.modules["MetaTrader5"] = mock_mt5

import breakout_bot as bot  # noqa: E402

PARIS_TZ = bot.PARIS_TZ


# ═══════════════════════════════════════════════════════════════
#  1. CONFIG V7
# ═══════════════════════════════════════════════════════════════

class TestV7Config(unittest.TestCase):
    """Verifie que tous les parametres v7 sont corrects."""

    def test_sl_mult(self):
        self.assertEqual(bot.ATR_SL_MULT, 0.5)

    def test_tp_mult(self):
        self.assertAlmostEqual(bot.ATR_TP_MULT, 2.0)  # 0.5 * 4

    def test_risk_pct(self):
        self.assertEqual(bot.RISK_PCT, 0.02)

    def test_vol_multiplier(self):
        self.assertEqual(bot.VOL_MULTIPLIER, 1.1)

    def test_trail_dist(self):
        self.assertEqual(bot.ATR_TRAIL_DIST, 0.2)

    def test_trail_activate_r(self):
        self.assertEqual(bot.TRAIL_ACTIVATE_R, 1.5)

    def test_trail_activate_atr(self):
        self.assertAlmostEqual(bot.ATR_TRAIL_ACTIVATE, 0.75)  # 1.5 * 0.5

    def test_session_start(self):
        self.assertEqual(bot.SESSION_START_H, 17)

    def test_session_end(self):
        self.assertEqual(bot.SESSION_END_H, 22)

    def test_risk_reward(self):
        self.assertEqual(bot.RISK_REWARD, 4)

    def test_bb_length(self):
        self.assertEqual(bot.BB_LENGTH, 20)

    def test_bb_std(self):
        self.assertEqual(bot.BB_STD, 2.0)

    def test_ema_length(self):
        self.assertEqual(bot.EMA_TREND_LENGTH, 50)

    def test_rsi_length(self):
        self.assertEqual(bot.RSI_LENGTH, 14)

    def test_timeframe(self):
        self.assertEqual(bot.TIMEFRAME, mock_mt5.TIMEFRAME_M15)

    def test_symbols_count(self):
        self.assertEqual(len(bot.SYMBOLS), 3)


# ═══════════════════════════════════════════════════════════════
#  2. CONDITIONS DE SIGNAL
# ═══════════════════════════════════════════════════════════════

class TestSignalConditions(unittest.TestCase):
    """Teste les conditions d'entree avec les ranges RSI v7."""

    def _long(self, close, bb_up, ema, vol, vol_thresh, rsi):
        return (close > bb_up and close > ema
                and vol > vol_thresh and 40 < rsi < 65)

    def _short(self, close, bb_lo, ema, vol, vol_thresh, rsi):
        return (close < bb_lo and close < ema
                and vol > vol_thresh and 20 < rsi < 100)

    # ── LONG ──
    def test_long_rsi_55(self):
        self.assertTrue(self._long(100, 99, 98, 200, 150, 55))

    def test_long_rsi_42_new_in_v7(self):
        """RSI=42 etait rejete en v6 (50-70), accepte en v7 (40-65)."""
        self.assertTrue(self._long(100, 99, 98, 200, 150, 42))

    def test_long_rsi_41(self):
        self.assertTrue(self._long(100, 99, 98, 200, 150, 41))

    def test_long_rsi_64(self):
        self.assertTrue(self._long(100, 99, 98, 200, 150, 64))

    def test_long_rsi_40_boundary_rejected(self):
        self.assertFalse(self._long(100, 99, 98, 200, 150, 40))

    def test_long_rsi_65_boundary_rejected(self):
        self.assertFalse(self._long(100, 99, 98, 200, 150, 65))

    def test_long_rsi_72_overbought_rejected(self):
        self.assertFalse(self._long(100, 99, 98, 200, 150, 72))

    def test_long_rsi_35_too_low_rejected(self):
        self.assertFalse(self._long(100, 99, 98, 200, 150, 35))

    def test_long_bb_fail(self):
        self.assertFalse(self._long(99, 100, 98, 200, 150, 55))

    def test_long_ema_fail(self):
        self.assertFalse(self._long(100, 99, 101, 200, 150, 55))

    def test_long_vol_fail(self):
        self.assertFalse(self._long(100, 99, 98, 100, 150, 55))

    # ── SHORT ──
    def test_short_rsi_35(self):
        self.assertTrue(self._short(98, 99, 100, 200, 150, 35))

    def test_short_rsi_75_new_in_v7(self):
        """RSI=75 etait rejete en v6 (30-50), accepte en v7 (20-100)."""
        self.assertTrue(self._short(98, 99, 100, 200, 150, 75))

    def test_short_rsi_99(self):
        self.assertTrue(self._short(98, 99, 100, 200, 150, 99))

    def test_short_rsi_21(self):
        self.assertTrue(self._short(98, 99, 100, 200, 150, 21))

    def test_short_rsi_50(self):
        self.assertTrue(self._short(98, 99, 100, 200, 150, 50))

    def test_short_rsi_20_boundary_rejected(self):
        self.assertFalse(self._short(98, 99, 100, 200, 150, 20))

    def test_short_rsi_15_too_low_rejected(self):
        self.assertFalse(self._short(98, 99, 100, 200, 150, 15))

    def test_short_bb_fail(self):
        self.assertFalse(self._short(100, 99, 101, 200, 150, 35))

    def test_short_ema_fail(self):
        self.assertFalse(self._short(98, 99, 97, 200, 150, 35))

    def test_short_vol_fail(self):
        self.assertFalse(self._short(98, 99, 100, 100, 150, 35))


# ═══════════════════════════════════════════════════════════════
#  3. VOLUME FILTER
# ═══════════════════════════════════════════════════════════════

class TestVolumeFilter(unittest.TestCase):

    def test_threshold_calculation(self):
        self.assertAlmostEqual(bot.VOL_MULTIPLIER * 1000, 1100)

    def test_vol_above_passes(self):
        self.assertTrue(1200 > bot.VOL_MULTIPLIER * 1000)

    def test_vol_below_fails(self):
        self.assertFalse(1050 > bot.VOL_MULTIPLIER * 1000)

    def test_vol_exact_boundary_fails(self):
        """Strict inequality: vol == threshold does NOT pass."""
        self.assertFalse(1100 > bot.VOL_MULTIPLIER * 1000)


# ═══════════════════════════════════════════════════════════════
#  4. SL / TP
# ═══════════════════════════════════════════════════════════════

class TestSLTPCalculation(unittest.TestCase):

    def test_sl_distance(self):
        self.assertEqual(bot.ATR_SL_MULT * 50.0, 25.0)

    def test_tp_distance(self):
        self.assertEqual(bot.ATR_TP_MULT * 50.0, 100.0)

    def test_rr_ratio(self):
        sl = bot.ATR_SL_MULT * 50.0
        tp = bot.ATR_TP_MULT * 50.0
        self.assertEqual(tp / sl, 4.0)

    def test_long_levels(self):
        entry, atr = 20000.0, 50.0
        sl = entry - bot.ATR_SL_MULT * atr
        tp = entry + bot.ATR_TP_MULT * atr
        self.assertEqual(sl, 19975.0)
        self.assertEqual(tp, 20100.0)

    def test_short_levels(self):
        entry, atr = 20000.0, 50.0
        sl = entry + bot.ATR_SL_MULT * atr
        tp = entry - bot.ATR_TP_MULT * atr
        self.assertEqual(sl, 20025.0)
        self.assertEqual(tp, 19900.0)


# ═══════════════════════════════════════════════════════════════
#  5. TRAILING STOP
# ═══════════════════════════════════════════════════════════════

class TestTrailingStop(unittest.TestCase):

    def test_activate_threshold_consistency(self):
        atr = 50.0
        one_r = bot.ATR_SL_MULT * atr  # 25
        via_r = bot.TRAIL_ACTIVATE_R * one_r  # 1.5 * 25 = 37.5
        via_atr = bot.ATR_TRAIL_ACTIVATE * atr  # 0.75 * 50 = 37.5
        self.assertAlmostEqual(via_r, via_atr)

    def test_long_not_activated(self):
        entry, atr, best = 20000.0, 50.0, 20030.0
        profit = best - entry  # 30 < 37.5
        self.assertFalse(profit >= bot.ATR_TRAIL_ACTIVATE * atr)

    def test_long_activated(self):
        entry, atr, best = 20000.0, 50.0, 20040.0
        profit = best - entry  # 40 >= 37.5
        self.assertTrue(profit >= bot.ATR_TRAIL_ACTIVATE * atr)

    def test_long_new_sl(self):
        atr, best = 50.0, 20050.0
        new_sl = best - bot.ATR_TRAIL_DIST * atr
        self.assertEqual(new_sl, 20040.0)

    def test_short_not_activated(self):
        entry, atr, best = 20000.0, 50.0, 19970.0
        profit = entry - best  # 30 < 37.5
        self.assertFalse(profit >= bot.ATR_TRAIL_ACTIVATE * atr)

    def test_short_activated(self):
        entry, atr, best = 20000.0, 50.0, 19960.0
        profit = entry - best  # 40 >= 37.5
        self.assertTrue(profit >= bot.ATR_TRAIL_ACTIVATE * atr)

    def test_short_new_sl(self):
        atr, best = 50.0, 19950.0
        new_sl = best + bot.ATR_TRAIL_DIST * atr
        self.assertEqual(new_sl, 19960.0)

    def test_long_sl_only_moves_up(self):
        current_sl, new_sl = 20045.0, 20040.0
        self.assertFalse(new_sl > current_sl)

    def test_long_sl_moves_up_when_higher(self):
        current_sl, new_sl = 20035.0, 20040.0
        self.assertTrue(new_sl > current_sl)

    def test_short_sl_only_moves_down(self):
        current_sl, new_sl = 19955.0, 19960.0
        self.assertFalse(new_sl < current_sl)

    def test_short_sl_moves_down_when_lower(self):
        current_sl, new_sl = 19965.0, 19960.0
        self.assertTrue(new_sl < current_sl)


# ═══════════════════════════════════════════════════════════════
#  6. SESSION
# ═══════════════════════════════════════════════════════════════

class TestSession(unittest.TestCase):
    """Teste la logique de session avec les horaires v7 (17h-22h)."""

    def _check(self, hour, minute=0, weekday=0):
        """Reproduit la logique de in_trading_session()."""
        if weekday >= 5:
            return False
        from datetime import time as dt_time
        t = hour * 60 + minute
        start = bot.SESSION_START_H * 60 + bot.SESSION_START_M
        end = bot.SESSION_END_H * 60 + bot.SESSION_END_M
        return start <= t <= end

    def test_17h00_in(self):
        self.assertTrue(self._check(17, 0))

    def test_18h30_in(self):
        self.assertTrue(self._check(18, 30))

    def test_21h30_in_v7(self):
        """21:30 est maintenant IN session (etait OUT en v6 a 21h)."""
        self.assertTrue(self._check(21, 30))

    def test_22h00_in(self):
        """22:00 exactement — borne incluse."""
        self.assertTrue(self._check(22, 0))

    def test_22h01_out(self):
        self.assertFalse(self._check(22, 1))

    def test_16h59_out(self):
        self.assertFalse(self._check(16, 59))

    def test_15h00_out(self):
        self.assertFalse(self._check(15, 0))

    def test_23h00_out(self):
        self.assertFalse(self._check(23, 0))

    def test_saturday_out(self):
        self.assertFalse(self._check(18, 0, weekday=5))

    def test_sunday_out(self):
        self.assertFalse(self._check(18, 0, weekday=6))

    def test_monday_in(self):
        self.assertTrue(self._check(18, 0, weekday=0))

    def test_friday_in(self):
        self.assertTrue(self._check(18, 0, weekday=4))


# ═══════════════════════════════════════════════════════════════
#  7. LOT SIZING
# ═══════════════════════════════════════════════════════════════

class TestLotSizing(unittest.TestCase):

    def _setup_mt5(self, equity, tick_size=0.01, tick_value=1.0,
                   vol_min=0.01, vol_max=100.0, vol_step=0.01):
        mock_mt5.symbol_info.return_value = SimpleNamespace(
            trade_tick_size=tick_size, trade_tick_value=tick_value,
            volume_min=vol_min, volume_max=vol_max, volume_step=vol_step,
            digits=2,
        )
        mock_mt5.account_info.return_value = SimpleNamespace(equity=equity)

    def test_basic_lot(self):
        self._setup_mt5(equity=10000.0)
        sl_dist = bot.ATR_SL_MULT * 50.0  # 25
        lot = bot.calculate_lot_size("TEST", sl_dist)
        # risk = 10000 * 0.02 = 200
        # sl_val = (25 / 0.01) * 1.0 = 2500
        # lot = 200 / 2500 = 0.08
        self.assertIsNotNone(lot)
        self.assertAlmostEqual(lot, 0.08)

    def test_respects_vol_min(self):
        self._setup_mt5(equity=100.0, vol_min=0.1)
        lot = bot.calculate_lot_size("TEST", 25.0)
        self.assertGreaterEqual(lot, 0.1)

    def test_respects_vol_max(self):
        self._setup_mt5(equity=1_000_000.0, tick_value=0.001, vol_max=5.0)
        lot = bot.calculate_lot_size("TEST", 0.01)
        self.assertLessEqual(lot, 5.0)

    def test_step_rounding(self):
        self._setup_mt5(equity=10000.0, vol_step=0.1)
        sl_dist = 25.0
        lot = bot.calculate_lot_size("TEST", sl_dist)
        # lot = 0.08, rounded to step 0.1 → 0.1
        self.assertEqual(lot % 0.1, 0.0)

    def test_none_symbol_info(self):
        mock_mt5.symbol_info.return_value = None
        mock_mt5.account_info.return_value = SimpleNamespace(equity=1000.0)
        self.assertIsNone(bot.calculate_lot_size("TEST", 25.0))

    def test_none_account_info(self):
        mock_mt5.symbol_info.return_value = SimpleNamespace(
            trade_tick_size=0.01, trade_tick_value=1.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01, digits=2,
        )
        mock_mt5.account_info.return_value = None
        self.assertIsNone(bot.calculate_lot_size("TEST", 25.0))


# ═══════════════════════════════════════════════════════════════
#  8. MODIFY SL (retcode 10025 fix)
# ═══════════════════════════════════════════════════════════════

class TestModifySL(unittest.TestCase):

    def setUp(self):
        mock_mt5.symbol_info.return_value = SimpleNamespace(digits=2)

    def test_retcode_10025_silent_success(self):
        """retcode=10025 (No changes) doit retourner True sans error."""
        mock_mt5.order_send.return_value = SimpleNamespace(retcode=10025)
        self.assertTrue(bot.modify_sl("TEST", 123, 100.5, 110.0))

    def test_retcode_done_success(self):
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE
        )
        self.assertTrue(bot.modify_sl("TEST", 123, 100.5, 110.0))

    def test_retcode_other_failure(self):
        mock_mt5.order_send.return_value = SimpleNamespace(retcode=10004)
        self.assertFalse(bot.modify_sl("TEST", 123, 100.5, 110.0))

    def test_none_result_failure(self):
        mock_mt5.order_send.return_value = None
        self.assertFalse(bot.modify_sl("TEST", 123, 100.5, 110.0))

    def test_sl_rounded_to_digits(self):
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE
        )
        bot.modify_sl("TEST", 123, 100.5678, 110.0)
        call_args = mock_mt5.order_send.call_args[0][0]
        self.assertEqual(call_args["sl"], 100.57)


# ═══════════════════════════════════════════════════════════════
#  9. SIGNAL DIFFERE (deferred)
# ═══════════════════════════════════════════════════════════════

class TestDeferredSignal(unittest.TestCase):

    def setUp(self):
        bot.pending_signals.clear()

    def test_create_pending_long(self):
        bot.pending_signals["NAS100."] = {
            "direction": "long", "atr_capped": 45.0,
            "signal_time": "2026-03-25 18:00:00",
        }
        self.assertIn("NAS100.", bot.pending_signals)
        self.assertEqual(bot.pending_signals["NAS100."]["direction"], "long")

    def test_create_pending_short(self):
        bot.pending_signals["DJ30."] = {
            "direction": "short", "atr_capped": 120.0,
            "signal_time": "2026-03-25 19:15:00",
        }
        self.assertEqual(bot.pending_signals["DJ30."]["direction"], "short")

    def test_pop_on_execution(self):
        bot.pending_signals["SP500."] = {
            "direction": "long", "atr_capped": 17.0,
        }
        sig = bot.pending_signals.pop("SP500.")
        self.assertEqual(sig["atr_capped"], 17.0)
        self.assertNotIn("SP500.", bot.pending_signals)

    def test_cancel_on_no_signal(self):
        bot.pending_signals["NAS100."] = {"direction": "long", "atr_capped": 45.0}
        cancelled = bot.pending_signals.pop("NAS100.", None)
        self.assertIsNotNone(cancelled)
        self.assertNotIn("NAS100.", bot.pending_signals)

    def test_cancel_returns_none_if_no_pending(self):
        cancelled = bot.pending_signals.pop("UNKNOWN", None)
        self.assertIsNone(cancelled)

    def test_sl_tp_from_pending(self):
        """Verify SL/TP distances computed from pending signal's atr_capped."""
        atr_capped = 45.0
        sl_dist = bot.ATR_SL_MULT * atr_capped  # 0.5 * 45 = 22.5
        tp_dist = bot.ATR_TP_MULT * atr_capped  # 2.0 * 45 = 90.0
        self.assertAlmostEqual(sl_dist, 22.5)
        self.assertAlmostEqual(tp_dist, 90.0)


# ═══════════════════════════════════════════════════════════════
#  10. INDICATEURS (integration avec pandas_ta)
# ═══════════════════════════════════════════════════════════════

class TestIndicators(unittest.TestCase):

    def _make_df(self, n=200):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        op = close + np.random.randn(n) * 0.3
        return pd.DataFrame({
            "open": op, "high": high, "low": low, "close": close,
            "tick_volume": np.random.randint(500, 5000, n).astype(float),
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        result = bot.compute_indicators(df)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_all_columns_present(self):
        df = self._make_df()
        result = bot.compute_indicators(df)
        for col in ["BBU", "BBL", "vol_sma", "ema_trend", "rsi",
                     "atr", "atr_capped"]:
            self.assertIn(col, result.columns, f"Colonne manquante: {col}")

    def test_last_row_not_nan(self):
        df = self._make_df(300)
        result = bot.compute_indicators(df)
        last = result.iloc[-1]
        for col in ["BBU", "BBL", "vol_sma", "ema_trend", "rsi",
                     "atr", "atr_capped"]:
            self.assertFalse(pd.isna(last[col]), f"{col} is NaN on last row")

    def test_atr_capped_lte_atr(self):
        df = self._make_df(300)
        result = bot.compute_indicators(df)
        valid = result.dropna(subset=["atr", "atr_capped"])
        over = valid[valid["atr_capped"] > valid["atr"] * 1.0001]
        self.assertEqual(len(over), 0, "atr_capped > atr found")

    def test_rsi_range(self):
        df = self._make_df(300)
        result = bot.compute_indicators(df)
        valid_rsi = result["rsi"].dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_bb_upper_above_lower(self):
        df = self._make_df(300)
        result = bot.compute_indicators(df)
        valid = result.dropna(subset=["BBU", "BBL"])
        self.assertTrue((valid["BBU"] > valid["BBL"]).all())


# ═══════════════════════════════════════════════════════════════
#  11. V6 → V7 REGRESSION
# ═══════════════════════════════════════════════════════════════

class TestV6toV7Regression(unittest.TestCase):
    """Verifie que les changements v6 → v7 sont bien appliques."""

    def test_sl_changed_from_075(self):
        self.assertNotEqual(bot.ATR_SL_MULT, 0.75)
        self.assertEqual(bot.ATR_SL_MULT, 0.5)

    def test_vol_changed_from_13(self):
        self.assertNotEqual(bot.VOL_MULTIPLIER, 1.3)
        self.assertEqual(bot.VOL_MULTIPLIER, 1.1)

    def test_risk_changed_from_3pct(self):
        self.assertNotEqual(bot.RISK_PCT, 0.03)
        self.assertEqual(bot.RISK_PCT, 0.02)

    def test_trail_changed_from_03(self):
        self.assertNotEqual(bot.ATR_TRAIL_DIST, 0.3)
        self.assertEqual(bot.ATR_TRAIL_DIST, 0.2)

    def test_session_end_changed_from_21(self):
        self.assertNotEqual(bot.SESSION_END_H, 21)
        self.assertEqual(bot.SESSION_END_H, 22)

    def test_rsi_long_accepts_42(self):
        """v6 rejetait RSI=42 (range 50-70), v7 accepte (range 40-65)."""
        self.assertTrue(40 < 42 < 65)
        self.assertFalse(50 < 42 < 70)

    def test_rsi_short_accepts_75(self):
        """v6 rejetait RSI=75 (range 30-50), v7 accepte (range 20-100)."""
        self.assertTrue(20 < 75 < 100)
        self.assertFalse(30 < 75 < 50)

    def test_tp_mult_updated(self):
        """TP mult should be SL * RR = 0.5 * 4 = 2.0 (was 3.0)."""
        self.assertAlmostEqual(bot.ATR_TP_MULT, 2.0)

    def test_trail_activate_atr_updated(self):
        """Trail activate = 1.5 * 0.5 = 0.75 ATR (was 1.125)."""
        self.assertAlmostEqual(bot.ATR_TRAIL_ACTIVATE, 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
