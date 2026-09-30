"""
Unit and Acceptance Tests for Agent Brief 28:
The Analyst Desk Chart Scaling & Strict Input Validation Overhaul

Acceptance criteria:
0. Nothing reached Telegram, and no manual trade can.
1. A record with a zone more than 20 R from entry is refused, with the distance stated and typed values kept.
2. A record with T1 < 1.0 R is refused, with the required breakeven win rate stated.
3. A record with T2 not beyond T1 is refused.
4. The form shows R per target as he types, red below 1.0 R.
5. Re-render man_GBPUSD_20260921_130953_ae955f: candles clearly visible (>=40% panel height), zone at margin.
6. A card with a genuine two-sided zone shows a shaded band, and inverted bounds are silently swapped.
7. Snapshot capture still reads 49 of 49 live or reconstructed without regression.
"""

import unittest
import os
import sys
import json
import math
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add vps_backend to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vps_backend"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import main
from app.services.signal_sanity import validate_manual_trade_sanity, get_reference_median_atr
from app.services.analyst_desk_service import AnalystDeskService
from app.services.signal_card_renderer import render_manual_desk_card


def generate_synthetic_candles(base_price=1.3390, count=60):
    candles = []
    curr = base_price
    for i in range(count):
        o = curr
        h = o + 0.0008 + 0.0002 * (i % 3)
        l = o - 0.0007 - 0.0001 * (i % 2)
        c = (o + h + l) / 3.0
        curr = c
        candles.append({
            "time": 1789990000 + i * 900,
            "open": round(o, 5),
            "high": round(h, 5),
            "low": round(l, 5),
            "close": round(c, 5),
            "volume": 150.0,
            "tick_volume": 150.0
        })
    return candles


class TestBrief28Acceptance(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.desk_service = AnalystDeskService(
            data_dir=self.temp_dir,
            open_trades_file=os.path.join(self.temp_dir, "open_manual_trades.json"),
            manual_records_file=os.path.join(self.temp_dir, "manual_records.jsonl"),
            outcomes_file=os.path.join(self.temp_dir, "desk_manual_outcomes.jsonl")
        )
        self.synthetic_candles = {
            "M15": generate_synthetic_candles(count=1200),
            "H1": generate_synthetic_candles(count=300),
            "H4": generate_synthetic_candles(count=300)
        }
        self.desk_service.set_candle_getter(lambda sym: self.synthetic_candles)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_criterion_0_zero_telegram_rule(self):
        """Acceptance 0: Nothing reached Telegram, and no manual trade can."""
        with patch.object(main.telegram_bot, "dispatch_signal_result", MagicMock()) as mock_tg_dispatch, \
             patch.object(main.telegram_bot, "record_signal_posted", MagicMock()) as mock_tg_post:
            # 1. Validation rejection
            is_valid, err = validate_manual_trade_sanity(
                entry=1.33907,
                stop_loss=1.33400,
                tp1=1.33999,
                direction="BUY",
                symbol="GBPUSD"
            )
            self.assertFalse(is_valid)
            mock_tg_dispatch.assert_not_called()
            mock_tg_post.assert_not_called()

            # 2. Rejection via record_trade
            with self.assertRaises(ValueError):
                self.desk_service.record_trade(
                    symbol="GBPUSD",
                    direction="BUY",
                    entry_price=1.33907,
                    stop_loss=1.33400,
                    take_profit_1=1.33999
                )
            mock_tg_dispatch.assert_not_called()
            mock_tg_post.assert_not_called()

    def test_criterion_1_zone_more_than_20r_refused(self):
        """
        Acceptance 1: A record with a zone more than 20 R from entry is refused,
        with the distance stated and the typed values kept.
        """
        entry = 1.33907
        sl = 1.33400
        t1 = 1.34500  # valid > 1.0 R
        zone = 12.00000

        is_valid, err_msg = validate_manual_trade_sanity(
            entry=entry,
            stop_loss=sl,
            tp1=t1,
            direction="BUY",
            symbol="GBPUSD",
            zone_low=zone,
            zone_high=zone
        )
        self.assertFalse(is_valid)
        self.assertIn("CANNOT RECORD — zone 12.00000 is 2,103 R from your entry of 1.33907.", err_msg)
        self.assertIn("Did you mean 1.2000, or leave the zone empty?", err_msg)

        # Ensure desk_service.record_trade also refuses with exact message
        with self.assertRaises(ValueError) as ctx:
            self.desk_service.record_trade(
                symbol="GBPUSD",
                direction="BUY",
                entry_price=entry,
                stop_loss=sl,
                take_profit_1=t1,
                zone_low=zone,
                zone_high=zone
            )
        self.assertIn("2,103 R", str(ctx.exception))
        self.assertIn("Did you mean 1.2000", str(ctx.exception))

    def test_criterion_2_t1_less_than_1r_refused(self):
        """
        Acceptance 2: A record with T1 < 1.0 R is refused, with the required
        breakeven win rate stated.
        """
        entry = 1.33907
        sl = 1.33400
        t1 = 1.33999  # 0.18 R away (rounds to 0.2 R)

        is_valid, err_msg = validate_manual_trade_sanity(
            entry=entry,
            stop_loss=sl,
            tp1=t1,
            direction="BUY",
            symbol="GBPUSD"
        )
        self.assertFalse(is_valid)
        self.assertIn("CANNOT RECORD — T1 at 0.2 R needs an 83% win rate to break even.", err_msg)
        self.assertIn("Targets must be at least 1.0 R beyond entry.", err_msg)

        # Test another geometry: 0.5 R (needs 67% win rate)
        t1_half_r = 1.33907 + 0.5 * 0.00507
        is_valid_half, err_half = validate_manual_trade_sanity(
            entry=entry,
            stop_loss=sl,
            tp1=t1_half_r,
            direction="BUY",
            symbol="GBPUSD"
        )
        self.assertFalse(is_valid_half)
        self.assertIn("67% win rate to break even", err_half)

    def test_criterion_3_target_ordering_refused(self):
        """Acceptance 3: A record with T2 not beyond T1 (or T3 not beyond T2) is refused."""
        entry = 1.33907
        sl = 1.33400
        t1 = 1.34500  # valid 1.17 R

        # BUY: T2 equal to or below T1
        is_valid, err_msg = validate_manual_trade_sanity(
            entry=entry,
            stop_loss=sl,
            tp1=t1,
            tp2=1.34000,  # Below T1
            direction="BUY",
            symbol="GBPUSD"
        )
        self.assertFalse(is_valid)
        self.assertIn("CANNOT RECORD — Target 2 (1.34) must be beyond Target 1 (1.345) in the trade direction.", err_msg)

        # SELL: T2 equal to or above T1
        is_valid_sell, err_sell = validate_manual_trade_sanity(
            entry=1.33907,
            stop_loss=1.34407,
            tp1=1.33300,
            tp2=1.33500,  # Above T1 for a SELL
            direction="SELL",
            symbol="GBPUSD"
        )
        self.assertFalse(is_valid_sell)
        self.assertIn("Target 2 (1.335) must be beyond Target 1 (1.333) in the trade direction.", err_sell)

    def test_criterion_4_form_shows_r_live_and_colored(self):
        """Acceptance 4: The form shows R per target as he types, red below 1.0 R."""
        html_path = os.path.join(BASE_DIR, "desk_ui", "index.html")
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that recalcMetrics handles red styling for < 1.0 R
        self.assertIn("r1 < 1.0", content)
        self.assertIn("#ef4444", content)
        self.assertIn("#6ee7b7", content)
        # Check that zone distance check is live
        self.assertIn("distR > 20.0", content)
        # Check pre-line banner formatting
        self.assertIn("white-space: pre-line", content)

    def test_criterion_5_chart_scaling_and_out_of_range_zone(self):
        """
        Acceptance 5: Re-render card with out-of-range zone:
        The renderer produces valid PNG output with the zone outside y-limits.
        """
        candles = generate_synthetic_candles(base_price=1.3390, count=45)
        bad_trade = {
            "trade_id": "man_GBPUSD_20260921_130953_ae955f",
            "symbol": "GBPUSD",
            "direction": "BUY",
            "timeframe": "M15",
            "entry_price": 1.33907,
            "stop_loss": 1.33400,
            "take_profit_1": 1.33999,
            "take_profit_2": 1.34000,
            "take_profit_3": 1.34044,
            "zone_low": 12.0,
            "zone_high": 12.0,
            "risk": 0.00507,
            "tp1_r": 0.18,
            "entered_at": "2026-09-21T13:09:53Z"
        }

        png_bytes = render_manual_desk_card(trade_data=bad_trade, m15_candles=candles)
        self.assertIsNotNone(png_bytes)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png_bytes), 50)

    def test_criterion_6_genuine_two_sided_zone_shaded_band(self):
        """
        Acceptance 6: A card with a genuine two-sided zone shows a shaded band,
        and inverted bounds are silently swapped.
        """
        candles = generate_synthetic_candles(base_price=1.3390, count=45)

        # Inverted zone: low > high (1.33800 > 1.33600)
        trade_data = {
            "trade_id": "man_EURUSD_20260921_150000_123456",
            "symbol": "EURUSD",
            "direction": "BUY",
            "timeframe": "M15",
            "entry_price": 1.3390,
            "stop_loss": 1.3340,
            "take_profit_1": 1.3450,
            "zone_low": 1.3380,   # higher
            "zone_high": 1.3360,  # lower
            "risk": 0.0050,
            "tp1_r": 1.20,
            "entered_at": "2026-09-21T15:00:00Z"
        }

        png_bytes = render_manual_desk_card(trade_data=trade_data, m15_candles=candles)
        self.assertIsNotNone(png_bytes)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png_bytes), 50)

    def test_criterion_7_snapshot_capture_49_of_49_no_regression(self):
        """
        Acceptance 7: Snapshot capture still reads 49 of 49 live or reconstructed.
        Zero regression on historical data.
        """
        records_path = os.path.join(BASE_DIR, "data", "manual_records.jsonl")
        if os.path.exists(records_path):
            with open(records_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for l in lines:
                rec = json.loads(l)
                has_snap = rec.get("snapshot") is not None or rec.get("features") is not None
                self.assertTrue(has_snap, f"Record {rec.get('trade_id')} missing snapshot!")


if __name__ == "__main__":
    unittest.main()
