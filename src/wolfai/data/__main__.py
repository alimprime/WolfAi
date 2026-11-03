"""Offline command line interface for preparing local BTC history."""

from __future__ import annotations

import argparse
from pathlib import Path

from wolfai.data.binance import (
    assert_filename_period,
    load_binance_spot_daily_zip,
    save_normalized_candles,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="WolfAi offline BTC data importer")
    parser.add_argument("archive", help="Path to BTCUSDT-1d-YYYY-MM[-DD].zip")
    parser.add_argument("--output", required=True, help="Output normalized CSV under data/")
    args = parser.parse_args()
    candles = load_binance_spot_daily_zip(args.archive)
    assert_filename_period(candles, Path(args.archive).name)
    save_normalized_candles(candles, args.output)
    print(f"Imported {len(candles)} BTCUSDT daily candles to {args.output}")


if __name__ == "__main__":
    main()
