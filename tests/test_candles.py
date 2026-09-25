from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wolfai.data import Candle, load_candles, validate_candles

UTC = timezone.utc


def _candle(day: int, close: float = 100.0, volume: float = 1.0) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
    )


def test_load_valid_csv(tmp_path: Path) -> None:
    path = tmp_path / "btc.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T00:00:00+00:00,100,110,90,105,12\n"
        "2024-01-02T00:00:00+00:00,105,112,101,108,8\n",
        encoding="utf-8",
    )
    candles = load_candles(path)
    assert len(candles) == 2
    assert candles[1].close == 108


def test_rejects_timestamp_without_timezone() -> None:
    candle = _candle(0)
    naive = Candle(
        timestamp=candle.timestamp.replace(tzinfo=None),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1,
    )
    with pytest.raises(ValueError, match="часового пояса"):
        validate_candles([naive])


def test_rejects_non_positive_price_and_negative_volume() -> None:
    with pytest.raises(ValueError, match="положительной"):
        validate_candles([_candle(0, close=0)])
    broken = _candle(0)
    with pytest.raises(ValueError, match="volume"):
        validate_candles([Candle(**{**broken.__dict__, "volume": -1})])


def test_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError, match="не возрастает"):
        validate_candles([_candle(1), _candle(0)])
