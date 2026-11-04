"""Семантика дневной свечи Binance. Метка времени — открытие бара, не решение.

`timestamp` дневной свечи BTCUSDT — 00:00 UTC дня, в который бар открылся.
OHLC и объём этого бара известны только на следующей полуночи UTC:
это `daily_bar_available_at`. Прогноз ставится в `decision_time`,
когда бар решения уже закрыт. Метка направления известна в `target_time`,
когда закрылся бар цели.

Равенство close[t+h] и close[t] — это не рост: y = 0.
Сдвиг индекса здесь явный, через эти две метки, а не через молчаливый lag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import Candle

TRAIN_LABEL_DEADLINE = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
DEV_DECISION_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
SERIES_END = datetime(2022, 1, 1, tzinfo=timezone.utc)
LABEL_RULE = "y = 1 только если close[t+h] > close[t]; равенство даёт 0"
TIMESTAMP_RULE = (
    "timestamp — открытие суток UTC. decision_time и target_time — "
    "следующие полуночи, когда соответствующий close уже известен."
)


def bar_open_time(candle: Candle) -> datetime:
    """Момент открытия бара. Цены этой свечи в этот момент ещё не известны."""
    _require_utc(candle.timestamp)
    return candle.timestamp


def decision_time(candle: Candle) -> datetime:
    """Момент прогноза: close бара решения уже опубликован."""
    return daily_bar_available_at(candle)


def target_known_time(candle: Candle) -> datetime:
    """Момент, когда close бара цели можно использовать как метку."""
    return daily_bar_available_at(candle)


def direction(close_now: float, close_later: float) -> int:
    """1 — рост, 0 — падение или равенство."""
    if close_later > close_now:
        return 1
    return 0


def train_label_closed(target_time: datetime) -> bool:
    """Метка созрела не позже 2020-12-31 23:59:59 UTC. 2021 в обучение не входит."""
    _require_utc(target_time)
    return target_time <= TRAIN_LABEL_DEADLINE


def dev_forecast(decision: datetime, target: datetime) -> bool:
    """Прогноз сделан в 2021, а метка известна строго до 2022-01-01."""
    _require_utc(decision)
    _require_utc(target)
    return DEV_DECISION_START <= decision < SERIES_END and target < SERIES_END


def _require_utc(moment: datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise ValueError("метка времени должна быть в UTC")
