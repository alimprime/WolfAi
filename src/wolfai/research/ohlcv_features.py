"""Дополнительные признаки из тех же свечей, что уже есть на момент решения.

Четыре ценовых признака считаются прежней функцией `feature_vector` и здесь
не переопределяются. `volume_change` в этом модуле — разность log1p,
а не отношение объёмов из `feature_vector`.
"""

from __future__ import annotations

import math

from wolfai.data.candles import Candle
from wolfai.research.features import feature_vector

PRICE_FEATURES = ("log_return", "momentum_7", "momentum_30", "volatility_7")
OHLCV_FEATURES = ("intraday_range", "candle_return", "volume_change", "relative_volume_7")
ALL_FEATURES = PRICE_FEATURES + OHLCV_FEATURES
VOLUME_WINDOW = 7


def causal_features(candles: list[Candle], index: int) -> tuple[dict[str, float | None], str | None]:
    """Признаки бара index. Свечи после него не читаются.

    Вторая величина — причина, по которой добавленные OHLCV не посчитаны.
    """
    history = candles[: index + 1]
    price = feature_vector(candles, index, _admitted=True)
    features: dict[str, float | None] = {name: price[name] for name in PRICE_FEATURES}
    extra, reason = _extra(history)
    if reason is not None:
        for name in OHLCV_FEATURES:
            features[name] = None
        return features, reason
    features.update(extra)
    return features, None


def _extra(history: list[Candle]) -> tuple[dict[str, float], str | None]:
    if len(history) < VOLUME_WINDOW:
        return {}, "мало баров для объёма"
    candle = history[-1]
    previous = history[-2]
    if candle.high <= 0 or candle.low <= 0 or candle.high < candle.low:
        return {}, "некорректный диапазон high/low"
    if candle.open <= 0 or candle.close <= 0:
        return {}, "некорректная цена open/close"
    if candle.volume < 0 or previous.volume < 0:
        return {}, "отрицательный объём"
    window = [bar.volume for bar in history[-VOLUME_WINDOW:]]
    if any(volume < 0 for volume in window):
        return {}, "отрицательный объём"
    median = _median(window)
    if median == 0:
        return {}, "нулевая медиана объёма"
    return {
        "intraday_range": math.log(candle.high / candle.low),
        "candle_return": math.log(candle.close / candle.open),
        "volume_change": math.log1p(candle.volume) - math.log1p(previous.volume),
        "relative_volume_7": candle.volume / median,
    }, None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
