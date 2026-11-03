"""Выбор версии ряда, уже опубликованной к моменту решения."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from wolfai.intelligence.schema import Observation


class SeriesIndex:
    """Индекс наблюдений. Ревизия с более поздним published_at скрыта до своего часа."""

    def __init__(self, observations: list[Observation]) -> None:
        self.by_series: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            self.by_series[observation.series_id].append(observation)

    def latest_strict(self, series_id: str, as_of: datetime) -> Observation | None:
        """Последнее событие серии, чья точная публикация не позже as_of.

        При равном event_at побеждает более поздняя уже видимая публикация.
        Если совпали и событие, и публикация, остаётся более ранняя загрузка:
        файл, скачанный потом, не переписывает то, что было известно раньше.
        """
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of без часового пояса")
        chosen: Observation | None = None
        for observation in self.by_series.get(series_id, ()):
            if not _visible_strict(observation, as_of):
                continue
            if chosen is None or _prefer(observation, chosen):
                chosen = observation
        return chosen

    def count(self, series_id: str) -> int:
        return len(self.by_series.get(series_id, ()))


def visible_exact_count(observations: list[Observation], as_of: datetime, category: str) -> int:
    """Сколько записей категории уже имели точное published_at к as_of.

    Это не допуск в backtest: строгий допуск проверяет ещё и strict_backtest.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of без часового пояса")
    return sum(
        item.category == category
        and item.publication_time_exact
        and item.published_at is not None
        and item.published_at <= as_of
        for item in observations
    )


def _visible_strict(observation: Observation, as_of: datetime) -> bool:
    return bool(
        observation.strict_backtest
        and observation.publication_time_exact
        and observation.published_at is not None
        and observation.published_at <= as_of
        and observation.event_at <= as_of
        and observation.value is not None
    )


def _prefer(candidate: Observation, incumbent: Observation) -> bool:
    if candidate.event_at != incumbent.event_at:
        return candidate.event_at > incumbent.event_at
    assert candidate.published_at is not None and incumbent.published_at is not None
    if candidate.published_at != incumbent.published_at:
        return candidate.published_at > incumbent.published_at
    if candidate.ingested_at != incumbent.ingested_at:
        return candidate.ingested_at < incumbent.ingested_at
    return candidate.source_version < incumbent.source_version
