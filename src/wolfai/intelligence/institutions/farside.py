"""Farside Investors публикует таблицу потоков на сайте.

Официального API, CSV и лицензии на выгрузку в документации нет.
Страница оставляет права за собой: https://farside.co.uk/bitcoin-etf-flow-all-data/
HTML не скачивается и не разбирается. Сторонние обёртки поверх страницы не используются.
"""

from __future__ import annotations

from wolfai.intelligence.schema import Observation, SourceUnavailable

FARSIDE_PAGE = "https://farside.co.uk/bitcoin-etf-flow-all-data/"


def fetch_farside() -> list[Observation]:
    raise SourceUnavailable(
        "У Farside Investors нет документированного API. "
        f"Страница {FARSIDE_PAGE} не скачивается. "
        "Исторический поток ETF можно подключить только локальным CSV с published_at."
    )
