"""SEC EDGAR. Ежедневных чистых потоков ETF в этом API нет.

Официальный индекс заявок: https://data.sec.gov/submissions/CIK##########.json
SEC требует User-Agent с контактом и ограничивает частоту.
https://www.sec.gov/os/accessing-edgar-data
Без такого заголовка запрос не отправляется.
"""

from __future__ import annotations

import json
from datetime import datetime
from urllib.request import Request, urlopen

from wolfai.intelligence.schema import USE_DESCRIPTIVE, Observation, SourceUnavailable, parse_utc

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


def fetch_submissions(cik: str, user_agent: str | None, *, ingested_at: datetime) -> list[Observation]:
    if not user_agent or "@" not in user_agent:
        raise SourceUnavailable("SEC EDGAR требует User-Agent с контактом. Запрос не отправлен.")
    if not cik.isdigit():
        raise SourceUnavailable("CIK должен состоять из цифр")
    url = EDGAR_SUBMISSIONS.format(cik=cik.zfill(10))
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise SourceUnavailable("EDGAR вернул не объект")
    return parse_submissions(payload, ingested_at=ingested_at, source=url)


def parse_submissions(
    payload: dict[str, object],
    *,
    ingested_at: datetime,
    source: str,
) -> list[Observation]:
    """Читает parallel arrays filings.recent. Это даты заявок, не дневные потоки."""
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        raise SourceUnavailable("в ответе EDGAR нет filings")
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        raise SourceUnavailable("в ответе EDGAR нет filings.recent")
    accepted = recent.get("acceptanceDateTime")
    forms = recent.get("form")
    if not isinstance(accepted, list) or not isinstance(forms, list) or len(accepted) != len(forms):
        raise SourceUnavailable("колонки EDGAR разной длины или отсутствуют")
    observations: list[Observation] = []
    for index, (stamp, form) in enumerate(zip(accepted, forms, strict=True)):
        if not isinstance(stamp, str) or not isinstance(form, str):
            raise SourceUnavailable(f"заявка {index} без времени или формы")
        published_at = parse_utc(stamp)
        observations.append(Observation(
            series_id="sec.edgar.filing",
            event_at=published_at,
            published_at=published_at,
            ingested_at=ingested_at,
            source=source,
            source_version=f"edgar:{stamp}:{form}",
            confidence=0.9,
            value=None,
            unit="filing",
            category="etf",
            use_class=USE_DESCRIPTIVE,
            publication_time_exact=True,
            strict_backtest=False,
            notes=(
                f"Форма {form} принята EDGAR в указанное время. "
                "Это не дневной чистый поток ETF и не on-chain покупка."
            ),
            entity=form,
        ))
    return observations
