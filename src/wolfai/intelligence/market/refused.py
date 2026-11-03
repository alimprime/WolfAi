"""Источники, которые в этом этапе не дают пригодной истории."""

from __future__ import annotations

from wolfai.intelligence.schema import Observation, SourceUnavailable
from wolfai.intelligence.whales.paid import SourceProfile

COINGLASS = SourceProfile(
    name="coinglass",
    docs_url="https://www.coinglass.com/",
    endpoint=None,
    connected=False,
    historical_coverage="не загружалась",
    license_note="API CoinGlass требует ключ; путь исторической ликвидации в этом этапе не подтверждался",
    reason="CoinGlass не вызывался: нет подтверждённого бесплатного исторического endpoint и нет ключа",
)

GDELT = SourceProfile(
    name="gdelt",
    docs_url="https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
    endpoint="https://api.gdeltproject.org/api/v2/doc/doc",
    connected=False,
    historical_coverage="полный корпус не загружался",
    license_note="DOC API публичный и просит не чаще одного запроса в 5 секунд",
    reason=(
        "GDELT не поставлен в модель: неполный корпус нельзя выдавать за известные на дату новости. "
        "Запись без seendate была бы отброшена, а не датирована догадкой."
    ),
)


def fetch_coinglass(api_key: str | None = None) -> list[Observation]:
    del api_key
    raise SourceUnavailable(COINGLASS.reason)


def parse_gdelt_doc(payload: dict[str, object], *, http_status: int) -> list[dict[str, str]]:
    """Проверяет ответ DOC API. Не создаёт время публикации, если seendate нет.

    Даже годный ответ не становится признаком прогноза: для этого нужен полный
    корпус, а не одна страница. Функция возвращает только поля, которые уже есть.
    """
    if http_status != 200:
        raise SourceUnavailable(f"GDELT ответил {http_status}; статьи не сохранены")
    articles = payload.get("articles")
    if not isinstance(articles, list):
        raise SourceUnavailable("в ответе GDELT нет списка articles")
    parsed: list[dict[str, str]] = []
    for article in articles:
        if not isinstance(article, dict) or not isinstance(article.get("seendate"), str):
            raise SourceUnavailable("у статьи GDELT нет seendate; время не подставляется")
        parsed.append({
            "seendate": article["seendate"],
            "url": str(article.get("url") or ""),
            "title": str(article.get("title") or ""),
        })
    return parsed
