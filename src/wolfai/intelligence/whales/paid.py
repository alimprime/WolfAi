"""Платные адаптеры крупных держателей. Данных они не имитируют.

Вызов с ключом тоже останавливается: в этом этапе схема ответа не проверялась
на живой выгрузке, а кредиты тратить нельзя.
"""

from __future__ import annotations

from dataclasses import dataclass

from wolfai.intelligence.schema import Observation, SourceUnavailable


@dataclass(frozen=True)
class SourceProfile:
    name: str
    docs_url: str
    endpoint: str | None
    connected: bool
    historical_coverage: str
    license_note: str
    reason: str


ARKHAM = SourceProfile(
    name="arkham",
    docs_url="https://arkm.com/api/docs",
    endpoint="https://api.arkm.com/transfers",
    connected=False,
    historical_coverage="не загружалась: API кредитный, /transfers тарифицируется по строкам",
    license_note="коммерческий API Arkham, ключ и кредиты обязательны",
    reason="Arkham не вызывался: нет ключа и нет права тратить кредиты на исторические переводы",
)

GLASSNODE = SourceProfile(
    name="glassnode",
    docs_url="https://docs.glassnode.com/basic-api/api",
    endpoint=None,
    connected=False,
    historical_coverage="бесплатного исторического API нет; Light API платного Advanced — 14 дней",
    license_note="подписка Glassnode, персональная или коммерческая",
    reason="Glassnode не вызывался: исторический on-chain ряд требует платный API",
)

CRYPTOQUANT = SourceProfile(
    name="cryptoquant",
    docs_url="https://docs.cryptoquant.com/v2/flows/reserve",
    endpoint="https://api.cryptoquant.com/v1/btc/exchange-flows/reserve",
    connected=False,
    historical_coverage="бесплатный Basic: 30 дней рыночных данных, on-chain не входит",
    license_note="on-chain начинается с плана Professional",
    reason="CryptoQuant не вызывался: бесплатный тариф не содержит исторические биржевые резервы",
)

PROFILES = {
    "arkham": ARKHAM,
    "glassnode": GLASSNODE,
    "cryptoquant": CRYPTOQUANT,
}


def fetch_paid(name: str, api_key: str | None = None) -> list[Observation]:
    """Всегда останавливается. Ключ не превращает отказ в синтетический ряд."""
    profile = PROFILES.get(name)
    if profile is None:
        raise SourceUnavailable(f"неизвестный платный источник {name}")
    del api_key
    raise SourceUnavailable(profile.reason)
