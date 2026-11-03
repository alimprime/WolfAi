"""Классификация перевода. Приток на биржу не помечается как продажа.

Один держатель может владеть несколькими адресами: совпадение сущности важнее
совпадения адреса. Метка, опубликованная после as_of, на это решение не действует.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WalletTransfer:
    """Один перевод между адресами. Сумма в BTC, не в долларах."""

    event_at: datetime
    published_at: datetime
    label_published_at: datetime | None
    from_address: str
    to_address: str
    from_entity: str | None
    to_entity: str | None
    amount_btc: float
    source: str
    source_version: str


@dataclass(frozen=True)
class TransferView:
    role: str
    counts_toward_exchange_flow: bool
    is_sale: bool
    is_purchase: bool
    confidence: float
    strict_backtest: bool
    reason: str


def classify_transfer(
    transfer: WalletTransfer,
    as_of: datetime,
    exchange_entities: frozenset[str],
) -> TransferView:
    """Вид перевода глазами наблюдателя в момент as_of."""
    if as_of.tzinfo is None or transfer.event_at.tzinfo is None or transfer.published_at.tzinfo is None:
        raise ValueError("время перевода или as_of без часового пояса")
    if transfer.amount_btc < 0:
        raise ValueError("amount_btc отрицателен")
    if transfer.published_at > as_of or transfer.event_at > as_of:
        return TransferView(
            "hidden",
            False,
            False,
            False,
            0.0,
            False,
            "перевод ещё не опубликован к моменту решения",
        )
    from_entity = _entity_if_known(transfer.from_entity, transfer.label_published_at, as_of)
    to_entity = _entity_if_known(transfer.to_entity, transfer.label_published_at, as_of)
    late_label = transfer.label_published_at is not None and transfer.label_published_at > transfer.event_at
    confidence = 0.4 if late_label else 0.7
    strict = transfer.label_published_at is not None and transfer.label_published_at <= as_of
    if from_entity and to_entity and from_entity == to_entity:
        return TransferView(
            "internal",
            False,
            False,
            False,
            confidence,
            False,
            "адреса одной сущности: внутренний перевод, не покупка и не продажа",
        )
    if from_entity is None and to_entity is None:
        return TransferView(
            "unattributed",
            False,
            False,
            False,
            0.2,
            False,
            "метки владельцев к as_of неизвестны",
        )
    to_exchange = to_entity in exchange_entities if to_entity else False
    from_exchange = from_entity in exchange_entities if from_entity else False
    if to_exchange and not from_exchange:
        return TransferView(
            "exchange_inflow",
            True,
            False,
            False,
            confidence,
            strict and not late_label,
            "зачисление на адрес биржи; это не доказанная продажа",
        )
    if from_exchange and not to_exchange:
        return TransferView(
            "exchange_outflow",
            True,
            False,
            False,
            confidence,
            strict and not late_label,
            "вывод с адреса биржи; это не доказанная покупка",
        )
    return TransferView(
        "external",
        False,
        False,
        False,
        confidence,
        False,
        "перевод вне выбранного множества бирж",
    )


def net_exchange_btc(
    transfers: list[WalletTransfer],
    as_of: datetime,
    exchange_entities: frozenset[str],
) -> float | None:
    """Сумма притоков минус оттоки. None, если ни один перевод не атрибутирован.

    Внутренние переводы в сумму не входят. Пустой видимый набор — не ноль.
    """
    total = 0.0
    seen = False
    for transfer in transfers:
        view = classify_transfer(transfer, as_of, exchange_entities)
        if not view.counts_toward_exchange_flow:
            continue
        seen = True
        sign = 1.0 if view.role == "exchange_inflow" else -1.0
        total += sign * transfer.amount_btc
    if not seen:
        return None
    return total


def _entity_if_known(entity: str | None, label_published_at: datetime | None, as_of: datetime) -> str | None:
    if not entity or not entity.strip():
        return None
    if label_published_at is None or label_published_at > as_of:
        return None
    return entity
