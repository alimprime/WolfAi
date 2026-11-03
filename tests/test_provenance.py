from datetime import datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.data.binance import daily_bar_available_at
from wolfai.data.provenance import DataProvenance, require_visible, visible_at

UTC = timezone.utc
PLUS_5 = timezone(timedelta(hours=5))


def _fact(
    published: datetime,
    *,
    event: datetime | None = None,
    ingested: datetime | None = None,
    confidence: float = 0.8,
    source: str = "binance-vision",
    source_version: str = "sha256:abc",
) -> DataProvenance:
    return DataProvenance(
        event_time=event or published,
        published_at=published,
        ingested_at=ingested or published,
        source=source,
        source_version=source_version,
        confidence=confidence,
    )


def test_visibility_compares_instants_across_offsets() -> None:
    as_of = datetime(2024, 12, 1, 12, tzinfo=UTC)
    same_instant = _fact(datetime(2024, 12, 1, 17, tzinfo=PLUS_5))
    one_minute_later = _fact(datetime(2024, 12, 1, 17, 1, tzinfo=PLUS_5))
    assert visible_at([one_minute_later, same_instant], as_of) == [same_instant]
    assert require_visible([same_instant], as_of) == [same_instant]
    with pytest.raises(ValueError, match="опубликован"):
        require_visible([same_instant, one_minute_later], as_of)


def test_later_revision_stays_hidden_and_order_is_kept() -> None:
    as_of = datetime(2024, 12, 15, tzinfo=UTC)
    original = _fact(datetime(2024, 12, 2, tzinfo=UTC), source_version="v1")
    earlier = _fact(datetime(2024, 12, 1, tzinfo=UTC), source_version="v0")
    revision = _fact(datetime(2024, 12, 20, tzinfo=UTC), source_version="v2")
    assert visible_at([original, earlier, revision], as_of) == [original, earlier]


def test_late_download_does_not_change_publication_time() -> None:
    published = datetime(2024, 12, 1, 15, tzinfo=UTC)
    fact = _fact(published, ingested=datetime(2026, 10, 26, tzinfo=UTC))
    assert visible_at([fact], datetime(2024, 12, 2, tzinfo=UTC)) == [fact]


def test_news_during_the_session_waits_for_the_daily_close() -> None:
    open_time = datetime(2024, 12, 1, tzinfo=UTC)
    candle = Candle(open_time, 100, 110, 90, 105, 1)
    news = _fact(
        datetime(2024, 12, 1, 15, tzinfo=UTC),
        ingested=datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert visible_at([news], open_time) == []
    assert visible_at([news], daily_bar_available_at(candle)) == [news]


def test_naive_clock_and_blank_source_are_rejected() -> None:
    with pytest.raises(ValueError, match="часового пояса"):
        _fact(datetime(2024, 12, 1))
    with pytest.raises(ValueError, match="source_version"):
        _fact(datetime(2024, 12, 1, tzinfo=UTC), source_version=" ")
    with pytest.raises(ValueError, match="source пуст"):
        _fact(datetime(2024, 12, 1, tzinfo=UTC), source="")
    fact = _fact(datetime(2024, 12, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="as_of"):
        visible_at([fact], datetime(2024, 12, 2))


def test_ingest_before_publication_is_rejected() -> None:
    with pytest.raises(ValueError, match="ingested_at"):
        _fact(
            datetime(2024, 12, 2, tzinfo=UTC),
            ingested=datetime(2024, 12, 1, tzinfo=UTC),
        )


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_confidence_must_be_a_finite_unit_interval(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _fact(datetime(2024, 12, 1, tzinfo=UTC), confidence=confidence)
