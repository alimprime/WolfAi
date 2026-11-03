"""Контрольные метрики на training и validation. Holdout не оценивается."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from wolfai.data.binance import validate_daily_candles
from wolfai.data.candles import Candle, load_candles
from wolfai.forecasting.baseline import (
    ForecastModel,
    HistoricalQuantileBaseline,
    WalkForwardResult,
    walk_forward,
)
from wolfai.forecasting.random_walk import RandomWalkBaseline
from wolfai.forecasting.splits import (
    EVALUATED_PERIODS,
    Period,
    period_bounds,
    refuse_holdout,
)

REQUIRED_HISTORY = (date(2019, 1, 1), date(2024, 12, 31))


def evaluate_split(
    candles: list[Candle],
    horizon_days: int,
    period: Period,
    model: ForecastModel | None = None,
    *,
    min_history: int = 30,
) -> WalkForwardResult:
    """Walk-forward внутри периода.

    В счёт идут только те решения, у которых и последний известный close,
    и целевой close лежат внутри периода. Исход из следующего периода,
    включая holdout, метрику не двигает.
    """
    refuse_holdout(period)
    start, end = period_bounds(period)
    return walk_forward(
        candles,
        horizon_days,
        min_history=min_history,
        model=model,
        decision_from=start,
        decision_to=end,
        target_from=start,
        target_to=end,
    )


def audit_period_leak(
    candles: list[Candle],
    period: Period,
    mutate_from: date,
    horizons: tuple[int, ...] = (1, 7),
) -> list[dict[str, object]]:
    """Метрики периода не меняются, если переписать close начиная с mutate_from."""
    refuse_holdout(period)
    mutated = _rewrite_from(candles, mutate_from)
    checks: list[dict[str, object]] = []
    for horizon in horizons:
        for factory in (HistoricalQuantileBaseline, RandomWalkBaseline):
            original = evaluate_split(candles, horizon, period, factory())
            altered = evaluate_split(mutated, horizon, period, factory())
            passed = original == altered
            checks.append({
                "period": period.value,
                "horizon_days": horizon,
                "model": factory.__name__,
                "mutate_from": mutate_from.isoformat(),
                "passed": passed,
            })
            if not passed:
                raise RuntimeError(
                    f"утечка в {period.value}, горизонт {horizon}, {factory.__name__}"
                )
    return checks


def score_controls(candles: list[Candle]) -> dict[str, object]:
    """MAE, MAPE, направление и интервалы. Holdout в расчёт не входит."""
    validate_daily_candles(candles)
    first = candles[0].timestamp.date()
    last = candles[-1].timestamp.date()
    required_start, required_end = REQUIRED_HISTORY
    if first > required_start or last < required_end:
        raise ValueError(
            "для контрольных метрик нужен непрерывный ряд "
            f"с {required_start.isoformat()} по {required_end.isoformat()}"
        )
    models: dict[str, ForecastModel] = {
        "historical_quantile": HistoricalQuantileBaseline(),
        "random_walk": RandomWalkBaseline(),
    }
    periods: dict[str, object] = {}
    for period in EVALUATED_PERIODS:
        start, end = period_bounds(period)
        model_scores: dict[str, dict[str, object]] = {}
        for name, model in models.items():
            horizons: dict[str, object] = {}
            for horizon in (1, 7):
                horizons[str(horizon)] = _result_dict(
                    evaluate_split(candles, horizon, period, model)
                )
            model_scores[name] = horizons
        periods[period.value] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "models": model_scores,
        }
    holdout_start, holdout_end = period_bounds(Period.HOLDOUT)
    leakage = []
    if last >= date(2025, 1, 1):
        leakage.extend(audit_period_leak(candles, Period.VALIDATION, date(2025, 1, 1)))
    if last >= date(2023, 1, 1):
        leakage.extend(audit_period_leak(candles, Period.TRAINING, date(2023, 1, 1)))
    return {
        "asset": "BTCUSDT",
        "market": "spot",
        "interval": "1d",
        "rows": len(candles),
        "first_open": candles[0].timestamp.isoformat(),
        "last_open": candles[-1].timestamp.isoformat(),
        "periods": periods,
        "holdout": {
            "start": holdout_start.isoformat(),
            "end": holdout_end.isoformat(),
            "metrics": None,
            "reason": "закрытый период: метрики не считались",
        },
        "leakage_checks": leakage,
    }


def render_markdown(payload: dict[str, object]) -> str:
    """Текст отчёта из уже посчитанного словаря. Holdout остаётся без метрик."""
    holdout = payload["holdout"]
    if not isinstance(holdout, dict) or holdout.get("metrics") is not None:
        raise ValueError("отчёт содержит метрики holdout")
    lines = [
        "# Контрольные прогнозы BTCUSDT, 2019–2024",
        "",
        "Числа ниже — ошибки хронологической проверки на прошлых закрытиях Binance Spot.",
        "Это не цена на будущее и не заявка.",
        "",
        "## Периоды",
        "",
        "Границы записаны в `wolfai.forecasting.splits` до расчёта.",
        "",
        "| Период | Начало | Конец | Метрики |",
        "| --- | --- | --- | --- |",
    ]
    periods = payload["periods"]
    if not isinstance(periods, dict):
        raise ValueError("в отчёте нет периодов")
    for name in ("training", "validation"):
        block = periods[name]
        if not isinstance(block, dict):
            raise ValueError(f"период {name} повреждён")
        lines.append(f"| {name} | {block['start']} | {block['end']} | посчитаны |")
    lines.append(
        f"| holdout | {holdout['start']} | {holdout['end']} | не считались |"
    )
    lines.extend([
        "",
        "## Ряд",
        "",
        f"- Свечей в файле: {payload['rows']}",
        f"- Первое открытие: `{payload['first_open']}`",
        f"- Последнее открытие: `{payload['last_open']}`",
        "- Цена — close BTCUSDT Spot. Метка свечи — открытие суток UTC.",
        "- `as_of` прогноза — `daily_bar_available_at` последней свечи в истории, на одни сутки позже её открытия.",
        "- Горизонт h — close через h суток после последнего известного close.",
        "- Архивы и SHA-256 лежат в `data/raw/binance/manifest.json`. CSV и ZIP в Git не входят.",
        "",
        "## Определения",
        "",
        "- MAE — средняя абсолютная ошибка медианы, в USDT.",
        "- MAPE — средняя абсолютная процентная ошибка медианы.",
        "- Directional accuracy — доля совпавших знаков доходности медианы и факта. Нулевой прогноз знака, как у случайного блуждания, в долю не входит.",
        "- Покрытие — доля фактов внутри интервала квантилей 0.1 и 0.9. Номинал интервала 80%.",
        "- Ошибка интервала — средний Winkler score этого интервала (alpha = 0.2), в USDT. Учитываются ширина и промах.",
        "- Квантильный baseline переносит прошлые доходности горизонта на последний close.",
        "- Случайное блуждание ставит медиану на последний close и центрирует тот же исторический разброс вокруг нулевого сноса.",
        "",
        "## Метрики",
        "",
        "| Период | Модель | h | N | MAE | MAPE | Направление | Вызовы | Покрытие | Ширина | Winkler |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    labels = {
        "historical_quantile": "квантили",
        "random_walk": "блуждание",
    }
    for name in ("training", "validation"):
        block = periods[name]
        if not isinstance(block, dict):
            continue
        models = block["models"]
        if not isinstance(models, dict):
            continue
        for model_name, horizons in models.items():
            if not isinstance(horizons, dict):
                continue
            for horizon, metrics in horizons.items():
                if not isinstance(metrics, dict):
                    continue
                lines.append(
                    "| {period} | {model} | {horizon} | {n} | {mae} | {mape} | {direction} | {calls} | {coverage} | {width} | {score} |".format(
                        period=name,
                        model=labels.get(str(model_name), model_name),
                        horizon=horizon,
                        n=metrics["observations"],
                        mae=_num(metrics["mean_absolute_error"], 2),
                        mape=_percent(metrics["mean_absolute_percentage_error"]),
                        direction=_percent(metrics["directional_accuracy"]),
                        calls=metrics["directional_calls"],
                        coverage=_percent(metrics["interval_coverage"]),
                        width=_num(metrics["mean_interval_width"], 2),
                        score=_num(metrics["mean_interval_score"], 2),
                    )
                )
    lines.extend([
        "",
        "Ширина интервала у двух моделей совпадает: случайное блуждание берёт тот же исторический разброс и сдвигает его к нулевому сносу.",
        "",
        "## Утечка",
        "",
        "Проверка переписывает close начиная с указанной даты и повторяет метрики.",
        "Совпадение значит, что эти close в расчёт периода не входили.",
        "",
    ])
    checks = payload.get("leakage_checks", [])
    if isinstance(checks, list) and checks:
        lines.append("| Период | Горизонт | Модель | Переписано с | Совпало |")
        lines.append("| --- | ---: | --- | --- | --- |")
        for item in checks:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {item['period']} | {item['horizon_days']} | {item['model']} | {item['mutate_from']} | {item['passed']} |"
            )
    else:
        lines.append("Проверка на этом запуске не записана.")
    lines.extend([
        "",
        "## Holdout",
        "",
        f"{holdout['reason']}. Границы: {holdout['start']} — {holdout['end']}.",
        "",
    ])
    return "\n".join(lines)


def write_report(candles: list[Candle], markdown_path: Path, json_path: Path) -> dict[str, object]:
    payload = score_controls(candles)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _result_dict(result: WalkForwardResult) -> dict[str, object]:
    return {
        "observations": result.observations,
        "mean_absolute_error": result.mean_absolute_error,
        "mean_absolute_percentage_error": result.mean_absolute_percentage_error,
        "directional_accuracy": result.directional_accuracy,
        "directional_calls": result.directional_calls,
        "interval_coverage": result.interval_coverage,
        "mean_interval_width": result.mean_interval_width,
        "mean_interval_score": result.mean_interval_score,
    }


def _rewrite_from(candles: list[Candle], start: date) -> list[Candle]:
    rewritten: list[Candle] = []
    for candle in candles:
        if candle.timestamp.date() < start:
            rewritten.append(candle)
            continue
        close = 50.0
        rewritten.append(Candle(
            timestamp=candle.timestamp,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=candle.volume,
        ))
    return rewritten


def _num(value: object, digits: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{digits}f}"


def _percent(value: object) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value * 100:.2f}%"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Контрольные метрики WolfAi без holdout")
    parser.add_argument("--csv", required=True, help="Нормализованный BTCUSDT 1d CSV")
    parser.add_argument("--markdown", default="docs/BASELINE_2019_2024.md")
    parser.add_argument("--json", default="docs/BASELINE_2019_2024.json")
    args = parser.parse_args()
    candles = load_candles(args.csv)
    write_report(candles, Path(args.markdown), Path(args.json))
    print(f"Report written to {args.markdown}")


if __name__ == "__main__":
    main()
