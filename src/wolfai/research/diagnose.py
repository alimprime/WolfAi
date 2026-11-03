"""Диагностика фиксированных формул. В промпт поиска не входят метрики 2023–2024."""

from __future__ import annotations

from datetime import date

from wolfai.data.candles import Candle
from wolfai.forecasting.splits import period_bounds, Period
from wolfai.research.features import FEATURE_NAMES, FeatureRow, build_rows, load_research_candles
from wolfai.research.formula import EvalState, evaluate, extract_linear_terms, parse_formula
from wolfai.research.journal import HypothesisRecord, Journal

FIT_START, FIT_END = date(2019, 1, 1), date(2021, 12, 31)


def prompt_diagnosis(rows: list[FeatureRow]) -> str:
    """Текст для Qwen только по окну подбора коэффициентов."""
    window = _between_raw(rows, FIT_START, FIT_END)
    if len(window) < 30:
        return "В окне 2019–2021 мало точек для диагностики масштаба."
    lines = [
        "Диагностика сырых признаков на 2019–2021, до подбора новых формул.",
        "Прошлые фиксированные формулы почти не сдвигали sigmoid, потому что признак умножался на коэффициент порядка единицы, а сам признак был на порядки меньше.",
        "Ниже среднее, стандартное отклонение и знак связи с ростом следующего close.",
    ]
    labels = [row.y_up for row in window]
    base = sum(labels) / len(labels)
    lines.append(f"Доля роста в этом окне: {base:.4f}.")
    for name in FEATURE_NAMES:
        values = [row.features[name] for row in window]
        mean = sum(values) / len(values)
        deviation = _std(values)
        link = _association(window, name)
        lines.append(
            f"- {name}: mean={mean:.6f}, std={deviation:.6f}, "
            f"связь={link:.6f} ({_link_word(link, deviation)})"
        )
    lines.append(
        "Python сам оценит свободный член и веса после стандартизации. "
        "В структуру клади преобразование identity, abs, tanh или sign, не числовой коэффициент."
    )
    return "\n".join(lines)


def diagnose_records(rows: list[FeatureRow], records: list[HypothesisRecord]) -> list[dict[str, object]]:
    """Разбор уже сохранённых фиксированных формул, включая ретроспективу 2023–2024."""
    windows = {
        "fit_2019_2021": _between_raw(rows, FIT_START, FIT_END),
        "training_2019_2022": _between_raw(rows, date(2019, 1, 1), date(2022, 12, 31)),
        "development_2023_2024": _between_raw(
            rows,
            period_bounds(Period.VALIDATION)[0],
            period_bounds(Period.VALIDATION)[1],
        ),
    }
    reports = []
    for record in records:
        if not record.expression.startswith("sigmoid"):
            continue
        try:
            tree = parse_formula(record.expression)
        except ValueError:
            continue
        terms = extract_linear_terms(tree) or []
        blocks = {}
        for name, window in windows.items():
            blocks[name] = _window_report(tree, terms, window)
        reports.append({
            "id": record.id,
            "expression": record.expression,
            "stored_validation_brier": (record.validation_metrics or {}).get("brier"),
            "stored_train_brier": (record.train_metrics or {}).get("brier"),
            "terms": [
                _term_report(coefficient, feature, windows)
                for coefficient, feature in terms
            ],
            "windows": blocks,
        })
    return reports


def render_diagnosis(reports: list[dict[str, object]]) -> str:
    lines = [
        "# Диагностика фиксированных формул Phase 2",
        "",
        "Пять формул Qwen оценивались с коэффициентами, которые предложила модель.",
        "Период 2023–2024 здесь разобран ретроспективно: эти числа уже были в журнале и не подаются в новый поиск.",
        "",
        "Причина концентрации около 0.5: логит равен сумме коэффициента и сырого признака.",
        "Сырые momentum и volatility имеют стандартное отклонение в сотых и меньше, а коэффициенты порядка единицы.",
        "Типичный сдвиг логита много меньше 1, поэтому sigmoid остаётся у 0.5.",
        "Исключение — range_position: он уже лежит около единицы, и коэффициент 0.4 сдвигает среднюю вероятность к 0.57 без настоящей связи.",
        "Свободного члена в формулах не было, базовая доля роста не оценивалась.",
        "Малый шум вокруг 0.5 увеличивает Brier относительно константы 0.5, если средний прогноз не совпадает с долей роста или дисперсия не несёт связи.",
        "",
    ]
    for report in reports:
        lines.append(f"## `{report['expression']}`")
        lines.append("")
        lines.append(f"Журнал: train Brier `{_num(report['stored_train_brier'])}`, development-validation Brier `{_num(report['stored_validation_brier'])}`.")
        lines.append("")
        lines.append("| Признак | Коэффициент | Типичный логит | Связь 2019–2021 | Связь 2023–2024 |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        terms = report["terms"]
        if isinstance(terms, list):
            for term in terms:
                if not isinstance(term, dict):
                    continue
                lines.append(
                    f"| {term['feature']} | {_num(term['coefficient'])} | {_num(term['typical_logit'])} | "
                    f"{term['fit_link']} | {term['development_link']} |"
                )
        lines.append("")
        windows = report["windows"]
        if isinstance(windows, dict):
            lines.append("| Окно | N | Средняя p | Std p | Доля |p−0.5|≤0.02 | Brier | Brier 0.5 |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for name, block in windows.items():
                if not isinstance(block, dict):
                    continue
                lines.append(
                    f"| {name} | {block['observations']} | {_num(block['mean_probability'])} | "
                    f"{_num(block['probability_stdev'])} | {_num(block['near_half_share'])} | "
                    f"{_num(block['brier'])} | {_num(block['brier_constant'])} |"
                )
        lines.append("")
    lines.append("Знак коэффициента задаёт, куда формула двигает свою вероятность.")
    lines.append("Фактическая связь — разница среднего признака в дни роста и падения следующего close.")
    lines.append("Если связь почти нулевая, верный знак коэффициента всё равно не улучшает Brier: сдвига почти нет, а шум есть.")
    lines.append("")
    return "\n".join(lines)


def _window_report(tree, terms: list[tuple[float, str]], rows: list[FeatureRow]) -> dict[str, object]:
    if not rows:
        return {"observations": 0}
    probabilities = [evaluate(tree, row.features, EvalState()) for row in rows]
    labels = [row.y_up for row in rows]
    mean = sum(probabilities) / len(probabilities)
    deviation = _std(probabilities)
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(labels)
    return {
        "observations": len(rows),
        "mean_probability": mean,
        "probability_stdev": deviation,
        "near_half_share": sum(abs(probability - 0.5) <= 0.02 for probability in probabilities) / len(probabilities),
        "brier": brier,
        "brier_constant": 0.25,
        "base_rate": sum(labels) / len(labels),
    }


def _term_report(
    coefficient: float,
    feature: str,
    windows: dict[str, list[FeatureRow]],
) -> dict[str, object]:
    fit = windows["fit_2019_2021"]
    development = windows["development_2023_2024"]
    deviation = _std([row.features[feature] for row in fit]) if fit else 0.0
    fit_link = _association(fit, feature)
    development_link = _association(development, feature)
    return {
        "feature": feature,
        "coefficient": coefficient,
        "typical_logit": abs(coefficient) * deviation,
        "fit_link": _link_phrase(coefficient, fit_link, deviation),
        "development_link": _link_phrase(coefficient, development_link, _std([row.features[feature] for row in development]) if development else 0.0),
    }


def _between_raw(rows: list[FeatureRow], start: date, end: date) -> list[FeatureRow]:
    return [
        row for row in rows
        if start <= row.decision_date <= end and start <= row.target_date <= end
    ]


def _association(rows: list[FeatureRow], feature: str) -> float:
    positive = [row.features[feature] for row in rows if row.y_up == 1]
    negative = [row.features[feature] for row in rows if row.y_up == 0]
    if not positive or not negative:
        return 0.0
    return sum(positive) / len(positive) - sum(negative) / len(negative)


def _link_word(link: float, deviation: float) -> str:
    if deviation <= 0 or abs(link) < 0.05 * deviation:
        return "связи почти нет"
    return "рост признака чаще перед ростом" if link > 0 else "рост признака чаще перед падением"


def _link_phrase(coefficient: float, link: float, deviation: float) -> str:
    relation = _link_word(link, deviation)
    if relation == "связи почти нет":
        return relation
    coefficient_positive = coefficient > 0
    link_positive = link > 0
    agreement = "знак совпал" if coefficient_positive == link_positive else "знак противоположен"
    return f"{relation}; {agreement}"


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _num(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.4f}"


def main() -> None:
    from pathlib import Path

    candles = load_research_candles("data/btcusdt_1d_2019-01_2026-09.csv")
    journal = Journal("data/research/hypotheses.sqlite")
    try:
        records = [
            row for row in journal.list_hypotheses()
            if (row.kind or "fixed") == "fixed" and row.expression.startswith("sigmoid")
            and "z_" not in row.expression
        ]
    finally:
        journal.close()
    text = render_diagnosis(diagnose_records(build_rows(candles), records))
    Path("docs/HYPOTHESIS_DIAGNOSIS.md").write_text(text, encoding="utf-8")
    print(f"diagnosis formulas={text.count('## `sigmoid')}")


if __name__ == "__main__":
    main()
