"""Жизненный цикл гипотез этапа 2.6.

Статусы: proposed, testing, insufficient_evidence, rejected, candidate, archived.
`candidate` — перспективная, но не подтверждённая гипотеза. Это не `supported`.
Переход в Validated Law запрещён. Отрицательный результат не удаляется.
Формула после ошибки не переписывается.

Правила противоречий зафиксированы в `RULE_TEXT` и не подстраиваются
под отдельный убыток.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from wolfai.research.formula import FEATURES, FormulaError, features_used, parse_formula
from wolfai.research.generator import expression_is_safe
from wolfai.research.hypothesis_generator import StructuredHypothesis
from wolfai.research.quality_contract import (
    dependence_block_length,
    linear_percentile,
    moving_block_means,
    point_brier,
)

RULE_VERSION = "2.6.0"
MIN_OBSERVATIONS = 30
RECENT_WINDOW = 20
APPLICABILITY_FRACTION = 0.8
REGIME_FRACTION = 0.8
RULE_TEXT = (
    "Одна ошибка при менее чем 30 наблюдениях не создаёт противоречие и не переводит "
    "гипотезу в rejected. rejected только если наблюдений не меньше 30, блочный интервал "
    "улучшения против константы 0.5 целиком ниже нуля, среднее поздней половины ниже нуля "
    "и либо обе половины ниже нуля, либо поздняя строго хуже ранней. Доля последних 20 точек "
    "вне области применимости или вне заявленных режимов не ниже 0.8 создаёт противоречие "
    "без смены формулы и без rejected. Интервал, который содержит ноль, считается совместимым "
    "с шумом. Формула после ошибки не переписывается. supported и Validated Law не назначаются."
)
STATUSES = frozenset({
    "proposed",
    "testing",
    "insufficient_evidence",
    "rejected",
    "candidate",
    "archived",
})
FORBIDDEN_STATUSES = frozenset({"supported", "validated_law", "validated law"})
TRANSITIONS = {
    "proposed": frozenset({"testing", "insufficient_evidence", "archived"}),
    "testing": frozenset({"insufficient_evidence", "rejected", "candidate", "archived"}),
    "insufficient_evidence": frozenset({"testing", "rejected", "candidate", "archived"}),
    "candidate": frozenset({"testing", "insufficient_evidence", "rejected", "archived"}),
    "rejected": frozenset({"archived"}),
    "archived": frozenset(),
}
HORIZONS = frozenset({1, 3, 7})


class HypothesisLifecycleError(ValueError):
    """Нарушение жизненного цикла, а не рыночный вывод."""


@dataclass(frozen=True)
class HypothesisView:
    hypothesis_id: str
    author_id: str
    name: str
    version: int
    description: str
    creation_reason: str
    features: tuple[str, ...]
    horizon_days: int
    expression: str
    applicability: dict[str, object]
    created_at: datetime
    data_hash: str
    status: str
    parent_hypothesis_id: str | None
    claim_status: str


@dataclass(frozen=True)
class HypothesisObservation:
    probability: float
    label: int
    regime: str
    in_domain: bool
    decision_date: date


@dataclass(frozen=True)
class ContradictionAssessment:
    kind: str
    reject: bool
    create_contradiction: bool
    create_task: bool
    status_to: str | None
    single_error: bool
    observations: int
    detail: dict[str, object]


class HypothesisStore:
    """Память гипотез, противоречий и попыток. Строки не удаляются."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                creation_reason TEXT NOT NULL,
                features_json TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                expression TEXT NOT NULL,
                applicability_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                parent_hypothesis_id TEXT,
                claim_status TEXT NOT NULL,
                formula_key TEXT NOT NULL UNIQUE
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hypothesis_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS contradictions (
                contradiction_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_tasks (
                task_id TEXT PRIMARY KEY,
                contradiction_id TEXT NOT NULL,
                parent_hypothesis_id TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hypothesis_experiments (
                experiment_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage (
                parent_hypothesis_id TEXT NOT NULL,
                derived_hypothesis_id TEXT NOT NULL,
                contradiction_id TEXT,
                experiment_id TEXT,
                PRIMARY KEY (parent_hypothesis_id, derived_hypothesis_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                author_id TEXT NOT NULL,
                succeeded INTEGER NOT NULL,
                detail TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def create(
        self,
        *,
        author_id: str,
        name: str,
        description: str,
        creation_reason: str,
        features: Sequence[str],
        horizon_days: int,
        expression: str,
        applicability: Mapping[str, object],
        created_at: datetime,
        data_hash: str,
        parent_hypothesis_id: str | None = None,
        contradiction_id: str | None = None,
        experiment_id: str | None = None,
        version: int | None = None,
    ) -> tuple[str, str]:
        ordered = _validate_formula(
            author_id=author_id,
            name=name,
            description=description,
            creation_reason=creation_reason,
            features=features,
            horizon_days=horizon_days,
            expression=expression,
            applicability=applicability,
            data_hash=data_hash,
        )
        _aware(created_at)
        key = formula_key(expression, ordered, horizon_days, applicability)
        existing = self._connection.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE formula_key = ?",
            (key,),
        ).fetchone()
        if existing is not None:
            return str(existing[0]), "duplicate"
        if parent_hypothesis_id is not None:
            parent = self.get(parent_hypothesis_id)
            if version is None:
                version = parent.version + 1
            if key == _stored_key(parent):
                raise HypothesisLifecycleError("идентичная формула не становится потомком самой себя")
            if key in self._ancestor_ids(parent_hypothesis_id):
                raise HypothesisLifecycleError("циклическое происхождение гипотез")
        elif version is None:
            version = 1
        self._connection.execute(
            """
            INSERT INTO hypotheses (
                hypothesis_id, author_id, name, version, description, creation_reason,
                features_json, horizon_days, expression, applicability_json, created_at,
                data_hash, status, parent_hypothesis_id, claim_status, formula_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, 'not_confirmed', ?)
            """,
            (
                key,
                author_id.strip(),
                name.strip(),
                version,
                description.strip(),
                creation_reason.strip(),
                json.dumps(list(ordered), ensure_ascii=False),
                horizon_days,
                expression.strip(),
                _canon(applicability),
                created_at.isoformat(),
                data_hash.strip(),
                parent_hypothesis_id,
                key,
            ),
        )
        self._event(key, "created", {"creation_reason": creation_reason.strip()}, created_at)
        self._connection.commit()
        if parent_hypothesis_id is not None:
            self.link_derived(
                parent_hypothesis_id,
                key,
                contradiction_id=contradiction_id,
                experiment_id=experiment_id,
            )
        self.note_attempt(
            kind="formula",
            author_id=author_id,
            succeeded=True,
            detail=key,
            created_at=created_at,
        )
        return key, "stored"

    def link_derived(
        self,
        parent_hypothesis_id: str,
        derived_hypothesis_id: str,
        *,
        contradiction_id: str | None = None,
        experiment_id: str | None = None,
    ) -> str:
        self.get(parent_hypothesis_id)
        self.get(derived_hypothesis_id)
        if parent_hypothesis_id == derived_hypothesis_id:
            raise HypothesisLifecycleError("циклическое происхождение гипотез")
        if derived_hypothesis_id in self._ancestor_ids(parent_hypothesis_id):
            raise HypothesisLifecycleError("циклическое происхождение гипотез")
        if parent_hypothesis_id in self._descendant_ids(derived_hypothesis_id):
            raise HypothesisLifecycleError("циклическое происхождение гипотез")
        existing = self._connection.execute(
            """
            SELECT contradiction_id, experiment_id FROM lineage
            WHERE parent_hypothesis_id = ? AND derived_hypothesis_id = ?
            """,
            (parent_hypothesis_id, derived_hypothesis_id),
        ).fetchone()
        if existing is not None:
            return "duplicate"
        self._connection.execute(
            """
            INSERT INTO lineage (
                parent_hypothesis_id, derived_hypothesis_id, contradiction_id, experiment_id
            ) VALUES (?, ?, ?, ?)
            """,
            (parent_hypothesis_id, derived_hypothesis_id, contradiction_id, experiment_id),
        )
        self._connection.commit()
        return "stored"

    def get(self, hypothesis_id: str) -> HypothesisView:
        row = self._connection.execute(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            raise KeyError(hypothesis_id)
        return _view(row)

    def list_all(self) -> list[HypothesisView]:
        rows = self._connection.execute(
            "SELECT * FROM hypotheses ORDER BY created_at, hypothesis_id"
        ).fetchall()
        return [_view(row) for row in rows]

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()
        return int(row[0])

    def transition(self, hypothesis_id: str, status: str, reason: str, *, when: datetime | None = None) -> None:
        normalized = status.strip().lower()
        if normalized in FORBIDDEN_STATUSES or "validated" in normalized:
            raise HypothesisLifecycleError("переход в Validated Law запрещён")
        if normalized not in STATUSES:
            raise HypothesisLifecycleError("неизвестный статус гипотезы")
        if not reason.strip():
            raise HypothesisLifecycleError("причина перехода обязательна")
        if normalized == "candidate" and ("supported" in reason.lower() or "validated" in reason.lower()):
            raise HypothesisLifecycleError("candidate не означает подтверждение")
        current = self.get(hypothesis_id)
        if current.status == normalized:
            return
        if normalized not in TRANSITIONS[current.status]:
            raise HypothesisLifecycleError(
                f"переход {current.status} -> {normalized} не входит в правила"
            )
        moment = when or current.created_at
        self._connection.execute(
            "UPDATE hypotheses SET status = ? WHERE hypothesis_id = ?",
            (normalized, hypothesis_id),
        )
        self._event(
            hypothesis_id,
            "status",
            {"from": current.status, "to": normalized, "reason": reason.strip(), "claim_status": "not_confirmed"},
            moment,
        )
        self._connection.commit()

    def mark_candidate(self, hypothesis_id: str, reason: str, *, when: datetime | None = None) -> None:
        self.transition(hypothesis_id, "candidate", reason, when=when)
        if self.confirms_law(hypothesis_id):
            raise HypothesisLifecycleError("candidate не подтверждает закон")

    def confirms_law(self, hypothesis_id: str) -> bool:
        self.get(hypothesis_id)
        return False

    def add_experiment(
        self,
        hypothesis_id: str,
        payload: Mapping[str, object],
        *,
        when: datetime | None = None,
    ) -> tuple[str, str]:
        current = self.get(hypothesis_id)
        if payload.get("claim_status") == "supported" or payload.get("validated_law") is True:
            raise HypothesisLifecycleError("эксперимент не назначает supported и не создаёт Validated Law")
        moment = when or current.created_at
        experiment_id = hashlib.sha256(
            json.dumps(
                {"hypothesis_id": hypothesis_id, "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        existing = self._connection.execute(
            "SELECT experiment_id FROM hypothesis_experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if existing is not None:
            return experiment_id, "duplicate"
        self._connection.execute(
            """
            INSERT INTO hypothesis_experiments (
                experiment_id, hypothesis_id, author_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                hypothesis_id,
                current.author_id,
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str),
                moment.isoformat(),
            ),
        )
        self._connection.commit()
        if current.status == "proposed":
            self.transition(hypothesis_id, "testing", "эксперимент зарегистрирован", when=moment)
        return experiment_id, "stored"

    def experiments_for(self, author_id: str) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT experiment_id, hypothesis_id, payload_json FROM hypothesis_experiments
            WHERE author_id = ? ORDER BY created_at, experiment_id
            """,
            (author_id,),
        ).fetchall()
        found: list[dict[str, object]] = []
        for row in rows:
            payload = json.loads(row[2])
            payload["experiment_id"] = row[0]
            payload["hypothesis_id"] = row[1]
            found.append(payload)
        return found

    def experiment_count(self, hypothesis_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM hypothesis_experiments WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return int(row[0])

    def add_contradiction(
        self,
        hypothesis_id: str,
        kind: str,
        payload: Mapping[str, object],
        *,
        when: datetime | None = None,
    ) -> tuple[str, str]:
        current = self.get(hypothesis_id)
        body = dict(payload)
        body["expression"] = current.expression
        body["rule_version"] = RULE_VERSION
        contradiction_id = hashlib.sha256(
            _canon({"hypothesis_id": hypothesis_id, "kind": kind, "payload": body}).encode()
        ).hexdigest()
        existing = self._connection.execute(
            "SELECT contradiction_id FROM contradictions WHERE contradiction_id = ?",
            (contradiction_id,),
        ).fetchone()
        if existing is not None:
            return contradiction_id, "duplicate"
        moment = when or current.created_at
        self._connection.execute(
            """
            INSERT INTO contradictions (
                contradiction_id, hypothesis_id, kind, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (contradiction_id, hypothesis_id, kind, _canon(body), moment.isoformat()),
        )
        self._connection.commit()
        return contradiction_id, "stored"

    def contradictions_for(self, hypothesis_id: str) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT contradiction_id, kind, payload_json FROM contradictions
            WHERE hypothesis_id = ? ORDER BY created_at, contradiction_id
            """,
            (hypothesis_id,),
        ).fetchall()
        return [
            {"contradiction_id": row[0], "kind": row[1], **json.loads(row[2])}
            for row in rows
        ]

    def add_task(
        self,
        *,
        contradiction_id: str,
        parent_hypothesis_id: str,
        question: str,
        when: datetime | None = None,
    ) -> tuple[str, str]:
        self.get(parent_hypothesis_id)
        if not question.strip():
            raise HypothesisLifecycleError("вопрос исследовательской задачи пуст")
        task_id = hashlib.sha256(
            f"{contradiction_id}:{parent_hypothesis_id}:{question.strip()}".encode()
        ).hexdigest()
        existing = self._connection.execute(
            "SELECT task_id FROM research_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if existing is not None:
            return task_id, "duplicate"
        moment = when or self.get(parent_hypothesis_id).created_at
        self._connection.execute(
            """
            INSERT INTO research_tasks (
                task_id, contradiction_id, parent_hypothesis_id, question, status, created_at
            ) VALUES (?, ?, ?, ?, 'proposed', ?)
            """,
            (task_id, contradiction_id, parent_hypothesis_id, question.strip(), moment.isoformat()),
        )
        self._connection.commit()
        return task_id, "stored"

    def tasks_for(self, hypothesis_id: str) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT task_id, contradiction_id, question, status FROM research_tasks
            WHERE parent_hypothesis_id = ?
            """,
            (hypothesis_id,),
        ).fetchall()
        return [
            {
                "task_id": row[0],
                "contradiction_id": row[1],
                "parent_hypothesis_id": hypothesis_id,
                "question": row[2],
                "status": row[3],
            }
            for row in rows
        ]

    def derived_ids(self, hypothesis_id: str) -> list[str]:
        rows = self._connection.execute(
            """
            SELECT derived_hypothesis_id FROM lineage
            WHERE parent_hypothesis_id = ? ORDER BY derived_hypothesis_id
            """,
            (hypothesis_id,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def note_attempt(
        self,
        *,
        kind: str,
        author_id: str,
        succeeded: bool,
        detail: str,
        created_at: datetime | None = None,
    ) -> str:
        moment = created_at or datetime.now(timezone.utc)
        _aware(moment)
        attempt_id = hashlib.sha256(
            _canon({
                "kind": kind,
                "author_id": author_id,
                "succeeded": succeeded,
                "detail": detail,
                "created_at": moment.isoformat(),
            }).encode()
        ).hexdigest()
        existing = self._connection.execute(
            "SELECT attempt_id FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if existing is not None:
            return attempt_id
        self._connection.execute(
            """
            INSERT INTO attempts (attempt_id, kind, author_id, succeeded, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (attempt_id, kind, author_id, 1 if succeeded else 0, detail, moment.isoformat()),
        )
        self._connection.commit()
        return attempt_id

    def attempt_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM attempts").fetchone()
        return int(row[0])

    def _ancestor_ids(self, hypothesis_id: str) -> set[str]:
        found: set[str] = set()
        current = hypothesis_id
        while current:
            if current in found:
                raise HypothesisLifecycleError("циклическое происхождение гипотез")
            found.add(current)
            row = self._connection.execute(
                "SELECT parent_hypothesis_id FROM hypotheses WHERE hypothesis_id = ?",
                (current,),
            ).fetchone()
            current = None if row is None or row[0] is None else str(row[0])
        return found

    def _descendant_ids(self, hypothesis_id: str) -> set[str]:
        found: set[str] = set()
        stack = [hypothesis_id]
        while stack:
            current = stack.pop()
            rows = self._connection.execute(
                "SELECT derived_hypothesis_id FROM lineage WHERE parent_hypothesis_id = ?",
                (current,),
            ).fetchall()
            for row in rows:
                child = str(row[0])
                if child in found:
                    raise HypothesisLifecycleError("циклическое происхождение гипотез")
                found.add(child)
                stack.append(child)
        return found

    def _event(self, hypothesis_id: str, event_type: str, payload: Mapping[str, object], when: datetime) -> None:
        self._connection.execute(
            """
            INSERT INTO hypothesis_events (hypothesis_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (hypothesis_id, event_type, _canon(payload), when.isoformat()),
        )


def accept_proposal(
    store: HypothesisStore,
    proposal: StructuredHypothesis,
    *,
    created_at: datetime,
    data_hash: str,
    contradiction_id: str | None = None,
    experiment_id: str | None = None,
) -> tuple[str, str]:
    """Принимает описание гипотезы. Текст выражения не исполняется как код."""
    try:
        if not expression_is_safe(proposal.expression):
            raise FormulaError("выражение содержит запрещённый символ")
        tree = parse_formula(proposal.expression)
        if features_used(tree) != set(proposal.features):
            raise FormulaError("набор признаков в формуле не совпадает с полями features")
        return store.create(
            author_id=proposal.author_id,
            name=proposal.name,
            description=proposal.description,
            creation_reason=proposal.creation_reason,
            features=proposal.features,
            horizon_days=proposal.horizon_days,
            expression=proposal.expression,
            applicability=proposal.applicability,
            created_at=created_at,
            data_hash=data_hash,
            parent_hypothesis_id=proposal.parent_hypothesis_id,
            contradiction_id=contradiction_id,
            experiment_id=experiment_id,
        )
    except (FormulaError, HypothesisLifecycleError, ValueError):
        store.note_attempt(
            kind="formula",
            author_id=proposal.author_id,
            succeeded=False,
            detail=proposal.expression[:120],
            created_at=created_at,
        )
        raise


def review(
    store: HypothesisStore,
    hypothesis_id: str,
    observations: Sequence[HypothesisObservation],
    *,
    when: datetime | None = None,
) -> ContradictionAssessment:
    """Применяет заранее заданные правила. Формулу не меняет и не удаляет."""
    before = store.get(hypothesis_id)
    assessment = assess(before, observations)
    moment = when or before.created_at
    if assessment.create_contradiction:
        contradiction_id, _state = store.add_contradiction(
            hypothesis_id,
            assessment.kind,
            assessment.detail,
            when=moment,
        )
        if assessment.create_task:
            store.add_task(
                contradiction_id=contradiction_id,
                parent_hypothesis_id=hypothesis_id,
                question=_task_question(assessment.kind),
                when=moment,
            )
    if assessment.status_to == "rejected" and before.status == "proposed":
        store.transition(hypothesis_id, "testing", "появились наблюдения", when=moment)
    if assessment.status_to is not None:
        store.transition(hypothesis_id, assessment.status_to, assessment.kind, when=moment)
    after = store.get(hypothesis_id)
    if after.expression != before.expression:
        raise HypothesisLifecycleError("формула изменилась после проверки")
    return assessment


def assess(
    hypothesis: HypothesisView,
    observations: Sequence[HypothesisObservation],
) -> ContradictionAssessment:
    """Чистая проверка. Статус и файлы не меняет."""
    sample = list(observations)
    count = len(sample)
    if count == 0:
        return _assessment("no_observations", observations=0)
    recent = sample[-RECENT_WINDOW:]
    if len(recent) >= RECENT_WINDOW:
        outside = sum(not item.in_domain for item in recent) / len(recent)
        if outside >= APPLICABILITY_FRACTION:
            return _assessment(
                "applicability_violation",
                create_contradiction=True,
                create_task=True,
                observations=count,
                detail={"outside_fraction": outside, "rule": RULE_TEXT},
            )
        regimes = hypothesis.applicability.get("regimes")
        if isinstance(regimes, list) and regimes and "any" not in regimes:
            shifted = sum(item.regime not in regimes for item in recent) / len(recent)
            if shifted >= REGIME_FRACTION:
                return _assessment(
                    "regime_shift",
                    create_contradiction=True,
                    create_task=True,
                    observations=count,
                    detail={"shifted_fraction": shifted, "regimes": list(regimes), "rule": RULE_TEXT},
                )
    misses = sum(_miss(item) for item in sample)
    last_miss = _miss(sample[-1])
    if count < MIN_OBSERVATIONS:
        if last_miss and misses == 1:
            return _assessment("single_error", single_error=True, observations=count)
        return _assessment(
            "insufficient_observations",
            status_to="insufficient_evidence",
            observations=count,
        )
    improvements = [_improvement(item) for item in sample]
    block, _overlapping = dependence_block_length(
        [item.decision_date for item in sample],
        hypothesis.horizon_days,
    )
    means = moving_block_means(improvements, block)
    low = linear_percentile(means, 0.025)
    high = linear_percentile(means, 0.975)
    midpoint = count // 2
    early = sum(improvements[:midpoint]) / midpoint
    late = sum(improvements[midpoint:]) / (count - midpoint)
    detail = {
        "low": low,
        "high": high,
        "early_mean": early,
        "late_mean": late,
        "rule": RULE_TEXT,
    }
    systematic = high < 0.0 and late < 0.0 and (early < 0.0 or late < early)
    if systematic:
        return _assessment(
            "systematic_deterioration",
            reject=True,
            create_contradiction=True,
            create_task=True,
            status_to="rejected",
            observations=count,
            detail=detail,
        )
    if last_miss and misses == 1:
        return _assessment("single_error", single_error=True, observations=count, detail=detail)
    if low < 0.0 < high:
        return _assessment("consistent_with_noise", observations=count, detail=detail)
    return _assessment("no_sustained_contradiction", observations=count, detail=detail)


def formula_key(
    expression: str,
    features: Sequence[str],
    horizon_days: int,
    applicability: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _canon({
            "expression": expression.strip(),
            "features": sorted(features),
            "horizon_days": horizon_days,
            "applicability": applicability,
        }).encode()
    ).hexdigest()


def _assessment(
    kind: str,
    *,
    reject: bool = False,
    create_contradiction: bool = False,
    create_task: bool = False,
    status_to: str | None = None,
    single_error: bool = False,
    observations: int,
    detail: dict[str, object] | None = None,
) -> ContradictionAssessment:
    return ContradictionAssessment(
        kind=kind,
        reject=reject,
        create_contradiction=create_contradiction,
        create_task=create_task,
        status_to=status_to,
        single_error=single_error,
        observations=observations,
        detail={} if detail is None else detail,
    )


def _task_question(kind: str) -> str:
    if kind == "systematic_deterioration":
        return "Разобрать устойчивое ухудшение. Прежнюю формулу не менять и не удалять."
    if kind == "regime_shift":
        return "Проверить, вышла ли гипотеза из заявленного режима. Формулу не переписывать."
    if kind == "applicability_violation":
        return "Проверить нарушение области применимости. Формулу не переписывать."
    return "Разобрать противоречие, не заменяя прежнюю формулу."


def _miss(observation: HypothesisObservation) -> bool:
    if observation.label not in (0, 1):
        raise HypothesisLifecycleError("исход должен быть 0 или 1")
    predicted_up = observation.probability >= 0.5
    return predicted_up != bool(observation.label)


def _improvement(observation: HypothesisObservation) -> float:
    return point_brier(0.5, observation.label) - point_brier(observation.probability, observation.label)


def _validate_formula(
    *,
    author_id: str,
    name: str,
    description: str,
    creation_reason: str,
    features: Sequence[str],
    horizon_days: int,
    expression: str,
    applicability: Mapping[str, object],
    data_hash: str,
) -> tuple[str, ...]:
    if not author_id.strip() or not name.strip() or not description.strip() or not creation_reason.strip():
        raise HypothesisLifecycleError("автор, имя, описание и причина обязательны")
    if not data_hash.strip():
        raise HypothesisLifecycleError("хэш исходных данных обязателен")
    if horizon_days not in HORIZONS:
        raise HypothesisLifecycleError("горизонт должен быть 1, 3 или 7 дней")
    if not expression_is_safe(expression):
        raise FormulaError("выражение содержит запрещённый символ")
    ordered = tuple(features)
    if not ordered or len(ordered) != len(set(ordered)):
        raise HypothesisLifecycleError("признаки пусты или повторяются")
    if any(feature not in FEATURES for feature in ordered):
        raise HypothesisLifecycleError("признак вне разрешённого набора")
    tree = parse_formula(expression)
    if features_used(tree) != set(ordered):
        raise FormulaError("набор признаков в формуле не совпадает с полями features")
    if not isinstance(applicability, Mapping):
        raise HypothesisLifecycleError("область применимости должна быть объектом")
    return ordered


def _stored_key(hypothesis: HypothesisView) -> str:
    return formula_key(
        hypothesis.expression,
        hypothesis.features,
        hypothesis.horizon_days,
        hypothesis.applicability,
    )


def _view(row: tuple[object, ...]) -> HypothesisView:
    applicability = json.loads(str(row[9]))
    features = json.loads(str(row[6]))
    if not isinstance(applicability, dict) or not isinstance(features, list):
        raise HypothesisLifecycleError("запись гипотезы повреждена")
    return HypothesisView(
        hypothesis_id=str(row[0]),
        author_id=str(row[1]),
        name=str(row[2]),
        version=int(row[3]),
        description=str(row[4]),
        creation_reason=str(row[5]),
        features=tuple(str(item) for item in features),
        horizon_days=int(row[7]),
        expression=str(row[8]),
        applicability=applicability,
        created_at=datetime.fromisoformat(str(row[10])),
        data_hash=str(row[11]),
        status=str(row[12]),
        parent_hypothesis_id=None if row[13] is None else str(row[13]),
        claim_status=str(row[14]),
    )


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("время гипотезы должно быть в UTC")
