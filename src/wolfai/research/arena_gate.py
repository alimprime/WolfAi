"""Минимальный вход для нескольких стратегий.

Турнира, голосования и копий Qwen здесь нет. Прогноз и оценка идут через
один и тот же журнал и `evaluate_settled`. Стратегия не получает чужую
историю и не назначает себе статус подтверждённой закономерности.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from wolfai.data.spot_live import SpotCandle, SpotSnapshot
from wolfai.research.forward_eval import evaluate_settled
from wolfai.research.forward_mode import features_visible_at, run_forward_cycle
from wolfai.research.hypothesis_engine import HypothesisStore, accept_proposal
from wolfai.research.hypothesis_generator import StructuredHypothesis
from wolfai.research.prediction_ledger import PredictionLedger
from wolfai.research.quality_contract import STATUS_SUPPORTED
from wolfai.research.strategy import FeatureView, ResearchStrategy


class ArenaGate:
    def __init__(self, store: HypothesisStore) -> None:
        self.store = store
        self._strategies: dict[str, ResearchStrategy] = {}

    def register(self, strategy: ResearchStrategy) -> str:
        if not strategy.strategy_id.strip() or not strategy.version.strip():
            raise PermissionError("у стратегии нет идентификатора или версии")
        current = self._strategies.get(strategy.strategy_id)
        if current is not None and current.version == strategy.version:
            return "duplicate"
        kind = "agent_registration" if current is None else "hyperparameter"
        self._strategies[strategy.strategy_id] = strategy
        self.store.note_attempt(
            kind=kind,
            author_id=strategy.strategy_id,
            succeeded=True,
            detail=strategy.version,
        )
        return "registered" if current is None else "version_changed"

    def propose(
        self,
        agent_id: str,
        proposal: StructuredHypothesis,
        *,
        created_at: datetime,
        data_hash: str,
    ) -> tuple[str, str]:
        self._require(agent_id)
        if proposal.author_id != agent_id:
            raise PermissionError("стратегия не может записать чужую гипотезу")
        return accept_proposal(
            self.store,
            proposal,
            created_at=created_at,
            data_hash=data_hash,
        )

    def snapshot(
        self,
        agent_id: str,
        candles: Sequence[SpotCandle],
        decision_time: datetime,
    ) -> FeatureView:
        """Снимок признаков без будущих свечей и без метки исхода."""
        self._require(agent_id)
        features = features_visible_at(candles, decision_time)
        return FeatureView(
            decision_time=decision_time,
            features=features,
            feature_available_at={name: decision_time for name in features},
        )

    def submit(
        self,
        agent_id: str,
        ledger: PredictionLedger,
        market: SpotSnapshot,
        *,
        commit: bool = False,
        horizon_days: int = 1,
    ) -> dict[str, object]:
        strategy = self._require(agent_id)
        result = run_forward_cycle(
            ledger=ledger,
            strategy=strategy,
            snapshot=market,
            horizon_days=horizon_days,
            commit=commit,
        )
        result["strategy_id"] = strategy.strategy_id
        return result

    def evaluation(self, ledger: PredictionLedger, channel: str) -> dict[str, object]:
        """Одна оценка на всех. Метод стратегии сюда не подставляется."""
        report = evaluate_settled(ledger, channel)
        if report.get("claim_status") == STATUS_SUPPORTED or report.get("validated_law") is True:
            raise RuntimeError("общая оценка не подтверждает закон")
        return report

    def own_history(self, agent_id: str) -> dict[str, object]:
        self._require(agent_id)
        hypotheses = [
            item for item in self.store.list_all() if item.author_id == agent_id
        ]
        return {
            "hypotheses": hypotheses,
            "experiments": self.store.experiments_for(agent_id),
        }

    def _require(self, agent_id: str) -> ResearchStrategy:
        try:
            return self._strategies[agent_id]
        except KeyError as error:
            raise PermissionError("стратегия не зарегистрирована") from error
