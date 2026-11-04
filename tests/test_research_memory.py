"""Реестр хранит повтор и отрицательный результат и не подтверждает 2021."""

import pytest

from wolfai.research.research_memory import ResearchMemory, experiment_key


def _payload(status: str = "insufficient_evidence", brier: float = 0.25) -> dict[str, object]:
    return {
        "model_name": "logistic_price",
        "config_hash": "abc",
        "dataset_fingerprint": "fff",
        "features": ["log_return"],
        "label_definition": "y = 1 только если close вырос",
        "horizon_days": 1,
        "train_start": "2019-02-01T00:00:00+00:00",
        "train_end": "2020-12-31T00:00:00+00:00",
        "dev_start": "2021-01-01T00:00:00+00:00",
        "dev_end": "2021-12-31T00:00:00+00:00",
        "seed": 1729,
        "brier": brier,
        "log_loss": 0.69,
        "observations": 100,
        "hypotheses_examined": 9,
        "evidence_role": "development_only",
        "claim_status": status,
        "provisional": False,
        "limitations": "слабее константы",
        "code_version": "test",
    }


def test_duplicate_and_negative_result_stay_one_record(tmp_path) -> None:
    memory = ResearchMemory(tmp_path / "registry.sqlite")
    first = _payload()
    assert memory.record(first) == "stored"
    assert memory.record(dict(first)) == "duplicate"
    assert memory.count() == 1
    stored = memory.find(experiment_key(first))
    assert stored is not None
    assert stored["claim_status"] == "insufficient_evidence"
    assert stored["brier"] == 0.25
    memory.close()


def test_development_window_cannot_be_stored_as_supported(tmp_path) -> None:
    memory = ResearchMemory(tmp_path / "registry.sqlite")
    with pytest.raises(ValueError, match="supported"):
        memory.record(_payload("supported", 0.1))
    assert memory.count() == 0
    memory.close()
