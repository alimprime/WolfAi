"""Один локальный исследовательский запуск. Holdout не читается."""

from __future__ import annotations

import argparse
import json

from wolfai.data.history import HistoricalStore
from wolfai.research.adaptive import run_adaptive
from wolfai.research.engine import ExperimentMemory
from wolfai.research.experiment import run_research
from wolfai.research.features import load_research_candles
from wolfai.research.journal import Journal
from wolfai.research.lab import lab_report, run_qwen_lab
from wolfai.research.ollama import OllamaClient, ResearchOllama


def main() -> None:
    parser = argparse.ArgumentParser(description="WolfAi: гипотезы Qwen и проверка Python")
    parser.add_argument("--csv", default="data/btcusdt_1d_2019-01_2026-09.csv")
    parser.add_argument("--journal", default="data/research/hypotheses.sqlite")
    parser.add_argument("--store", default="data/history")
    parser.add_argument("--source", default="binance-vision")
    parser.add_argument("--memory", default="data/research/experiments.sqlite")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-hypotheses", type=int, default=5)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--lab", action="store_true")
    args = parser.parse_args()
    if args.lab and args.adaptive:
        raise SystemExit("режим lab не совмещается с adaptive")
    if args.lab:
        _run_lab(args)
        return
    candles = load_research_candles(args.csv)
    journal = Journal(args.journal)
    runner = run_adaptive if args.adaptive else run_research
    try:
        result = runner(
            OllamaClient(),
            candles,
            journal,
            max_hypotheses=args.max_hypotheses,
        )
    finally:
        rows = journal.list_hypotheses()
        journal.close()
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "stop_reason": result.stop_reason,
                "message": result.message,
                "hypotheses": len(result.hypothesis_ids),
                "last_open": candles[-1].timestamp.date().isoformat(),
                "journal_rows": len(rows),
            },
            ensure_ascii=False,
        )
    )
    if result.stop_reason == "ollama_unavailable":
        raise SystemExit(2)


def _run_lab(args) -> None:
    store = HistoricalStore(args.store)
    memory = ExperimentMemory(args.memory)
    client = ResearchOllama(model=args.model or None, timeout=args.timeout)
    try:
        result = run_qwen_lab(
            client,
            store,
            args.source,
            memory,
            max_hypotheses=args.max_hypotheses,
        )
    finally:
        memory.close()
    print(json.dumps(lab_report(result), ensure_ascii=False))
    if result.stop_reason == "ollama_unavailable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
