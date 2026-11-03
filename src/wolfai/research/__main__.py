"""Один локальный исследовательский запуск. Holdout не читается."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wolfai.research.adaptive import run_adaptive
from wolfai.research.experiment import run_research
from wolfai.research.features import load_research_candles
from wolfai.research.journal import Journal
from wolfai.research.ollama import OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser(description="WolfAi: гипотезы Qwen и проверка Python")
    parser.add_argument("--csv", default="data/btcusdt_1d_2019-01_2026-09.csv")
    parser.add_argument("--journal", default="data/research/hypotheses.sqlite")
    parser.add_argument("--max-hypotheses", type=int, default=5)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
