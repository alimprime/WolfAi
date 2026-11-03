"""Загрузка внешних рядов и отчёт ablation. Holdout не оценивается."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from wolfai.intelligence.acquire import acquire
from wolfai.intelligence.store import ObservationStore, coverage, individual_whale_events
from wolfai.research.features import load_research_candles


def main() -> None:
    parser = argparse.ArgumentParser(description="WolfAi phase 3: внешние ряды и ablation")
    parser.add_argument("command", choices=("acquire", "evaluate"))
    parser.add_argument("--directory", default="data/intelligence")
    parser.add_argument("--csv", default="data/btcusdt_1d_2019-01_2026-09.csv")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--report", default="data/intelligence/phase3_report.json")
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire(args.directory, end=date.fromisoformat(args.end))
        print(json.dumps({
            "stored_rows": manifest.get("stored_rows"),
            "counts": manifest.get("store_counts"),
            "pit_http_status": (manifest.get("coinmetrics") or {}).get("pit_http_status")
            if isinstance(manifest.get("coinmetrics"), dict) else None,
        }, ensure_ascii=False, indent=2))
        return
    _evaluate(args)


def _evaluate(args: argparse.Namespace) -> None:
    from wolfai.intelligence.ablation import run_ablation
    from wolfai.intelligence.qwen_run import run_phase3
    from wolfai.research.ollama import OllamaClient

    candles = load_research_candles(args.csv)
    store = ObservationStore(Path(args.directory) / "observations.sqlite")
    try:
        observations = store.load()
    finally:
        store.close()
    ablation = run_ablation(candles, observations)
    qwen: dict[str, object]
    if args.skip_qwen:
        qwen = {"status": "skipped"}
    else:
        qwen = run_phase3(OllamaClient(timeout=180), candles, observations)
    report = {
        "coverage": coverage(observations),
        "individual_whale_events": individual_whale_events(observations),
        "ablation": ablation,
        "qwen": qwen,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "whale_events": report["individual_whale_events"],
        "series": len(report["coverage"]),
        "reliable": ablation["statistically_reliable_improvement"],
        "qwen": qwen.get("status"),
        "report": str(destination),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
