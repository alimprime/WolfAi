"""Исследовательский движок: Qwen предлагает формулу, Python её проверяет."""

from wolfai.research.experiment import run_research
from wolfai.research.journal import Journal
from wolfai.research.lab import run_qwen_lab
from wolfai.research.compete import compete
from wolfai.research.optuna_engine import optimize_coefficients
from wolfai.research.pysr_engine import discover

__all__ = [
    "Journal",
    "compete",
    "discover",
    "optimize_coefficients",
    "run_qwen_lab",
    "run_research",
]
