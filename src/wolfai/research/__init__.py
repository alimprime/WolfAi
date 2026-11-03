"""Исследовательский движок: Qwen предлагает формулу, Python её проверяет."""

from wolfai.research.experiment import run_research
from wolfai.research.journal import Journal
from wolfai.research.lab import run_qwen_lab

__all__ = ["Journal", "run_qwen_lab", "run_research"]
