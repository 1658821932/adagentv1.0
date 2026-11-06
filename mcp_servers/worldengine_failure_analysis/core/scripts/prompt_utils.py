"""Prompt files belong to this standalone native component."""
from pathlib import Path


def load_prompt(name):
    return (Path(__file__).resolve().parents[1] / "prompts" / name).read_text(encoding="utf-8").strip()
