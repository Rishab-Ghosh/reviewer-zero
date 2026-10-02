"""Versioned meta-reviewer prompts (mirrors reviewer.claims.prompts)."""

from pathlib import Path

from reviewer.claims.prompts import Prompt

PROMPTS_DIR = Path(__file__).resolve().parent


def load_meta_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=path.stem, text=path.read_text(encoding="utf-8").strip())
