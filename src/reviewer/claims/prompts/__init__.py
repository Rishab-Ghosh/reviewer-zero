"""Versioned prompt files. The version is the file stem (e.g. 'extract_v1')."""

from pathlib import Path

from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parent


class Prompt(BaseModel):
    version: str
    """File stem, e.g. 'extract_v1'. Part of the LLM cache key."""
    text: str


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    """Load `<name>.md` from `prompts_dir` (`name` may include the .md suffix)."""
    stem = name.removesuffix(".md")
    path = prompts_dir / f"{stem}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=stem, text=path.read_text(encoding="utf-8").strip())
