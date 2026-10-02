"""Full-text GROBID fixtures of arXiv papers stay in the private repo (DECISIONS Q-M4); tests
that need them skip in the public export."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fulltext_tei(paper_id: str) -> str:
    path = FIXTURES / f"{paper_id}.tei.xml"
    if not path.exists():
        pytest.skip(f"full-text fixture {path.name} is not in the public repo")
    return path.read_text(encoding="utf-8")
