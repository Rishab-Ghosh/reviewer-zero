"""Stable paper ids from PDF paths. The own-paper rule lives in `reviewer.corpus.ids`."""

from pathlib import Path

from reviewer.corpus.ids import is_arxiv_id, own_paper_id, slugify, strip_version

__all__ = ["paper_id_for", "slugify"]


def paper_id_for(pdf: Path, sha256: str) -> str:
    """arXiv id (no version) for `.../arxiv/<id>.pdf`; otherwise `own-<slug>-<sha8>`."""
    if is_arxiv_id(pdf.stem) and any(part.lower() == "arxiv" for part in pdf.parent.parts):
        return strip_version(pdf.stem)
    return own_paper_id(pdf, sha256)
