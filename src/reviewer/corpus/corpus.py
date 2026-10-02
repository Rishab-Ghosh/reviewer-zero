"""Load the corpus: data/corpus.yaml (arXiv papers) plus every PDF in data/papers/own/."""

import logging
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from reviewer.cache import sha256_file
from reviewer.config import Settings
from reviewer.corpus.ids import is_arxiv_id, own_paper_id
from reviewer.models import CorpusEntry, PaperTier

logger = logging.getLogger(__name__)


class CorpusError(ValueError):
    """corpus.yaml is malformed, or paper ids collide."""


class _CorpusFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    papers: list[CorpusEntry]


def load_corpus_yaml(path: Path) -> list[CorpusEntry]:
    """Parse and validate corpus.yaml. Only famous/ordinary arXiv entries belong there."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = _CorpusFile.model_validate(raw).papers
    for e in entries:
        if e.tier == PaperTier.own:
            raise CorpusError(f"{e.paper_id}: own papers are discovered from data/papers/own/")
        if not is_arxiv_id(e.paper_id):
            raise CorpusError(f"{e.paper_id!r} is not an arXiv id")
    return entries


def discover_own(own_dir: Path) -> list[CorpusEntry]:
    """One `own` entry per PDF in `own_dir`, sorted case-insensitively by filename."""
    if not own_dir.is_dir():
        return []
    pdfs = sorted(
        (p for p in own_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )
    return [
        CorpusEntry(
            paper_id=own_paper_id(p, sha256_file(p)),
            tier=PaperTier.own,
            reason="user-supplied PDF from data/papers/own/",
            title=None,
            pdf_path=str(p),
        )
        for p in pdfs
    ]


def _check_unique(entries: list[CorpusEntry]) -> None:
    counts = Counter(e.paper_id for e in entries)
    dupes = sorted(pid for pid, n in counts.items() if n > 1)
    if dupes:
        raise CorpusError(f"duplicate paper ids in corpus: {dupes}")


def load_corpus(settings: Settings) -> list[CorpusEntry]:
    """All corpus entries: corpus.yaml first, then own PDFs. Raises CorpusError on duplicates."""
    entries = load_corpus_yaml(settings.corpus_file) + discover_own(settings.own_dir)
    _check_unique(entries)
    logger.debug("corpus: %d entries", len(entries))
    return entries
