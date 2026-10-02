"""Lab-review gold and critiques, driven by a private spec of verbatim anchors.

The spec (data/private/gold_src/review_spec.yaml, gitignored) names, per paper, which spans of
the literature review go into which gold field, and where each limitations table or list is.
Every span is located by exact anchors in the extracted text; nothing is paraphrased.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from reviewer.gold.pdf_text import (
    Line,
    full_text,
    group_lines,
    read_words,
    span,
    table_lines,
    two_column_rows,
)
from reviewer.models import CritiqueEntry, Critiques, GoldClaim, GoldClaims, GoldSource

_SENTENCE_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")
_GOLD_FIELDS = ("statement", "method", "problem", "setting", "result", "novelty_assertion")


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Anchor(_Spec):
    start: str
    end: str


class ClaimSpec(_Spec):
    statement: Anchor
    method: Anchor | None = None
    problem: Anchor | None = None
    setting: Anchor | None = None
    result: Anchor | None = None
    novelty_assertion: Anchor | None = None
    notes: str | None = None


class GoldSpec(_Spec):
    paper_id: str
    claims: list[ClaimSpec]


class BlockSpec(_Spec):
    kind: Literal["table", "sentences", "comma_list", "bullets", "single"]
    label: str
    """How the block is labelled in the source, e.g. 'limitations table'."""
    span: Anchor | None = None
    """For list/prose kinds: the text of the list (without its 'Limitations:' label)."""
    pages: list[int] | None = None
    header: str | None = None
    end_marker: str | None = None
    split_x: float | None = None


class CritiqueSpec(_Spec):
    paper_id: str
    blocks: list[BlockSpec]


class AuthorGoldSpec(_Spec):
    gold_pdf: str
    """Path under data/private/ of the author's gold-claims document."""
    own_pdf: str
    """File name under data/papers/own/ of the paper it describes."""
    author: str | None = None
    setting_overrides: dict[int, str] = {}
    """1-based claim number -> Setting text supplied by the author outside the document,
    used verbatim in place of the document's Setting."""
    override_note: str | None = None
    """Provenance recorded on each overridden claim."""


class ReviewSpec(_Spec):
    review_pdf: str = "lit_review.pdf"
    """Path under data/private/ of the literature review."""
    author_gold: list[AuthorGoldSpec] = []
    gold: list[GoldSpec] = []
    skipped_gold: dict[str, str] = {}
    """paper_id -> why no gold (e.g. the review gives no numeric results)."""
    critiques: list[CritiqueSpec] = []
    no_critiques: dict[str, str] = {}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def lab_gold(text: str, spec: GoldSpec, source_document: str) -> GoldClaims:
    claims = []
    for c in spec.claims:
        fields = {
            f: span(text, a.start, a.end) for f in _GOLD_FIELDS if (a := getattr(c, f)) is not None
        }
        claims.append(GoldClaim(**fields, notes=c.notes))
    return GoldClaims(
        paper_id=spec.paper_id,
        source=GoldSource.lab_summary,
        source_document=source_document,
        claims=claims,
    )


def critique_entries(text: str, lines: list[Line], block: BlockSpec) -> list[CritiqueEntry]:
    if block.kind == "table":
        assert block.pages and block.header and block.split_x is not None
        rows = two_column_rows(
            table_lines(lines, block.pages, block.header, block.end_marker), block.split_x
        )
        return [
            CritiqueEntry(limitation=r.left, explanation=r.right or None, source_block=block.label)
            for r in rows
        ]
    assert block.span is not None
    body = span(text, block.span.start, block.span.end)
    if block.kind == "sentences":
        items = split_sentences(body)
    elif block.kind == "comma_list":
        items = [p.strip() for p in body.split(",") if p.strip()]
    elif block.kind == "bullets":
        items = [p.strip() for p in body.split("●") if p.strip()]
    else:
        items = [body]
    return [CritiqueEntry(limitation=i, source_block=block.label) for i in items]


def build_from_review(pdf_path: Path, spec: ReviewSpec) -> tuple[list[GoldClaims], list[Critiques]]:
    text = full_text(pdf_path)
    lines = group_lines(read_words(pdf_path))
    golds = [lab_gold(text, g, str(pdf_path)) for g in spec.gold]
    critiques = [
        Critiques(
            paper_id=c.paper_id,
            source_document=str(pdf_path),
            entries=[e for b in c.blocks for e in critique_entries(text, lines, b)],
        )
        for c in spec.critiques
    ]
    return golds, critiques
