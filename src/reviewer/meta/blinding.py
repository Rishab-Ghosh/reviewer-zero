"""Strip identifying and venue-status text from a paper before the meta-reviewer sees it
(PLAN_meta.md §2). Pure, network-free."""

import re

from reviewer.claims.prompt_builder import RELATED_RE
from reviewer.eval.judge import judge_context_full
from reviewer.models import ParsedPaper

ACK_RE = re.compile(r"\b(acknowledg(e)?ments?|funding)\b", re.IGNORECASE)
SKIP_RE = re.compile(f"{RELATED_RE.pattern}|{ACK_RE.pattern}", re.IGNORECASE)
"""Sections left out entirely, on top of judge_context_full's own related-work skip."""

VENUE_HEADER_RE = re.compile(
    r"^[ \t]*(published as a conference paper at|accepted (as|to|at)|to appear (in|at)|"
    r"camera[- ]ready|under review as a conference paper at)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
"""Running headers/footers PDF text extraction leaves on their own line, e.g. 'Published as a
conference paper at ICLR 2024' -- these state or imply the decision, so they must never reach
the model."""

STATUS_PHRASE_RE = re.compile(
    r"\b(under review as a conference paper|published as a conference paper|"
    r"accepted as a conference paper|camera[- ]ready( version)?)"
    r"( (at|in|for) (the )?[A-Z][\w.-]*( \d{4})?)?[.,:;]?",
    re.IGNORECASE,
)
"""The same headers anywhere in a line: GROBID usually merges the running header into the
title or into the middle of a paragraph ('<title>Under review as a conference paper at ICLR 2024
GRADIENT DESCENT ...'), where the line-anchored VENUE_HEADER_RE never sees it. The 2026-09-28
dev run leaked the decision this way (25/31 rejected+withdrawn full texts, 14 titles)."""


def strip_status(text: str) -> str:
    """Remove decision-revealing header phrases anywhere in `text` (titles, abstracts,
    evidence), collapsing the whitespace they leave."""
    return re.sub(r"[ \t]{2,}", " ", STATUS_PHRASE_RE.sub("", text)).strip()


def blind_paper(paper: ParsedPaper) -> ParsedPaper:
    """Title and abstract with status phrases removed, for the calls that see only those
    (baseline a, contamination probe)."""
    return paper.model_copy(
        update={"title": strip_status(paper.title), "abstract": strip_status(paper.abstract)}
    )


URL_RE = re.compile(
    r"\bhttps?://(www\.)?(github\.com|gitlab\.com|huggingface\.co|anonymous\.4open\.science|"
    r"osf\.io|zenodo\.org)\S*",
    re.IGNORECASE,
)

FUNDING_SENTENCE_RE = re.compile(
    r"[^.\n]*\b(this (work|research|project) was (partially |in part )?(funded|supported)|"
    r"grant (no\.|number|award)|we (thank|gratefully acknowledge)|"
    r"(funded|supported) (in part )?by)\b[^.\n]*\.",
    re.IGNORECASE,
)


def blind_text(paper: ParsedPaper) -> str:
    """The full paper (judge_context_full), with acknowledgements sections also skipped, then
    venue headers/footers, code/hosting URLs, and funding/thanks sentences removed. Title is
    kept (it is never anonymised on OpenReview either, and the checkers need it)."""
    text = judge_context_full(blind_paper(paper), skip=SKIP_RE)
    text = VENUE_HEADER_RE.sub("", text)
    text = STATUS_PHRASE_RE.sub("", text)
    text = URL_RE.sub("[url removed]", text)
    text = FUNDING_SENTENCE_RE.sub("", text)
    return text
