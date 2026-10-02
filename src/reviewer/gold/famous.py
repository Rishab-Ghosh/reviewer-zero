"""Public gold for famous papers, copied verbatim from our ParsedPaper JSON (no LLM):
each item of the paper's own contributions statement, plus every abstract sentence that
reports a number (the paper's headline results)."""

import re

from reviewer.eval.checks import has_number
from reviewer.models import GoldClaim, GoldClaims, GoldSource, ParsedPaper

# Sentence boundary: end punctuation, space, then a capital or digit. Protects decimals
# ("28.4") and common abbreviations by requiring the space.
_SENTENCE_RE = re.compile(
    r"(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<=[.?!])\s+(?=[A-Z0-9(])"
)


def abstract_result_sentences(abstract: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(" ".join(abstract.split()))]
    return [s for s in sentences if s and has_number(s)]


def contribution_items(contributions: str) -> list[str]:
    """One item per line of the stored statement; a line that only announces the list
    (ends with ':') is not itself a claim."""
    items = [" ".join(line.split()) for line in contributions.splitlines()]
    return [i for i in items if i and not i.endswith(":")]


def famous_gold(paper: ParsedPaper) -> GoldClaims:
    claims: list[GoldClaim] = []
    if paper.contributions:
        claims += [
            GoldClaim(
                statement=item, notes="Verbatim item from the paper's contributions statement."
            )
            for item in contribution_items(paper.contributions)
        ]
    seen = {c.statement for c in claims}
    for sentence in abstract_result_sentences(paper.abstract):
        if sentence not in seen:
            claims.append(
                GoldClaim(
                    statement=sentence,
                    result=sentence,
                    notes="Verbatim abstract sentence that reports a number.",
                )
            )
    return GoldClaims(
        paper_id=paper.paper_id,
        source=GoldSource.paper_contributions,
        source_document=f"data/parsed/{paper.paper_id}.json",
        claims=claims,
    )
