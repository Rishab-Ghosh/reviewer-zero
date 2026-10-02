"""Quote verification: does a quote the model gave appear verbatim in the paper?

Pure and network-free. Both sides are squashed before comparing: NFKC (ligatures), lowercase,
soft hyphens dropped, a decimal point between digits kept as a marker, and every other
non-alphanumeric character removed. That forgives what the parse changes (line-break
hyphenation, whitespace, curly quotes, dashes, table cell separators) but no word may be added,
dropped, or changed. "..." (or "…") splits a quote into fragments that must appear in order,
each within `MAX_GAP` squashed characters of the previous one, so a quote cannot stitch
together separate passages.
"""

import re
import unicodedata

from reviewer.claims.grounding import paper_text
from reviewer.eval.judge import judge_context_full
from reviewer.models import ParsedPaper

MIN_QUOTE_WORDS = 5
"""Shorter quotes ('Table 2', 'we use Adam') are not evidence; they never verify."""
MIN_FRAGMENT_WORDS = 2
MAX_GAP = 600
ELLIPSIS_RE = re.compile(r"\.{3,}|…|\[\s*\.\.\.\s*\]")
DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
DROP_RE = re.compile(r"[^0-9a-z·]")
WORD_RE = re.compile(r"\w+")


def squash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("­", "")
    return DROP_RE.sub("", DECIMAL_RE.sub("·", text))


def fragments(quote: str) -> list[str]:
    return [f for f in (p.strip() for p in ELLIPSIS_RE.split(quote)) if f]


def _find_in_order(haystack: str, frags: list[str], start: int, first: bool) -> bool:
    if not frags:
        return True
    head, rest = frags[0], frags[1:]
    pos = haystack.find(head, start)
    while pos != -1 and (first or pos - start <= MAX_GAP):
        if _find_in_order(haystack, rest, pos + len(head), False):
            return True
        pos = haystack.find(head, pos + 1)
    return False


class QuoteIndex:
    """The paper's text, squashed once, for checking many quotes. Several texts are checked
    separately, so a quote cannot run from the end of one into the start of another."""

    def __init__(self, *texts: str) -> None:
        self.texts = [squash(t) for t in texts]

    @classmethod
    def for_paper(cls, paper: ParsedPaper) -> "QuoteIndex":
        """Everything the parse has (grounding's full text) and the exact context the checker
        was shown, so table rows match however they were rendered."""
        return cls(paper_text(paper), judge_context_full(paper))

    def verify(self, quote: str) -> bool:
        frags = fragments(quote)
        if len(WORD_RE.findall(" ".join(frags))) < MIN_QUOTE_WORDS:
            return False
        if any(len(WORD_RE.findall(f)) < MIN_FRAGMENT_WORDS for f in frags):
            return False
        squashed = [squash(f) for f in frags]
        if not all(squashed):
            return False
        return any(_find_in_order(t, squashed, 0, True) for t in self.texts)
