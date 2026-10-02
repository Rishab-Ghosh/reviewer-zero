"""Number grounding: every number in a claim must appear somewhere in the paper's full text.

Pure and network-free. Formats are normalised before comparing: thousands separators, signs,
trailing zeros, and percent vs fraction (a paper's 50% may be quoted as 0.5, and a fraction up to
1 as a percent). A claim number with decimals may be the
paper's number rounded to that precision; a whole number must match a whole number exactly (so a
derived "75%" is not grounded by a 74.8 somewhere in the paper).
"""

import re

from reviewer.claims.prompt_builder import row_line
from reviewer.models import Claim, ClaimGrounding, ParsedPaper

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\d])([kKMB](?![A-Za-z]))?(\s?%)?"
)
"""Numbers not glued to a preceding letter or digit (so 'Qwen3' and 'R1' are names, not
numbers); '1,000' and '12.5' are one number each; an attached k/K/M/B suffix is captured."""
SCALE = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9}
MAX_DECIMALS = 6
GROUNDED_FIELDS = ("statement", "setting", "result")


def _parse(match: re.Match[str]) -> tuple[float, int]:
    whole, frac = match.group(1), match.group(2)
    value = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
    return value, len(frac) if frac else 0


def numbers_in(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals) for every number in `text` (suffix not applied)."""
    return [(m.group(0), *_parse(m)) for m in NUMBER_RE.finditer(text)]


def _tuple_in_paper(written: str, text: str) -> bool:
    """'2,128' can be a tuple (2, 128) rather than a thousands number: accept it only when the
    paper writes the same comma-separated sequence (spaces allowed)."""
    core = written.rstrip("% kKMB")
    if "," not in core or "." in core:
        return False
    pattern = r"(?<![\d.])" + r"\s*,\s*".join(map(re.escape, core.split(","))) + r"(?![\d])"
    return re.search(pattern, text) is not None


def _forms(value: float, decimals: int) -> set[str]:
    """String forms a paper number can be quoted as: itself, zero-padded, and rounded to fewer
    (but at least one) decimals."""
    out = {f"{value:.{decimals}f}"}
    out.update(f"{round(value, k):.{k}f}" for k in range(1, MAX_DECIMALS + 1))
    return out


def paper_number_forms(text: str) -> set[str]:
    """Every acceptable form of every number in the paper, including x100 and /100 (percent vs
    fraction)."""
    forms: set[str] = set()
    for m in NUMBER_RE.finditer(text):
        value, d = _parse(m)
        variants = [(value, d)]
        if m.group(3):  # 16K -> 16000, 4.7M -> 4700000
            scaled = value * SCALE[m.group(3)]
            variants.append((scaled, max(d - len(str(int(SCALE[m.group(3)]))) + 1, 0)))
        for v, dv in variants:
            forms |= _forms(v, dv)
            if m.group(4):  # 50% may be quoted as 0.5
                forms |= _forms(v / 100, dv + 2)
            elif v <= 1:  # 0.5 may be quoted as 50%
                forms |= _forms(v * 100, max(dv - 2, 0))
    return forms


def is_grounded(value: float, decimals: int, forms: set[str]) -> bool:
    return f"{value:.{decimals}f}" in forms


def paper_text(paper: ParsedPaper) -> str:
    """Everything the paper says, for grounding: title, abstract, every section (fallback pages
    included), figure and table captions, table cells or raw text."""
    parts = [paper.title, paper.abstract, paper.contributions or ""]
    for s in paper.sections:
        parts.append(f"{s.number or ''} {s.heading}")
        parts.extend(s.paragraphs)
    for f in paper.figures:
        parts.append(f"{f.label} {f.caption}")
    for t in paper.tables:
        parts.append(f"{t.label} {t.caption}")
        parts.extend(row_line(r) for r in t.rows or [])
        parts.append(t.raw_text or "")
    return "\n".join(parts)


def ungrounded_numbers(claim: Claim, forms: set[str], text: str = "") -> list[str]:
    """Numbers in the claim's statement, setting, and result that do not appear in the paper,
    as written in the claim, deduplicated in order of appearance."""
    missing: list[str] = []
    for field in GROUNDED_FIELDS:
        for written, value, d in numbers_in(getattr(claim, field)):
            if is_grounded(value, d, forms):
                continue
            if _tuple_in_paper(written, text):
                continue
            if written not in missing:
                missing.append(written)
    return missing


def ground_claims(claims: list[Claim], paper: ParsedPaper) -> list[ClaimGrounding]:
    """One entry per claim with at least one ungrounded number."""
    text = paper_text(paper)
    forms = paper_number_forms(text)
    out = []
    for i, c in enumerate(claims):
        missing = ungrounded_numbers(c, forms, text)
        if missing:
            out.append(ClaimGrounding(claim_index=i, missing=missing))
    return out
