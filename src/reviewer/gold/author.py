"""Author gold: a labelled claims document ("Claim N" blocks with Claim/Method/Problem/Setting/
Result/What's new fields) copied verbatim into GoldClaims. No rewriting; 'As above' is resolved
only when it points at real text, and every resolution attempt is noted."""

import re
from pathlib import Path

from reviewer.gold.pdf_text import group_lines, join_lines, read_words
from reviewer.models import GoldClaim, GoldClaims, GoldSource

CLAIM_HEADER_RE = re.compile(r"^Claim\s+(\d+)$")
FIELD_LABELS: dict[str, str] = {
    "Claim": "statement",
    "Method": "method",
    "Problem": "problem",
    "Setting": "setting",
    "Result": "result",
    "What's new": "novelty_assertion",
    "What’s new": "novelty_assertion",
}
FIELD_RE = re.compile(r"^(" + "|".join(re.escape(k) for k in FIELD_LABELS) + r"):\s*(.*)$")
AS_ABOVE_RE = re.compile(r"^As above\b", re.IGNORECASE)


def parse_blocks(lines: list[str]) -> list[dict[str, str]]:
    """Split document lines into claim blocks of {field: verbatim text}."""
    blocks: list[dict[str, list[str]]] = []
    field: str | None = None
    for line in lines:
        text = line.strip()
        if CLAIM_HEADER_RE.match(text):
            blocks.append({})
            field = None
            continue
        if not blocks:
            continue  # title and anything before the first claim
        m = FIELD_RE.match(text)
        if m:
            field = FIELD_LABELS[m.group(1)]
            blocks[-1][field] = [m.group(2)]
        elif field is not None:
            blocks[-1][field].append(text)
    return [{k: join_lines(v) for k, v in b.items()} for b in blocks]


def resolve_as_above(blocks: list[dict[str, str]]) -> list[tuple[dict[str, str], list[str]]]:
    """For each field starting 'As above', look at the same field of the previous claim. If
    that text is itself an unresolved 'As above' (or there is no previous claim), the chain
    has no referent: leave the text verbatim and say so. Returns (block, notes) pairs."""
    out: list[tuple[dict[str, str], list[str]]] = []
    for i, block in enumerate(blocks):
        notes: list[str] = []
        for field, text in block.items():
            if not AS_ABOVE_RE.match(text):
                continue
            prev = blocks[i - 1].get(field) if i else None
            if prev is None:
                notes.append(
                    f"{field}: starts with 'As above' but nothing precedes Claim {i + 1} in the "
                    "source; left verbatim, UNRESOLVED."
                )
            elif AS_ABOVE_RE.match(prev):
                notes.append(
                    f"{field}: 'As above' points to Claim {i}'s {field}, which is itself an "
                    "unresolved 'As above'; left verbatim, UNRESOLVED."
                )
            else:
                block[field] = prev + text[len("As above") :]
                notes.append(f"{field}: 'As above' resolved to Claim {i}'s {field} text.")
        out.append((block, notes))
    return out


def apply_setting_overrides(
    resolved: list[tuple[dict[str, str], list[str]]],
    overrides: dict[int, str],
    note: str | None,
) -> list[tuple[dict[str, str], list[str]]]:
    """Replace the Setting of claim k (1-based) with author-supplied text, verbatim. Notes about
    the document's own Setting ('As above' handling) are superseded and dropped."""
    out = []
    for k, (block, notes) in enumerate(resolved, start=1):
        if k in overrides:
            original = block.get("setting")
            block = {**block, "setting": overrides[k]}
            notes = [n for n in notes if not n.startswith("setting:")]
            notes.append(
                f"setting: supplied by the author outside the document ({note or 'no note'}); "
                f"used verbatim in place of the document's Setting {original!r}."
            )
        out.append((block, notes))
    return out


def author_gold(
    pdf_path: Path,
    paper_id: str,
    author: str | None = None,
    setting_overrides: dict[int, str] | None = None,
    override_note: str | None = None,
) -> GoldClaims:
    lines = [ln.text for ln in group_lines(read_words(pdf_path))]
    claims = []
    resolved = resolve_as_above(parse_blocks(lines))
    if setting_overrides:
        resolved = apply_setting_overrides(resolved, setting_overrides, override_note)
    for block, notes in resolved:
        claims.append(
            GoldClaim(
                statement=block["statement"],
                method=block.get("method"),
                problem=block.get("problem"),
                setting=block.get("setting"),
                result=block.get("result"),
                novelty_assertion=block.get("novelty_assertion"),
                notes=" ".join(["Copied verbatim from the author's document.", *notes]),
            )
        )
    return GoldClaims(
        paper_id=paper_id,
        source=GoldSource.author,
        source_document=str(pdf_path),
        author=author,
        claims=claims,
    )
