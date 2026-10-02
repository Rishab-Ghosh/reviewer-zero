"""Parse-quality diagnostics over a ParsedPaper. Pure functions; heuristics, not verdicts.

- ``lost_tables``: tables the text refers to ("Table 3", "Tab. 3", "Tables 2 and 3") in the
  abstract, paragraphs or captions whose number is not among the parsed tables' labels
  (compared by number). Appendix labels ("Table A1") are only compared when some parsed table
  uses the same letter-prefixed scheme. Parsed tables with neither rows nor raw text are
  reported too.
- ``suspicious_authors``: author names that look like organizations or parsing artifacts, and
  (heuristic) a possible affiliation smear when 4+ authors all have the identical affiliation.
"""

import re
from collections.abc import Iterable, Sequence

from reviewer.models import ParsedPaper, Table

from .schemas import ParseQualityRow

# "Table 3", "Tab. 3", "Tables 2 and 3", "Tables 2-4", "Table A1"
_TABLE_REF_RE = re.compile(
    r"\b(?i:tables?|tabs?\.)\s*~?\s*"
    r"([A-Z]?\d+(?:\s*(?:,|and|&|–|-|to)\s*[A-Z]?\d+)*)"
)
_LABEL_NUM_RE = re.compile(r"\b([A-Z]?\d+)\b")
_XML_ID_RE = re.compile(r"^tab_?\d+$", re.IGNORECASE)

ORG_WORDS: tuple[str, ...] = (
    "Google",
    "Research",
    "Brain",
    "University",
    "Univ",
    "Institute",
    "Lab",
    "Labs",
    "Laboratory",
    "Inc",
    "Corp",
    "Corporation",
    "DeepMind",
    "OpenAI",
    "Microsoft",
    "Meta",
    "Department",
    "Dept",
    "School",
    "College",
    "Center",
    "Centre",
)
_ORG_RE = re.compile(rf"\b({'|'.join(ORG_WORDS)})\b\.?", re.IGNORECASE)
SMEAR_MIN_AUTHORS = 4


def _key_sort(key: str) -> tuple[str, int]:
    letter = key[0] if key[0].isalpha() else ""
    return letter, int(key[len(letter) :])


def referenced_table_keys(texts: Iterable[str]) -> set[str]:
    """Table numbers referenced in `texts`, as keys like '3' or 'A1'."""
    keys: set[str] = set()
    for text in texts:
        for m in _TABLE_REF_RE.finditer(text):
            keys.update(k.upper() for k in re.findall(r"[A-Za-z]?\d+", m.group(1)))
    return keys


def parsed_table_key(table: Table) -> str | None:
    """The number of a parsed table ('3', 'A1'), from its label or its caption; None if none."""
    label = table.label.strip()
    if label and not _XML_ID_RE.match(label):
        m = _LABEL_NUM_RE.search(label)
        if m:
            return m.group(1).upper()
    m = re.match(r"\s*(?i:table)\s+([A-Z]?\d+)", table.caption)
    return m.group(1).upper() if m else None


def _paper_texts(paper: ParsedPaper) -> list[str]:
    texts = [paper.abstract]
    texts += [p for s in paper.sections for p in s.paragraphs]
    texts += [f.caption for f in paper.figures]
    texts += [t.caption for t in paper.tables]
    return texts


def lost_tables(paper: ParsedPaper) -> list[str]:
    """Referenced tables that were not parsed, then parsed tables that came back empty."""
    parsed_keys = {k for t in paper.tables if (k := parsed_table_key(t)) is not None}
    lettered = any(k[0].isalpha() for k in parsed_keys)
    referenced = {
        k for k in referenced_table_keys(_paper_texts(paper)) if lettered or not k[0].isalpha()
    }
    lost = [f"Table {k}" for k in sorted(referenced - parsed_keys, key=_key_sort)]
    for t in paper.tables:
        if not t.rows and not (t.raw_text and t.raw_text.strip()):
            lost.append(f"{t.label or '(unlabeled)'} (parsed, no rows or text)")
    return lost


def suspicious_authors(paper: ParsedPaper) -> list[str]:
    """Author entries that look like organizations or artifacts, plus a smear heuristic."""
    out: list[str] = []
    for a in paper.authors:
        name = " ".join(a.name.split())
        reasons: list[str] = []
        m = _ORG_RE.search(name)
        if m:
            reasons.append(f"organization word '{m.group(1)}'")
        if len(name.split()) <= 1:
            reasons.append("single token")
        if any(c.isdigit() for c in name) or "@" in name:
            reasons.append("digit or @")
        if reasons:
            out.append(f"{name or '(empty)'} ({', '.join(reasons)})")
    affiliations = {(a.affiliation or "").strip() for a in paper.authors}
    if len(paper.authors) >= SMEAR_MIN_AUTHORS and len(affiliations) == 1:
        (aff,) = affiliations
        if aff:
            out.append(
                f"heuristic: all {len(paper.authors)} authors have the identical affiliation "
                f"'{aff}' (possible affiliation smear)"
            )
    return out


def parse_quality_row(paper: ParsedPaper, tier: str) -> ParseQualityRow:
    return ParseQualityRow(
        paper_id=paper.paper_id,
        tier=tier,
        n_tables=len(paper.tables),
        lost_tables=lost_tables(paper),
        suspicious_authors=suspicious_authors(paper),
    )


def render_parse_quality(rows: Sequence[ParseQualityRow]) -> str:
    """A plain fixed-width table for the terminal."""
    header = ("paper_id", "tier", "tables", "lost tables", "suspicious authors")
    body = [
        (
            r.paper_id,
            r.tier,
            str(r.n_tables),
            "; ".join(r.lost_tables) or "-",
            "; ".join(r.suspicious_authors) or "-",
        )
        for r in rows
    ]
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(4)]

    def fmt(row: tuple[str, ...]) -> str:
        cells = [row[i].ljust(widths[i]) for i in range(4)]
        return "  ".join([*cells, row[4]]).rstrip()

    lines = [fmt(header), "  ".join("-" * w for w in widths) + "  " + "-" * 18]
    lines += [fmt(row) for row in body]
    if not body:
        lines.append("(no parsed papers)")
    return "\n".join(lines)
