"""Pure helpers for paper identifiers."""

import re
from pathlib import Path

# New-style arXiv ids (YYMM.NNNN or YYMM.NNNNN, optional vN) and old-style (archive/YYMMNNN).
_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_STYLE = re.compile(r"^[a-z][a-z.-]*(\.[A-Z]{2})?/\d{7}(v\d+)?$")
_VERSION = re.compile(r"v\d+$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

SLUG_MAX_LEN = 40


def is_arxiv_id(s: str) -> bool:
    """True for a new- or old-style arXiv identifier, with or without a version suffix."""
    return bool(_NEW_STYLE.match(s) or _OLD_STYLE.match(s))


def strip_version(s: str) -> str:
    """'2106.09685v2' -> '2106.09685'. Non-versioned ids are returned unchanged."""
    return _VERSION.sub("", s) if is_arxiv_id(s) else s


def slugify(stem: str, max_len: int = SLUG_MAX_LEN) -> str:
    slug = _NON_ALNUM.sub("-", stem.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "paper"


def own_paper_id(path: Path, sha256: str) -> str:
    """`own-<slug>-<sha8>`: slug from the file stem, sha8 = first 8 hex chars of the PDF sha256."""
    return f"own-{slugify(path.stem)}-{sha256[:8].lower()}"
