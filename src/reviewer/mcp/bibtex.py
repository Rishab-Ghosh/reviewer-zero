"""BibTeX entries built locally from index metadata (no lookup, no network)."""

import re
import unicodedata

from reviewer.models import IndexPaperDetail

_SPECIAL = str.maketrans({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"})
_STOP = {"a", "an", "the", "on", "of", "for", "to", "in", "and", "with", "towards", "toward"}
_JOURNAL = re.compile(r"\b(journal|transactions|letters|review|magazine)\b", re.IGNORECASE)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _escape(text: str) -> str:
    return " ".join(text.translate(_SPECIAL).replace("{", "").replace("}", "").split())


def cite_key(p: IndexPaperDetail) -> str:
    """`surnameYEARword`, ASCII lowercase: vaswani2017attention; `anon` without authors."""
    surname = "anon"
    if p.authors:
        surname = re.sub(r"[^a-z]", "", _ascii(p.authors[0].split()[-1]).lower()) or "anon"
    words = [w for w in re.findall(r"[a-z0-9]+", _ascii(p.title or "").lower()) if w not in _STOP]
    return f"{surname}{p.year or ''}{words[0] if words else ''}"


def bibtex(p: IndexPaperDetail, key: str | None = None) -> str:
    key = key or cite_key(p)
    fields: list[tuple[str, str]] = [("title", "{" + _escape(p.title or "") + "}")]
    if p.authors:
        fields.append(("author", " and ".join(_escape(a) for a in p.authors)))
    else:
        fields.append(("author", ""))
    if p.year:
        fields.append(("year", str(p.year)))
    if p.venue and not (p.arxiv_id and p.venue.lower().startswith("arxiv")):
        kind = "article" if _JOURNAL.search(p.venue) else "inproceedings"
        fields.append(("journal" if kind == "article" else "booktitle", _escape(p.venue)))
    else:
        kind = "misc"
    if p.doi:
        fields.append(("doi", p.doi))
    if p.arxiv_id:
        fields += [("eprint", p.arxiv_id), ("archivePrefix", "arXiv")]
        fields.append(("url", f"https://arxiv.org/abs/{p.arxiv_id}"))
    elif p.doi:
        fields.append(("url", f"https://doi.org/{p.doi}"))
    head = "" if p.authors else "% authors not in the Reviewer Zero index; fill in before citing\n"
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"{head}@{kind}{{{key},\n{body}\n}}"


def _suffix(i: int) -> str:
    """0 -> a, 25 -> z, 26 -> aa, 27 -> ab, ..."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(ord("a") + r) + out
    return out


def unique_keys(papers: list[IndexPaperDetail]) -> list[str]:
    """Cite keys for a list, with a, b, ..., z, aa, ab, ... suffixes on collisions (in list
    order)."""
    base = [cite_key(p) for p in papers]
    seen: dict[str, int] = {}
    total = {k: base.count(k) for k in base}
    out = []
    for k in base:
        if total[k] == 1:
            out.append(k)
            continue
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append(f"{k}{_suffix(i)}")
    return out
