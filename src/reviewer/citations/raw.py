"""Pure helpers over a reference's raw string (what the paper prints).

GROBID runs with citation consolidation, so a Reference's title/authors/year/venue/doi may have
been replaced by looked-up metadata. Every check reads the raw string instead; the parsed fields
are only lookup hints.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:\s/]*(?:abs/)?|abs/|arxiv\.org/(?:abs|pdf)/)\s*"
    r"(\d{4}\.\s?\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"<>]+[^\s,;\"<>.)])", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(?<!\d\.)(19[5-9]\d|20[0-4]\d)(?!\d|\.\d)")
ARXIV_MARKER_RE = re.compile(r"\barxiv\b|\bcorr\b|\bpreprint\b", re.IGNORECASE)
VENUE_MARKER_RE = re.compile(
    r"\bproceedings\b|\bproc\.|\bconference\b|\bworkshop\b|\bjournal\b|\btransactions\b"
    r"|\bsymposium\b|\bin advances in\b|\bneurips\b|\bnips\b|\biclr\b|\bicml\b|\bcvpr\b"
    r"|\biccv\b|\beccv\b|\bacl\b|\bemnlp\b|\bnaacl\b|\beacl\b|\bcoling\b|\baaai\b|\bijcai\b"
    r"|\bcolm\b|\btmlr\b|\bjmlr\b|\bkdd\b|\bsigir\b|\bwww\b|\bnature\b|\bscience\b|\bpmlr\b"
    r"|\bfindings of\b|\bcoRL\b|\buai\b|\baistats\b|\bicassp\b|\binterspeech\b|\bosdi\b"
    r"|\bmlsys\b|\bsosp\b|\bisca\b|\bmicro\b|\bhpca\b|\bsc\d{2}\b|\bpp\.\s*\d",
    re.IGNORECASE,
)
ET_AL_RE = re.compile(r"\bet\.?\s*al\b", re.IGNORECASE)
TRAILING_PREFIX_YEAR_RE = re.compile(r"[\s,.(]*(?:19|20)\d{2}[a-z]?[).,:]*\s*$")
NAME_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|&)\s*", re.IGNORECASE)
TITLE_MATCH_MIN = 0.9


def fold(text: str) -> str:
    """Lowercase, strip accents, map punctuation to spaces, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).split())


NUMBER_RANGE_RE = re.compile(r"\b\d+\s*[-–—]+\s*\d+\b")
TITLE_SPLIT_RE = re.compile(
    r"(?<=[a-z0-9)]{2}[.?!])\s+|(?<=[a-z0-9)][?!])\s+"
    r"|(?<=\.)\s+(?=(?:In|Proceedings|Advances|arXiv|Journal|Transactions)\b)"
)
"""Sentence breaks in a reference: after '.', '?' or '!' preceded by two lowercase letters or
digits, so author initials ('Geoffrey E. Hinton') do not split; and after any '.' that is
followed by a venue opener, so a title ending in an acronym ('... for NLP. In ICML') splits."""
QUOTED_RE = re.compile(r"[“\"]([^”\"]{8,}?)[,.]?[”\"]")


def strip_identifiers(raw: str) -> str:
    """Raw string without URLs, DOIs, arXiv ids and page ranges (digits that look like years)."""
    text = URL_RE.sub(" ", raw)
    text = DOI_RE.sub(" ", text)
    text = ARXIV_ID_RE.sub(" arXiv ", text)
    return NUMBER_RANGE_RE.sub(" ", text)


YEAR_CHUNK_RE = re.compile(r"^\(?(19|20)\d{2}[a-z]?\)?[.,]?$")
LNCS_RE = re.compile(r"^.{0,600}?(?:[A-Z]\.|et al\.)\s*:\s*(?P<rest>.+)$")
"""'Zhu, C., Han, S., Dally, W.J.: Trained ternary quantization (2016)'"""
TRAILING_YEAR_RE = re.compile(r"\s*\((?:19|20)\d{2}[a-z]?\)\s*\.?$")


def raw_title(raw: str) -> str | None:
    """Best guess at the printed title, used only as a search query (matches are still
    validated against the whole raw string). Styles handled: a quoted title (IEEE); 'Authors:
    Title (Year)' (LNCS); 'Authors. [Year.] Title. Venue' (ACL/NeurIPS/author-year)."""
    raw = re.sub(r"(\w)- (\w)", r"\1\2", raw)  # PDF line-break hyphenation
    m = QUOTED_RE.search(raw)
    if m and len(m.group(1).split()) >= 2:
        return m.group(1).strip(" ,.")
    lncs = LNCS_RE.match(raw)
    if lncs:
        rest = TRAILING_YEAR_RE.sub("", lncs.group("rest"))
        title = TITLE_SPLIT_RE.split(rest)[0].strip(" ,.")
        if len(title.split()) >= 2:
            return title
    chunks = [c.strip() for c in TITLE_SPLIT_RE.split(raw) if c.strip()]
    for chunk in chunks[1:]:
        if YEAR_CHUNK_RE.match(chunk):
            continue
        if len(chunk.split()) >= 2:
            return TRAILING_YEAR_RE.sub("", chunk).strip(" ,.")
        break
    return None


def looks_like_reference(raw: str) -> bool:
    """False for bibliography entries that are parse artifacts (table-of-contents lines,
    appendix text): no printed year, identifier or venue."""
    return bool(
        cited_years(raw)
        or arxiv_ids(raw)
        or dois(raw)
        or URL_RE.search(raw)
        or VENUE_MARKER_RE.search(strip_identifiers(raw))
        or ARXIV_MARKER_RE.search(raw)
    )


def cited_years(raw: str) -> list[int]:
    return sorted({int(y) for y in YEAR_RE.findall(strip_identifiers(raw))})


def arxiv_ids(raw: str) -> list[str]:
    """arXiv ids printed in the string ('2410. 04707', broken by a PDF line wrap, is rejoined)."""
    return list(dict.fromkeys(m.group(1).replace(" ", "") for m in ARXIV_ID_RE.finditer(raw)))


def dois(raw: str) -> list[str]:
    return list(
        dict.fromkeys(m.group(1).lower() for m in DOI_RE.finditer(URL_RE.sub(_url_doi, raw)))
    )


def _url_doi(m: re.Match[str]) -> str:
    url = m.group(0)
    return url.split("doi.org/", 1)[1] if "doi.org/" in url else " "


def cites_arxiv_only(raw: str) -> bool:
    """The reference points to arXiv/CoRR (or 'preprint') and names no venue."""
    text = strip_identifiers(raw)
    return bool(ARXIV_MARKER_RE.search(raw)) and not VENUE_MARKER_RE.search(text)


def arxiv_year(arxiv_id: str) -> int | None:
    """v1 year from a new-style id (YYMM.NNNNN) or an old-style id (cat/YYMMNNN)."""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id) or re.search(r"/(\d{2})(\d{2})\d{3}$", arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


@dataclass(frozen=True)
class TitleSpan:
    score: float
    """SequenceMatcher ratio of the folded title against the best window of the raw string."""
    start: int
    """Index (in folded raw words) where the matched title starts."""


def title_span(raw: str, title: str) -> TitleSpan:
    """Best fuzzy location of `title` inside `raw`, over word windows of length n-1..n+1."""
    raw_words = fold(raw).split()
    t = fold(title)
    n = len(t.split())
    if not raw_words or n == 0:
        return TitleSpan(0.0, 0)
    best = TitleSpan(0.0, 0)
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(raw_words) - size + 1)):
            window = " ".join(raw_words[i : i + size])
            score = SequenceMatcher(None, t, window, autojunk=False).ratio()
            if score > best.score:
                best = TitleSpan(score, i)
    return best


def title_similarity(raw: str, title: str) -> float:
    return title_span(raw, title).score


def author_segment(raw: str, title: str) -> str | None:
    """Folded raw text before the title, or None if the title cannot be located."""
    span = title_span(raw, title)
    if span.score < TITLE_MATCH_MIN or span.start == 0:
        return None
    return " ".join(fold(raw).split()[: span.start])


def author_chunks(raw: str, title: str) -> tuple[list[str], bool]:
    """Author names as printed (folded), and whether the list ends in 'et al.'.

    Splits the unfolded author prefix on commas / 'and' / '&', drops initials-only chunks and
    merges 'Surname, I.' pairs, which are ambiguous with 'Given Surname, Given Surname'."""
    segment = author_segment(raw, title)
    if segment is None:
        return [], False
    n_words = len(segment.split())
    # Map the folded prefix back onto the raw text: take the raw prefix with as many words.
    raw_prefix = _raw_prefix(raw, n_words)
    # Author-year styles end the author list with the year ("... Nick Ryder. 2020.").
    raw_prefix = TRAILING_PREFIX_YEAR_RE.sub("", raw_prefix)
    et_al = bool(ET_AL_RE.search(raw_prefix))
    raw_prefix = ET_AL_RE.split(raw_prefix)[0]
    chunks = [fold(c) for c in NAME_SPLIT_RE.split(raw_prefix)]
    names = [c for c in chunks if c and any(len(tok) > 1 for tok in c.split())]
    return names, et_al


def _raw_prefix(raw: str, n_folded_words: int) -> str:
    """Shortest raw prefix whose folded form has `n_folded_words` words."""
    tokens = re.split(r"(\s+)", raw)
    out: list[str] = []
    for tok in tokens:
        out.append(tok)
        if len(fold("".join(out)).split()) >= n_folded_words:
            break
    return "".join(out)


def name_tokens(name: str) -> set[str]:
    """Folded name tokens of length > 1 (initials dropped)."""
    return {tok for tok in fold(name).split() if len(tok) > 1}


def surname(name: str) -> str:
    """Last folded token of a display name ('Jianpeng Cheng' -> 'cheng')."""
    toks = fold(name).split()
    return toks[-1] if toks else ""


def is_web_resource(raw: str) -> bool:
    """A URL citation with no arXiv id, DOI or venue: a blog post, repo or product page."""
    return (
        bool(URL_RE.search(raw) or re.search(r"\bURL\b|https?\s*:", raw))
        and not arxiv_ids(raw)
        and not dois(raw)
        and not VENUE_MARKER_RE.search(strip_identifiers(raw))
        and not ARXIV_MARKER_RE.search(raw)
    )


ORDINALS = frozenset(
    [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
    ]
)


def _number_tokens(text: str) -> set[str]:
    return {t for t in fold(text).split() if t.isdigit() or t in ORDINALS}


def numbers_agree(record_title: str, raw: str) -> bool:
    """Every number or ordinal in the record's title also appears in the raw string."""
    return _number_tokens(record_title) <= set(fold(raw).split())


AUTHOR_LIST_RE = re.compile(r",|\band\b|&")


def strip_author_prefix(title: str) -> str:
    """GROBID's title minus a leading author-list sentence it sometimes swallows ('Ivo
    Danihelka, Oriol Vinyals, and Koray Kavukcuoglu. Video pixel networks' -> 'Video pixel
    networks'), so searches use the title, not names."""
    chunks = [c.strip() for c in TITLE_SPLIT_RE.split(title) if c.strip()]
    if len(chunks) < 2:
        return title
    head, rest = chunks[0], " ".join(chunks[1:])
    if AUTHOR_LIST_RE.search(head) and has_person_names(head) and len(rest.split()) >= 2:
        return rest.strip(" ,.")
    return title


VERSION_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)+(?![\d.]*\d)")
"""Dotted version numbers in a title: 'Qwen2.5' -> '2.5', 'GPT-3.5' -> '3.5', 'Llama 3.1'."""


def version_numbers(text: str) -> set[str]:
    """Dotted version numbers, ignoring URLs, DOIs, arXiv ids and page ranges."""
    return set(VERSION_RE.findall(strip_identifiers(text)))


def versions_agree(record_title: str, cited_title: str | None, raw: str) -> bool:
    """Version suffixes must match exactly: fuzzy title similarity ignores them ('Qwen2
    Technical Report' scores 0.96 against a raw 'Qwen2.5 technical report'). Every version in
    the record's title must be printed in the reference, and every version in the cited title
    (when one was parsed) must be in the record's title."""
    rec = version_numbers(record_title)
    if not rec <= version_numbers(raw):
        return False
    return cited_title is None or version_numbers(cited_title) <= rec


def has_person_names(raw: str) -> bool:
    """The raw string starts with something that looks like a personal-name list (not an
    organization or a bare title)."""
    head = raw.split(".")[0] if not QUOTED_RE.search(raw) else raw[: QUOTED_RE.search(raw).start()]  # type: ignore[union-attr]
    return bool(re.search(r"[A-Z][a-z]+,?\s+(?:[A-Z]\.|[A-Z][a-z]+)", head)) or bool(
        re.search(r"[A-Z][a-z]+,\s*[A-Z]\.", raw[:200])
    )


VENUE_ALIASES: dict[str, tuple[str, ...]] = {
    "neurips": ("neurips", "nips", "neural information processing systems"),
    "iclr": ("iclr", "learning representations"),
    "icml": ("icml", "international conference on machine learning", "pmlr"),
    "cvpr": ("cvpr", "computer vision and pattern recognition"),
    "iccv": ("iccv", "international conference on computer vision"),
    "eccv": ("eccv", "european conference on computer vision"),
    "acl": ("acl", "association for computational linguistics"),
    "emnlp": ("emnlp", "empirical methods in natural language processing"),
    "naacl": ("naacl", "north american chapter"),
    "aaai": ("aaai",),
    "ijcai": ("ijcai",),
    "jmlr": ("jmlr", "journal of machine learning research"),
    "tacl": ("tacl", "transactions of the association for computational linguistics"),
    "tmlr": ("tmlr", "transactions on machine learning research"),
    "tpami": ("tpami", "pattern analysis and machine intelligence"),
    "colm": ("colm", "conference on language modeling"),
    "aistats": ("aistats", "artificial intelligence and statistics"),
    "nature": ("nature",),
    "science": ("science",),
}


def venue_keys(text: str) -> set[str]:
    folded = f" {fold(text)} "
    return {k for k, names in VENUE_ALIASES.items() if any(f" {n} " in folded for n in names)}
