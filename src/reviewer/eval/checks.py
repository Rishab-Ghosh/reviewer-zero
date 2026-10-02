"""Automated claim checks. Pure functions; every result is a hint for the human grader, not a gate.

Rules:
- ``vague_phrase``: the statement contains a phrase from ``VAGUE_PHRASES`` (case-insensitive,
  word-boundary aware). "state of the art" / "SOTA" is flagged only when the statement has no
  number (see ``has_number``).
- ``missing_numbers``: neither the statement nor the result has a number, but the abstract does.
- ``too_long``: the statement has more than ``MAX_STATEMENT_WORDS`` words.
- ``near_duplicate``: two claims of the same paper have TF-IDF cosine similarity at or above
  the duplicate threshold (``settings.duplicate_threshold``, default
  ``DEFAULT_DUPLICATE_THRESHOLD``). No embedding model is available, so this is lexical
  similarity. ``max_similarities`` gives every claim's highest cosine to another claim.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from reviewer.models import Flag, FlagKind, PaperClaims, ParsedPaper

MAX_STATEMENT_WORDS = 60
DEFAULT_DUPLICATE_THRESHOLD = 0.6


def similarity_method(threshold: float = DEFAULT_DUPLICATE_THRESHOLD) -> str:
    """One-paragraph description of the similarity check, for report headers."""
    return (
        "TF-IDF cosine in pure Python over claim statements (lowercase alphanumeric tokens, "
        "small stopword list, smoothed IDF over every extracted claim in the report); flag when "
        f"cosine ≥ {threshold:.2f}. Jaccard token overlap is reported alongside. "
        "No embedding model is used, so paraphrases with different wording are not caught."
    )


# Case-insensitive regex patterns, matched with word boundaries on both sides.
VAGUE_PHRASES: tuple[str, ...] = (
    r"improves?\s+(?:the\s+)?performance",
    r"improved\s+performance",
    r"better\s+performance",
    r"novel\s+framework",
    r"novel\s+approach",
    r"novel\s+method",
    r"various\s+tasks",
    r"a\s+variety\s+of\s+tasks",
    r"wide\s+range\s+of\s+tasks",
    r"extensive\s+experiments",
    r"comprehensive\s+experiments",
    r"significantly",
    r"substantially",
    r"remarkable",
    r"promising\s+results",
    r"competitive\s+performance",
    r"strong\s+performance",
    r"effectiveness",
    r"effective",
    r"robust",
    r"outperforms\s+existing\s+methods",
    r"outperforms\s+(?:the\s+|strong\s+|all\s+)?baselines",
    r"superior\s+performance",
    r"good\s+results",
    r"plays\s+an\s+important\s+role",
    r"sheds\s+light\s+on",
    r"paves\s+the\s+way",
    r"holistic",
    r"seamlessly",
)
# Flagged only when the statement contains no number.
SOTA_PATTERN = r"state[\s-]+of[\s-]+the[\s-]+art|SOTA"

_VAGUE_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"(?<!\w){p}(?!\w)", re.IGNORECASE) for p in VAGUE_PHRASES
)
_SOTA_RE = re.compile(rf"(?<!\w)(?:{SOTA_PATTERN})(?!\w)", re.IGNORECASE)

# --------------------------------------------------------------------------- numbers
#
# Number heuristic. Text is split on whitespace; surrounding punctuation is stripped. A token
# counts as a number when, after stripping a leading sign/currency/approx symbol (~ ≈ ± + $ < >):
#   - it starts with a letter            -> never (model names: GPT-4, T5, Llama-3.1-8B, F1)
#   - it is a percentage (5%, 12.3%)     -> yes
#   - it has a decimal point (0.5, 28.4) -> yes (also 1e-4 and 1,000-style thousands)
#   - it is a multiplier (3x, 2.5×)      -> yes
#   - it has a scale suffix (175B, 10K)  -> yes
#   - it is followed by a unit/metric word ("1 GPU", "2048 tokens", "10-shot") -> yes
#   - it is a plain integer >= 2         -> yes, unless it is a year (1900-2099) or follows a
#     reference word (Table 2, Figure 3, Section 4, Eq. 5)
#   - otherwise (3D, 2nd, bare 1)        -> no


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


UNIT_WORDS: frozenset[str] = _words(
    """
    token tokens parameter parameters param params layer layers head heads example examples
    sample samples image images video videos hour hours day days minute minutes second seconds
    ms gpu gpus tpu tpus epoch epochs step steps point points pp percent percentage bleu rouge
    f1 em accuracy acc auc map miou psnr fid perplexity ppl language languages task tasks dataset
    datasets benchmark benchmarks model models time times fold shot shots bit bits billion
    million thousand trillion gb mb tb kb flops dimension dimensions dim dims word words sentence
    sentences document documents query queries user users participant participants annotator
    annotators question questions problem problems game games environment environments episode
    episodes trial trials run runs seed seeds class classes category categories domain domains
    hop hops page pages line lines node nodes edge edges expert experts agent agents speedup
    fewer more less higher lower lang pairs pair subjects prompts
    """
)
_REFERENCE_WORDS: frozenset[str] = _words(
    "table tables figure figures fig fig. section sections sec sec. eq eq. equation appendix "
    "chapter theorem lemma algorithm line"
)
_STRIP_CHARS = "()[]{}\"'`,;:!?.…“”‘’*"
_LEADING = "~≈±+$<>≤≥−-"
_PERCENT_RE = re.compile(r"^\d+(?:\.\d+)?%$")
_DECIMAL_RE = re.compile(r"^\d*\.\d+|^\d+(?:,\d{3})+$|^\d+(?:\.\d+)?e-?\d+$")
_MULT_RE = re.compile(r"^\d+(?:\.\d+)?[x×]$", re.IGNORECASE)
_SCALE_RE = re.compile(r"^\d+(?:\.\d+)?[KMBT]$")
_INT_RE = re.compile(r"^\d+$")
_POWER_RE = re.compile(r"^\d+\^[-−]?\d+$")
_HYPHEN_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)-([a-z]+)$", re.IGNORECASE)
_SCALED_HYPHEN_RE = re.compile(r"^\d+(?:\.\d+)?[KMBT]-[A-Za-z]+$")
_RANGE_RE = re.compile(r"^([\d.]+%?)[-–—]([\d.]+%?)$")


def _clean(tok: str) -> str:
    return tok.strip(_STRIP_CHARS)


def _is_year(n: int) -> bool:
    return 1900 <= n <= 2099


def _core_is_number(core: str, next_word: str, prev_word: str) -> bool:
    if not core or not core[0].isdigit():
        return False
    if _PERCENT_RE.match(core) or _DECIMAL_RE.match(core) or _MULT_RE.match(core):
        return True
    if _SCALE_RE.match(core) or _POWER_RE.match(core):
        return True
    m = _HYPHEN_UNIT_RE.match(core)
    if m:
        return m.group(2).lower() in UNIT_WORDS
    if _SCALED_HYPHEN_RE.match(core):
        return True  # "540B-parameter", "1.3B-token"
    m = _RANGE_RE.match(core)
    if m:
        return any(_core_is_number(part, next_word, "") for part in m.groups())
    if _INT_RE.match(core):
        value = int(core)
        if next_word.lower() in UNIT_WORDS:
            return True
        if prev_word.lower() in _REFERENCE_WORDS:
            return False
        return value >= 2 and not _is_year(value)
    return False


def find_numbers(text: str) -> list[str]:
    """Return the tokens of `text` that count as numbers under the heuristic above."""
    raw = text.split()
    words = [_clean(t) for t in raw]
    found: list[str] = []
    for i, word in enumerate(words):
        if not any(c.isdigit() for c in word):
            continue
        if word[0].isalpha():
            continue  # GPT-4, T5, Llama-3, F1
        core = word.lstrip(_LEADING)
        next_word = words[i + 1] if i + 1 < len(words) else ""
        prev_word = raw[i - 1].strip("()[]{}\"',;:") if i > 0 else ""
        if _core_is_number(core, next_word, prev_word):
            found.append(word)
    return found


def has_number(text: str) -> bool:
    return bool(find_numbers(text))


# --------------------------------------------------------------------------- similarity

STOPWORDS: frozenset[str] = _words(
    """
    a an the and or of to in on for with by from at as is are was were be been being this that
    these those it its we our they their which who whom than then into over under via using
    use used can could may might will would also both such not no but if while when where how
    """
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def build_idf(docs: Iterable[str]) -> dict[str, float]:
    """Smoothed IDF: log((1 + N) / (1 + df)) + 1, so shared terms keep a positive weight."""
    doc_tokens = [set(tokenize(d)) for d in docs]
    n = len(doc_tokens)
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        df.update(toks)
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def _vector(text: str, idf: dict[str, float], default_idf: float) -> dict[str, float]:
    tf = Counter(tokenize(text))
    return {t: c * idf.get(t, default_idf) for t, c in tf.items()}


def tfidf_cosine(a: str, b: str, idf: dict[str, float]) -> float:
    default_idf = max(idf.values(), default=1.0)
    va, vb = _vector(a, idf, default_idf), _vector(b, idf, default_idf)
    dot = sum(w * vb.get(t, 0.0) for t, w in va.items())
    na = math.sqrt(sum(w * w for w in va.values()))
    nb = math.sqrt(sum(w * w for w in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- checks


def check_vague(index: int, statement: str) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[str] = set()
    for pattern in _VAGUE_RES:
        for m in pattern.finditer(statement):
            key = m.group(0).lower()
            if key not in seen:
                seen.add(key)
                flags.append(
                    Flag(claim_index=index, kind=FlagKind.vague_phrase, detail=f"matched '{key}'")
                )
    if not has_number(statement):
        m = _SOTA_RE.search(statement)
        if m:
            flags.append(
                Flag(
                    claim_index=index,
                    kind=FlagKind.vague_phrase,
                    detail=f"matched '{m.group(0).lower()}' with no number in the statement",
                )
            )
    return flags


def check_missing_numbers(
    index: int, statement: str, result: str, abstract: str | None
) -> list[Flag]:
    if abstract is None or has_number(statement) or has_number(result):
        return []
    abstract_numbers = find_numbers(abstract)
    if not abstract_numbers:
        return []
    examples = ", ".join(f"'{n}'" for n in abstract_numbers[:3])
    return [
        Flag(
            claim_index=index,
            kind=FlagKind.missing_numbers,
            detail=f"no number in statement or result, but the abstract has some ({examples})",
        )
    ]


def check_too_long(index: int, statement: str) -> list[Flag]:
    n_words = len(statement.split())
    if n_words <= MAX_STATEMENT_WORDS:
        return []
    return [
        Flag(
            claim_index=index,
            kind=FlagKind.too_long,
            detail=f"{n_words} words (limit {MAX_STATEMENT_WORDS})",
        )
    ]


def max_similarities(
    statements: Sequence[str], idf: dict[str, float]
) -> list[tuple[int | None, float]]:
    """For each statement: (index of the most similar other statement, its TF-IDF cosine).

    (None, 0.0) when the paper has a single claim. Ties go to the lowest index.
    """
    out: list[tuple[int | None, float]] = []
    for i, a in enumerate(statements):
        best: tuple[int | None, float] = (None, 0.0)
        for j, b in enumerate(statements):
            if i == j:
                continue
            cos = tfidf_cosine(a, b, idf)
            if best[0] is None or cos > best[1]:
                best = (j, cos)
        out.append(best)
    return out


def check_near_duplicates(
    statements: Sequence[str],
    idf: dict[str, float],
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> list[Flag]:
    flags: list[Flag] = []
    for i in range(len(statements)):
        for j in range(i + 1, len(statements)):
            cos = tfidf_cosine(statements[i], statements[j], idf)
            if cos < threshold:
                continue
            jac = jaccard(statements[i], statements[j])
            for me, other in ((i, j), (j, i)):
                flags.append(
                    Flag(
                        claim_index=me,
                        kind=FlagKind.near_duplicate,
                        detail=(
                            f"similar to claim {other + 1}: TF-IDF cosine {cos:.2f}, "
                            f"Jaccard {jac:.2f}"
                        ),
                    )
                )
    return flags


def run_checks(
    claims: PaperClaims,
    paper: ParsedPaper | None,
    idf: dict[str, float] | None = None,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> list[Flag]:
    """All flags for one paper, sorted by claim index. `idf` defaults to the paper's own claims."""
    statements = [c.statement for c in claims.claims]
    if idf is None:
        idf = build_idf(statements)
    abstract = paper.abstract if paper is not None else None
    flags: list[Flag] = []
    for i, claim in enumerate(claims.claims):
        flags += check_vague(i, claim.statement)
        flags += check_missing_numbers(i, claim.statement, claim.result, abstract)
        flags += check_too_long(i, claim.statement)
    flags += check_near_duplicates(statements, idf, threshold)
    return sorted(flags, key=lambda f: f.claim_index)
