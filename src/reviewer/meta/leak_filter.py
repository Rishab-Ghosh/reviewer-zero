"""Score and verdict leak filter for every serialized meta-review text field (MASTER_PLAN D12).

`is_score_leak(sentence, paper_text, evidence_texts, scores)` returns the id of the first leak
pattern that matches, or None. Step 1 blanks statistics and technical vocabulary ("reject the
null", "acceptance probability", "reject option"); step 2 matches three classes, every pattern
case-insensitive and tied to the paper or the reviewer as its subject:
- L1: a score phrase in the reviewer's voice ("my overall score is 5", "I'd put this at 6");
- L2: a verdict phrase ("borderline paper", "should be accepted", "move this to an accept");
- L3: a bare scale fraction that is a score, not a count (numerator equals a stored score, or
  a rating word within 30 characters before it).
A match whose span occurs verbatim in the paper text or any evidence text is exempt.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from reviewer.models import MetaJudgment, MetaReviewOut, MetaReviewSink, MetaReviewText

_I = re.IGNORECASE

MASKS = [
    re.compile(
        r"\b(fail(s|ed)?\s+to\s+)?(reject|accept)(s|ed|ing)?\s+(the\s+)?"
        r"(null(\s+hypothes[ie]s)?|H_?0|H₀|hypothes[ie]s)\b",
        _I,
    ),
    re.compile(
        r"\b(acceptance|rejection)\s+(probabilit(y|ies)|rates?|ratios?|thresholds?|sampling|"
        r"steps?|rules?|options?|regions?|criteri(on|a))\b",
        _I,
    ),
    re.compile(r"\baccept(ance)?[-\s](reject(ion)?)\b", _I),
    re.compile(
        r"\breject(s|ed|ing)?\s+(option|inputs?|samples?|tokens?|draft|proposals?|queries)\b", _I
    ),
]

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "L1-score-phrase",
        re.compile(
            r"\b(overall|final|my|our|this\s+(paper|submission|work)'?s?)\s+(rating|score|grade)"
            r"\s*(of|is|would\s+be|:|=|around|about)?\s*\d",
            _I,
        ),
    ),
    (
        "L1-rating-change",
        # D12 extension (DECISIONS M8-1): a change to the reviewer's own rating, no number needed;
        # "my/our" only, so "improve the BLEU score" stays legitimate
        re.compile(
            r"\b(raise|raising|lower|lowering|increase|increasing|decrease|decreasing|revise|"
            r"revising|update|updating|bump|bumping|improve|improving)\s+(my|our)\s+"
            r"(overall\s+|final\s+)?(rating|score|grade|assessment|recommendation)\b",
            _I,
        ),
    ),
    (
        "L1-i-would-change-rating",
        # "the" rating is the reviewer's own only after "I would/I'd/I will" (DECISIONS M8-1)
        re.compile(
            r"\bI\s*(would|'d|’d|will)\s+(raise|lower|increase|decrease|revise|update|bump|"
            r"improve)\s+(the|my|our)\s+(overall\s+|final\s+)?"
            r"(rating|score|grade|assessment|recommendation)\b",
            _I,
        ),
    ),
    (
        "L1-i-would-rate",
        re.compile(r"\bI\s*(would|'d|’d|will)\s+(rate|score|give|put)\b[^.]{0,30}\d", _I),
    ),
    (
        "L1-out-of",
        re.compile(r"\b\d+(\.\d+)?\s+out\s+of\s+(10|ten|5|five|4|four)\b(?!\s+[a-z])", _I),
    ),
    (
        "L2-strength-verdict",
        re.compile(r"\b(clear|strong|weak|marginal(ly)?)\s+(accept|reject)\b", _I),
    ),
    ("L2-borderline", re.compile(r"\bborderline\s+(paper|submission|accept|reject)\b", _I)),
    (
        "L2-is-borderline",
        re.compile(r"\b(paper|submission|manuscript|work)\s+(is|seems|looks)\s+borderline\b", _I),
    ),
    (
        "L2-lean",
        re.compile(
            r"\blean(s|ing)?\s+(towards?\s+)?(accept(ance)?|reject(ion)?|positive|negative)\b", _I
        ),
    ),
    (
        "L2-should-be-accepted",
        re.compile(
            r"\b(paper|submission|manuscript|work|it)\s+(should|would|will|could)\s+(likely\s+)?"
            r"be\s+(accepted|rejected)\b",
            _I,
        ),
    ),
    ("L2-i-would-accept", re.compile(r"\bI\s+(would|will|should)\s+(accept|reject)\b", _I)),
    (
        "L2-recommend",
        re.compile(
            r"\brecommend(s|ed|ing|ation)?\s+(for\s+|of\s+|to\s+)?(an?\s+)?"
            r"(acceptance|rejection|accept|reject)\b",
            _I,
        ),
    ),
    (
        "L2-recommendation",
        re.compile(
            r"\b(my|the|overall|final)\s+(recommendation|verdict)\b"
            r"(?!\s+(system|model|task|engine|quality))",
            _I,
        ),
    ),
    (
        "L2-move-to",
        re.compile(
            r"\b(move|moves|flip|flips|change|changes|shift|shifts|turn|turns|push|raise|lower)\b"
            r"[^.]{0,40}\b(to|toward|towards|into|from)\s+(an?\s+)?"
            r"((clear|strong|weak)(er)?\s+)?"
            r"(accept(ance)?|reject(ion)?)\b",
            _I,
        ),
    ),
    (
        "L2-publishable",
        # D12 extension (DECISIONS M8-4): found in the first live run ("... would make this
        # publishable")
        re.compile(
            r"\b(un)?publishable\b|\b(not\s+(yet\s+)?)?ready\s+for\s+publication\b"
            r"|\b(acceptable|worthy|suitable)\s+(for|of)\s+(publication|acceptance)\b",
            _I,
        ),
    ),
    ("L2-likely", re.compile(r"\b(likely|unlikely)\s+to\s+be\s+(accepted|rejected)\b", _I)),
    (
        "L2-chance",
        re.compile(
            r"\b(chance|odds)\s+of\s+(being\s+)?(accepted|rejected|acceptance|rejection)\b", _I
        ),
    ),
    (
        "L2-bar",
        re.compile(r"\b(above|below|meets?|clears?|misses)\s+the\s+(acceptance\s+)?bar\b", _I),
    ),
]
FRACTION = re.compile(r"(?<![\d.]/)\b(\d+(\.\d+)?)\s*/\s*(10|5|4)\b(?![./]\d)(?!\s*[A-Za-z])")
"""D12's L3 pattern, plus: not part of a longer slash list ("Tables 2/3/5", "3.3/4.3/5.5")."""
RATING_WORD = re.compile(r"\b(rate|rating|score|grade|verdict)", _I)

_PROTECT = [
    (re.compile(r"(?<=\d)\.(?=\d)"), "\x00"),
    (re.compile(r"\b(e\.g|i\.e)\.", _I), lambda m: m.group(0).replace(".", "\x00")),
    (re.compile(r"\bet al\.", _I), lambda m: m.group(0).replace(".", "\x00")),
]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|(?<=:)\s+")
"""D12's split, plus a colon followed by whitespace, so a lead-in ("Three results would raise my
rating: (1) ...") is judged apart from the content it introduces (DECISIONS M8-1)."""


def split_sentences(text: str) -> list[str]:
    """Sentences, not split inside decimals, 'e.g.', 'i.e.' or 'et al.'. Line breaks are
    boundaries too, so a list ("would move me toward acceptance:\n- multi-seed results ...")
    loses only its leaking lead-in, never the items."""
    protected = text
    for rx, repl in _PROTECT:
        protected = rx.sub(repl, protected)
    parts = [
        p.strip()
        for line in protected.splitlines()
        for p in SENTENCE_SPLIT.split(line)
        if p.strip()
    ]
    return [p.replace("\x00", ".") for p in parts]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _blank(text: str) -> str:
    for rx in MASKS:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def is_score_leak(
    sentence: str,
    paper_text: str = "",
    evidence_texts: Iterable[str] = (),
    scores: Iterable[float] = (),
) -> str | None:
    """The id of the first leak pattern in `sentence`, or None (D12). Spans that occur verbatim
    in the paper text or any evidence text are exempt."""
    sources = [_norm(paper_text), *(_norm(e) for e in evidence_texts)]
    text = _blank(sentence)

    def exempt(span: str) -> bool:
        s = _norm(span)
        return bool(s) and any(s in src for src in sources if src)

    for pid, rx in PATTERNS:
        for m in rx.finditer(text):
            if not exempt(sentence[m.start() : m.end()]):
                return pid
    values = {float(v) for v in scores}
    for m in FRACTION.finditer(text):
        numerator = float(m.group(1))
        before = text[max(0, m.start() - 30) : m.start()]
        if (numerator in values or RATING_WORD.search(before)) and not exempt(m.group(0)):
            return "L3-fraction"
    return None


@dataclass
class Dropped:
    field: str
    sentence: str
    pattern: str


@dataclass
class FilterResult:
    out: MetaReviewOut
    dropped: list[Dropped] = field(default_factory=list)


SINK_TOP = 3
SECTION_NAMES = {
    "weaknesses": ("point", "Most likely to sink this paper"),
    "strengths": ("point", "Strengths"),
    "questions": ("question", "Questions"),
    "summary": ("sentence", "Summary"),
    "would_change_my_mind": ("sentence", "What would change my mind"),
}


def left_out_note(dropped: list[Dropped]) -> str | None:
    """One line naming each section that lost content (D12: a dropped weakness is never
    silently replaced by the next one)."""
    parts = []
    for name, (unit, title) in SECTION_NAMES.items():
        n = sum(1 for d in dropped if d.field == name)
        if n:
            parts.append(f"{n} {unit}{'s' if n > 1 else ''} in '{title}'")
    if not parts:
        return None
    total = len(dropped)
    return (
        "; ".join(parts)
        + f" {'were' if total > 1 else 'was'} left out because "
        + f"{'they' if total > 1 else 'it'} read as a score or a verdict."
    )


def filter_meta(
    j: MetaJudgment,
    paper_text: str,
    evidence_texts: list[str],
    scores: Iterable[float],
    checks: list[tuple[str, bool]] | None = None,
) -> FilterResult:
    """The allowlisted meta-review (MASTER_PLAN §4.4 `MetaReviewOut`): every string has passed
    `is_score_leak` sentence by sentence. List items are dropped whole; summary and
    would_change_my_mind lose only leaking sentences (an emptied field becomes None)."""
    scores = list(scores)
    dropped: list[Dropped] = []

    def leak(text: str) -> str | None:
        return is_score_leak(text, paper_text, evidence_texts, scores)

    def keep_sentences(name: str, text: str) -> str | None:
        kept = []
        for s in split_sentences(text):
            if (pid := leak(s)) is not None:
                dropped.append(Dropped(name, s, pid))
            else:
                kept.append(s)
        return "\n".join(kept) or None

    def keep_item(name: str, text: str) -> bool:
        for s in split_sentences(text):
            if (pid := leak(s)) is not None:
                dropped.append(Dropped(name, s, pid))
                return False
        return True

    ordered = sorted(j.weaknesses, key=lambda w: 0 if w.severity.value == "major" else 1)
    weak_kept = [w for w in ordered if keep_item("weaknesses", w.text)]
    strengths = [MetaReviewText(text=s.text) for s in j.strengths if keep_item("strengths", s.text)]
    questions = [q for q in j.questions if keep_item("questions", q)]
    summary = keep_sentences("summary", j.summary)
    wcmm = keep_sentences("would_change_my_mind", j.would_change_my_mind)
    note = left_out_note(dropped)
    out = MetaReviewOut(
        checks=[{"checker": c, "ok": ok} for c, ok in (checks or [])],
        summary=summary,
        sink=[MetaReviewSink(text=w.text) for w in weak_kept[:SINK_TOP]],
        strengths=strengths,
        questions=questions,
        would_change_my_mind=wcmm,
        left_out_note=note,
    )
    return FilterResult(out=out, dropped=dropped)
