"""LLM judge for extracted claims: specificity and correctness, 1-5 each.

One call per paper (all its claims together) with the paper's parsed text and, where it exists,
its gold claims as reference. Critiques are never shown. Also builds the human calibration
sample and measures judge-human agreement. Everything here is pure except `judge_paper`.
"""

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from reviewer.claims.prompt_builder import (
    CONCLUSION_RE,
    INTRO_RE,
    RELATED_RE,
    cap_text,
    full_text,
    matching_subtrees,
    table_text,
)
from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.models import (
    Claim,
    GoldClaims,
    JudgeOutput,
    PaperClaims,
    PaperJudgment,
    ParsedPaper,
    TokenUsage,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RESULTS_RE = re.compile(
    r"result|experiment|evaluat|analys|ablation|finding|discussion|benchmark|empirical",
    re.IGNORECASE,
)
SECTION_CHAR_BUDGET = 60_000
TABLE_CHAR_BUDGET = 1_500
TABLES_TOTAL_BUDGET = 25_000
LOW_SCORE = 3
CALIBRATION_SEED = 20260925


class JudgeError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def load_judge_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=path.stem, text=path.read_text(encoding="utf-8").strip())


# --------------------------------------------------------------------------- context


FULL_TABLE_BUDGET = 1_000_000
"""Effectively uncut."""


def judge_context_full(
    paper: ParsedPaper, skip: re.Pattern[str] = RELATED_RE, figures: bool = False
) -> str:
    """judge_v2: every section except related work and references (with their subsections),
    uncut, then every table uncut. `skip` changes which headings are left out; `figures` adds
    every figure caption after the tables (the writing checker uses both)."""
    secs = paper.sections
    skipped = set(matching_subtrees(secs, skip))
    parts = [f"<title>{paper.title}</title>", f"<abstract>\n{paper.abstract}\n</abstract>"]
    for i, sec in enumerate(secs):
        st = full_text(sec)
        if i in skipped or not st.text:
            continue
        parts.append(f'<section heading="{st.heading}">\n{st.text}\n</section>')
    for t in paper.tables:
        tt = table_text(t, FULL_TABLE_BUDGET)
        parts.append(f"<table>\n{tt.header}\n{tt.body}\n</table>")
    if figures:
        parts += [f"<figure>\n{f.label}: {f.caption}\n</figure>" for f in paper.figures]
    return "\n\n".join(parts)


def judge_context(paper: ParsedPaper) -> str:
    """Title, abstract, intro, results-like sections, conclusion (within a character budget),
    then every table."""
    secs = paper.sections
    picked = sorted(
        set(matching_subtrees(secs, INTRO_RE))
        | set(matching_subtrees(secs, RESULTS_RE))
        | set(matching_subtrees(secs, CONCLUSION_RE))
    )
    parts = [f"<title>{paper.title}</title>", f"<abstract>\n{paper.abstract}\n</abstract>"]
    budget = SECTION_CHAR_BUDGET
    for i in picked:
        st = full_text(secs[i])
        if not st.text or budget <= 0:
            continue
        body = cap_text(st.text, budget)
        budget -= len(body)
        parts.append(f'<section heading="{st.heading}">\n{body}\n</section>')
    tbudget = TABLES_TOTAL_BUDGET
    for t in paper.tables:
        if tbudget <= 0:
            break
        tt = table_text(t, min(TABLE_CHAR_BUDGET, tbudget))
        tbudget -= len(tt.body)
        parts.append(f"<table>\n{tt.header}\n{tt.body}\n</table>")
    return "\n\n".join(parts)


def render_claims(claims: list[Claim]) -> str:
    blocks = []
    for i, c in enumerate(claims):
        blocks.append(
            f'<claim index="{i}">\nStatement: {c.statement}\nMethod: {c.method}\n'
            f"Setting: {c.setting}\nResult: {c.result}\n</claim>"
        )
    return "<claims>\n" + "\n\n".join(blocks) + "\n</claims>"


def render_gold(gold: GoldClaims | None) -> str:
    if gold is None or not gold.claims:
        return ""
    lines = [
        f"- {g.statement}" + (f" (result: {g.result})" if g.result else "") for g in gold.claims
    ]
    return "<reference_claims>\n" + "\n".join(lines) + "\n</reference_claims>"


CONTEXTS = {"judge_v1": judge_context, "judge_v2": judge_context_full}
"""Paper text given to each judge prompt version."""


def judge_user_message(
    paper: ParsedPaper,
    claims: PaperClaims,
    gold: GoldClaims | None,
    prompt_version: str = "judge_v1",
) -> str:
    parts = [f"<paper>\n{CONTEXTS[prompt_version](paper)}\n</paper>"]
    if ref := render_gold(gold):
        parts.append(ref)
    parts.append(render_claims(claims.claims))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- judging


def judgment_problems(out: JudgeOutput, n: int) -> list[str]:
    problems = []
    ids = [j.claim_index for j in out.judgments]
    if sorted(ids) != list(range(n)):
        problems.append(f"expected one judgment for each claim index 0..{n - 1}, got {ids}")
    for j in out.judgments:
        for dim in ("specificity", "correctness"):
            score = getattr(j, dim)
            reason = getattr(j, f"{dim}_reason").strip()
            if not 1 <= score <= 5:
                problems.append(f"claim {j.claim_index}: {dim} {score} is not 1-5")
            elif score <= LOW_SCORE and not reason:
                problems.append(f"claim {j.claim_index}: {dim} {score} needs a one-line reason")
    return problems


def judge_paper(
    client: LLMClient,
    paper: ParsedPaper,
    claims: PaperClaims,
    gold: GoldClaims | None,
    *,
    tier: str,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> PaperJudgment:
    user = judge_user_message(paper, claims, gold, prompt.version)
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if problem is None
            else (
                f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: your previous answer "
                f"was rejected: {problem}. Answer again.\n</previous_attempt_error>"
            )
        )
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=JudgeOutput,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problem = "; ".join(judgment_problems(out, len(claims.claims))) or None
        if problem is None:
            return PaperJudgment(
                paper_id=claims.paper_id,
                tier=tier,
                model=model,
                prompt_version=prompt.version,
                claims_prompt_version=claims.extraction_metadata.prompt_version,
                used_gold=gold is not None,
                judgments=sorted(out.judgments, key=lambda j: j.claim_index),
                token_usage=usage,
            )
        logger.warning("judge %s retry %d: %s", claims.paper_id, attempt + 1, problem)
    raise JudgeError(f"judge output still invalid after {max_retries} retries: {problem}", usage)


# --------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class ClaimRef:
    paper_id: str
    claim_index: int
    tier: str

    @property
    def id(self) -> str:
        return f"{self.paper_id}#{self.claim_index + 1}"


def calibration_sample(
    refs: list[ClaimRef], n: int = 10, seed: int = CALIBRATION_SEED
) -> list[ClaimRef]:
    """All own-paper claims plus a seeded random sample of lab and ordinary claims, to `n`
    total, in a seeded shuffled order."""
    own = sorted((r for r in refs if r.tier == "own"), key=lambda r: (r.paper_id, r.claim_index))
    pool = sorted(
        (r for r in refs if r.tier in ("lab", "ordinary")),
        key=lambda r: (r.paper_id, r.claim_index),
    )
    rng = random.Random(seed)
    picked = own + rng.sample(pool, max(0, min(n - len(own), len(pool))))
    rng.shuffle(picked)
    return picked


FRESH_SEED = 20260926


def fresh_sample(
    refs: list[ClaimRef], exclude: set[str], n: int = 5, seed: int = FRESH_SEED
) -> list[ClaimRef]:
    """A seeded random sample of `n` claims from every tier, skipping ids in `exclude` (the
    claims already graded), for a clean check after calibration."""
    pool = sorted(
        (r for r in refs if r.id not in exclude), key=lambda r: (r.paper_id, r.claim_index)
    )
    return random.Random(seed).sample(pool, min(n, len(pool)))


_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "our",
        "their",
        "than",
        "into",
        "over",
        "under",
        "using",
        "used",
        "which",
        "when",
        "while",
        "also",
        "more",
        "most",
        "such",
        "these",
        "those",
        "have",
        "has",
        "had",
        "not",
        "can",
        "its",
        "per",
        "across",
    ]
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def best_excerpt(paper: ParsedPaper, claim: Claim, limit: int = 1200) -> tuple[str, str]:
    """(where, text): the paragraph or table most relevant to the claim. Score = shared content
    words + 3 per shared number + 5 if the claim cites its section or table as evidence."""
    target_words = _words(f"{claim.statement} {claim.result}")
    target_nums = set(_NUM.findall(f"{claim.statement} {claim.result}"))
    evidence = " | ".join(claim.evidence).lower()
    options: list[tuple[str, str]] = []
    for s in paper.sections:
        for p in s.paragraphs:
            if p.strip():
                options.append((s.heading, p.strip()))
    for t in paper.tables:
        tt = table_text(t, 2000)
        options.append((t.label, f"{tt.header}\n{tt.body}"))
    if paper.abstract:
        options.append(("Abstract", paper.abstract))

    def score(opt: tuple[str, str]) -> float:
        where, text = opt
        s = len(target_words & _words(text)) + 3 * len(target_nums & set(_NUM.findall(text)))
        if where and where.lower() in evidence:
            s += 5
        return s

    if not options:
        return ("(no text)", "")
    where, text = max(options, key=score)
    return where, cap_text(" ".join(text.split()) if "\n" not in text else text, limit)


SCORE_LINE = re.compile(r"^- (specificity|correctness) \(1-5\):[ \t]*(\d)?", re.MULTILINE)
CLAIM_HEADER = re.compile(r"^## C\d+ · \S+ · (\S+)#(\d+)", re.MULTILINE)


def render_calibration(
    items: list[tuple[ClaimRef, Claim, tuple[str, str]]],
    title: str = "Claim grading calibration (10 claims)",
) -> str:
    lines = [
        f"# {title}",
        "",
        "Grade each claim on the same scales the judge uses; write a digit after each colon.",
        "",
        "- **Specificity:** 5 = names method, setting and a number; 4 = method and setting, no",
        "  number; 3 = method or setting but not both; 2 = mostly generic; 1 = could describe a",
        "  hundred papers.",
        "- **Correctness:** 5 = the paper says exactly this; 4 = small imprecision; 3 = partly",
        "  supported; 2 = mostly unsupported; 1 = the paper doesn't say this.",
        "",
        "The excerpt is the passage that best matches the claim (picked automatically); check the",
        "paper itself when it isn't enough. The judge's scores are not in this file.",
    ]
    for k, (ref, claim, (where, text)) in enumerate(items, start=1):
        quoted = "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
        lines += [
            "",
            f"## C{k} · {ref.tier} · {ref.id}",
            "",
            f"**Claim:** {claim.statement}",
            "",
            f"*Setting:* {claim.setting}  ",
            f"*Result:* {claim.result}",
            "",
            f"**Excerpt ({where}):**",
            "",
            quoted,
            "",
            "- specificity (1-5): ",
            "- correctness (1-5): ",
            "- notes: ",
        ]
    return "\n".join(lines) + "\n"


def parse_calibration(text: str) -> dict[str, dict[str, int | None]]:
    """{'paper#n': {'specificity': s, 'correctness': c}} from a filled calibration file;
    None for blank scores."""
    out: dict[str, dict[str, int | None]] = {}
    headers = list(CLAIM_HEADER.finditer(text))
    for h, nxt in zip(headers, [*headers[1:], None], strict=True):
        block = text[h.end() : nxt.start() if nxt else len(text)]
        scores: dict[str, int | None] = {"specificity": None, "correctness": None}
        for m in SCORE_LINE.finditer(block):
            scores[m.group(1)] = int(m.group(2)) if m.group(2) else None
        out[f"{h.group(1)}#{h.group(2)}"] = scores
    return out


def agreement(
    human: dict[str, dict[str, int | None]], judge: dict[str, dict[str, int]]
) -> dict[str, dict[str, object]]:
    """Per dimension: n graded, exact-match rate, within-1 rate, and claims differing by 2+."""
    out: dict[str, dict[str, object]] = {}
    for dim in ("specificity", "correctness"):
        pairs = [
            (cid, h[dim], judge[cid][dim])
            for cid, h in human.items()
            if h.get(dim) is not None and cid in judge
        ]
        n = len(pairs)
        out[dim] = {
            "n": n,
            "exact": sum(a == b for _, a, b in pairs) / n if n else 0.0,
            "within_1": sum(abs(a - b) <= 1 for _, a, b in pairs) / n if n else 0.0,
            "differ_2plus": [(cid, a, b) for cid, a, b in pairs if abs(a - b) >= 2],
        }
    return out
