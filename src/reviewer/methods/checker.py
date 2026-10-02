"""Methodology checker: one structured call per paper over the full text (judge_v2 context),
then code verifies every quote, sends findings with no verified quote back once for corrected
quotes, and downgrades whatever is still unverified to unclear.

Everything except `check_paper` is pure. Critiques are never shown to the checker.
"""

import logging
from pathlib import Path

from reviewer.claims.prompts import Prompt
from reviewer.eval.judge import judge_context_full
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.methods.quotes import QuoteIndex
from reviewer.models import (
    CheckedConcern,
    CheckedFinding,
    ChecklistFinding,
    ChecklistItemId,
    MethodsOutput,
    MethodsVerdict,
    OtherConcern,
    PaperMethodsReview,
    ParsedPaper,
    QuoteCheck,
    QuoteRepairOutput,
    TokenUsage,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_I = ChecklistItemId
PROMPT_ITEMS: dict[str, list[ChecklistItemId]] = {
    "methods_v1": [
        _I.baseline_tuning,
        _I.seeds_variance,
        _I.significance,
        _I.leakage,
        _I.compute_matched,
        _I.ablation_isolation,
        _I.data_disclosure,
        _I.hparam_disclosure,
        _I.abstract_support,
    ],
    "methods_v2": [
        _I.baseline_tuning,
        _I.seeds_variance,
        _I.significance,
        _I.leakage,
        _I.compute_matched,
        _I.ablation_isolation,
        _I.data_disclosure,
        _I.hparam_disclosure,
        _I.claims_support,
        _I.evaluation_scope,
    ],
}
PROMPT_ITEMS["methods_v2s"] = PROMPT_ITEMS["methods_v2"]
"""The checklist each prompt version asks, in its order."""
ITEMS = PROMPT_ITEMS["methods_v1"]


def items_for(prompt_version: str) -> list[ChecklistItemId]:
    """Items for a prompt version (a '+repair' suffix is ignored)."""
    return PROMPT_ITEMS[prompt_version.split("+")[0]]


MAX_OTHER_CONCERNS = 5
MAX_QUOTES = 3
RAISED = (MethodsVerdict.no, MethodsVerdict.unclear)


class MethodsError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def load_methods_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=path.stem, text=path.read_text(encoding="utf-8").strip())


def paper_block(paper: ParsedPaper) -> str:
    return f"<paper>\n{judge_context_full(paper)}\n</paper>"


def output_problems(out: MethodsOutput, items: list[ChecklistItemId] = ITEMS) -> list[str]:
    """Every checklist item answered exactly once, and no item from another version."""
    got = [f.item for f in out.findings]
    missing = [i.value for i in items if i not in got]
    dupes = sorted({i.value for i in got if got.count(i) > 1})
    problems = []
    if extra := sorted({i.value for i in got if i not in items}):
        problems.append(f"items not on this checklist: {', '.join(extra)}")
    if missing:
        problems.append(f"missing checklist items: {', '.join(missing)}")
    if dupes:
        problems.append(f"items answered more than once: {', '.join(dupes)}")
    return problems


def other_key(i: int) -> str:
    return f"other:{i}"


def check_quotes(index: QuoteIndex, quotes: list[str]) -> list[QuoteCheck]:
    return [QuoteCheck(quote=q, verified=index.verify(q)) for q in quotes[:MAX_QUOTES]]


def any_verified(checks: list[QuoteCheck]) -> bool:
    return any(c.verified for c in checks)


def repair_message(
    paper: ParsedPaper,
    failed: list[tuple[str, ChecklistFinding | OtherConcern]],
) -> str:
    blocks = []
    for key, f in failed:
        quotes = "\n".join(f"- {q}" for q in f.quotes) or "- (none given)"
        if isinstance(f, ChecklistFinding):
            body = (
                f"Verdict: {f.verdict.value}\nRationale: {f.rationale}\n"
                f"Concern: {f.concern or '(none)'}"
            )
        else:
            body = f"Concern: {f.concern}"
        blocks.append(
            f'<finding key="{key}">\n{body}\nQuotes not found in the paper:\n{quotes}\n</finding>'
        )
    return (
        paper_block(paper)
        + "\n\n<findings_to_repair>\n"
        + "\n\n".join(blocks)
        + "\n</findings_to_repair>"
    )


def assemble(
    out: MethodsOutput,
    index: QuoteIndex,
    repaired: dict[str, list[str]],
    items: list[ChecklistItemId] = ITEMS,
) -> tuple[list[CheckedFinding], list[CheckedConcern]]:
    """Verify quotes (original plus any repaired ones) and downgrade findings with none
    verified. Pure."""
    by_item = {f.item: f for f in out.findings}
    findings = []
    for item in items:
        f = by_item[item]
        checks = check_quotes(index, f.quotes)
        extra = repaired.get(item.value)
        if extra is not None:
            checks += check_quotes(index, extra)
        ok = any_verified(checks)
        downgrade = not ok and f.verdict != MethodsVerdict.unclear
        findings.append(
            CheckedFinding(
                item=item,
                verdict=MethodsVerdict.unclear if downgrade else f.verdict,
                model_verdict=f.verdict,
                rationale=f.rationale,
                concern=f.concern,
                quotes=checks,
                downgraded=downgrade,
                repaired=extra is not None,
            )
        )
    concerns = []
    for i, c in enumerate(out.other_concerns[:MAX_OTHER_CONCERNS]):
        checks = check_quotes(index, c.quotes)
        extra = repaired.get(other_key(i))
        if extra is not None:
            checks += check_quotes(index, extra)
        concerns.append(
            CheckedConcern(
                concern=c.concern,
                quotes=checks,
                verified=any_verified(checks),
                repaired=extra is not None,
            )
        )
    return findings, concerns


def failed_findings(
    out: MethodsOutput, index: QuoteIndex
) -> list[tuple[str, ChecklistFinding | OtherConcern]]:
    """Findings (other than unclear ones, which have nothing to lose) and other concerns with
    no verified quote."""
    failed: list[tuple[str, ChecklistFinding | OtherConcern]] = [
        (f.item.value, f)
        for f in out.findings
        if f.verdict != MethodsVerdict.unclear and not any_verified(check_quotes(index, f.quotes))
    ]
    failed += [
        (other_key(i), c)
        for i, c in enumerate(out.other_concerns[:MAX_OTHER_CONCERNS])
        if not any_verified(check_quotes(index, c.quotes))
    ]
    return failed


def cache_tag_for(version: str, replicate: int) -> str:
    """Replicate 0 is the normal cache entry; replicate k > 0 is an independent sample."""
    return version if replicate == 0 else f"{version}#rep{replicate}"


def _call_checker(
    client: LLMClient,
    paper: ParsedPaper,
    prompt: Prompt,
    model: str,
    max_retries: int,
    replicate: int = 0,
) -> tuple[MethodsOutput, TokenUsage]:
    user = paper_block(paper)
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
                schema=MethodsOutput,
                cache_tag=cache_tag_for(prompt.version, replicate),
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problem = "; ".join(output_problems(out, items_for(prompt.version))) or None
        if problem is None:
            return out, usage
        logger.warning("methods %s retry %d: %s", paper.paper_id, attempt + 1, problem)
    raise MethodsError(
        f"checker output still invalid after {max_retries} retries: {problem}", usage
    )


def _call_repair(
    client: LLMClient,
    paper: ParsedPaper,
    failed: list[tuple[str, ChecklistFinding | OtherConcern]],
    prompt: Prompt,
    model: str,
    cache_tag: str,
) -> tuple[dict[str, list[str]], TokenUsage]:
    """One repair call. A malformed response or unknown keys just leave findings unrepaired."""
    try:
        out, usage = client.structured(
            system=prompt.text,
            user=repair_message(paper, failed),
            model=model,
            schema=QuoteRepairOutput,
            cache_tag=cache_tag,
        )
    except LLMMalformedOutputError as e:
        logger.warning("methods %s repair output malformed: %s", paper.paper_id, e)
        return {}, e.usage or TokenUsage()
    wanted = {k for k, _ in failed}
    got = {r.key: r.quotes for r in out.repairs if r.key in wanted}
    if missing := wanted - got.keys():
        logger.warning("methods %s repair skipped keys: %s", paper.paper_id, sorted(missing))
    return got, usage


def check_paper(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    tier: str,
    prompt: Prompt,
    repair_prompt: Prompt | None,
    model: str,
    effort: str,
    max_retries: int,
    replicate: int = 0,
) -> PaperMethodsReview:
    """Checklist review of one paper; `repair_prompt` None skips the repair round.
    `replicate` k > 0 draws an independent sample (its own cache entries)."""
    out, usage = _call_checker(client, paper, prompt, model, max_retries, replicate)
    index = QuoteIndex.for_paper(paper)
    repaired: dict[str, list[str]] = {}
    failed = failed_findings(out, index)
    if failed and repair_prompt is not None:
        logger.info(
            "methods %s: %d findings with no verified quote; repairing",
            paper.paper_id,
            len(failed),
        )
        repaired, repair_usage = _call_repair(
            client,
            paper,
            failed,
            repair_prompt,
            model,
            cache_tag_for(f"{prompt.version}+{repair_prompt.version}", replicate),
        )
        usage += repair_usage
    findings, concerns = assemble(out, index, repaired, items_for(prompt.version))
    return PaperMethodsReview(
        paper_id=paper.paper_id,
        tier=tier,
        model=model,
        prompt_version=prompt.version
        + (f"+{repair_prompt.version}" if repair_prompt is not None else ""),
        effort=effort,
        findings=findings,
        other_concerns=concerns,
        token_usage=usage,
    )
