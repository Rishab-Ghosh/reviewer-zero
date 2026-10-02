"""Structured methodology checker: the model reports facts per checklist item (applies,
violations with fixed codes, compliance quotes) and code derives the verdict.

  not applies + verified applies_quote            -> na
  any violation (code of this item) + verified quote -> no
  otherwise any verified compliance quote          -> yes
  otherwise                                        -> unclear

The verdict the model's fields imply before quote checking is kept as `model_verdict`; a
difference to unclear is marked `downgraded`. Everything except `check_paper_structured` is
pure. No repair round: an unverified quote simply does not count.
"""

import logging

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.methods.checker import (
    MAX_OTHER_CONCERNS,
    MethodsError,
    any_verified,
    cache_tag_for,
    check_quotes,
    items_for,
    paper_block,
)
from reviewer.methods.quotes import QuoteIndex
from reviewer.models import (
    CheckedConcern,
    CheckedFinding,
    ChecklistItemId,
    MethodsVerdict,
    PaperMethodsReview,
    ParsedPaper,
    StructuredFinding,
    StructuredMethodsOutput,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def implied_verdict(f: StructuredFinding) -> MethodsVerdict:
    """What the fields say, ignoring quote verification (codes of other items ignored)."""
    if not f.applies:
        return MethodsVerdict.na
    if any(v.code.item == f.item.value for v in f.violations):
        return MethodsVerdict.no
    if f.compliance_quotes:
        return MethodsVerdict.yes
    return MethodsVerdict.unclear


def derive(f: StructuredFinding, index: QuoteIndex) -> CheckedFinding:
    """The code-derived verdict for one item. Pure."""
    model_verdict = implied_verdict(f)
    if not f.applies:
        checks = check_quotes(index, [f.applies_quote] if f.applies_quote else [])
        verdict = MethodsVerdict.na if any_verified(checks) else MethodsVerdict.unclear
        return CheckedFinding(
            item=f.item,
            verdict=verdict,
            model_verdict=model_verdict,
            rationale="does not apply",
            concern="",
            quotes=checks,
            downgraded=verdict != model_verdict,
        )
    own = [v for v in f.violations if v.code.item == f.item.value]
    if dropped := [v.code.value for v in f.violations if v.code.item != f.item.value]:
        logger.warning("%s: ignoring violation codes of other items: %s", f.item.value, dropped)
    vchecks = [(v, index.verify(v.quote)) for v in own]
    held = [v for v, ok in vchecks if ok]
    cchecks = check_quotes(index, f.compliance_quotes)
    quotes = check_quotes(index, [v.quote for v in own]) + cchecks
    if held:
        verdict = MethodsVerdict.no
    elif any_verified(cchecks):
        verdict = MethodsVerdict.yes
    else:
        verdict = MethodsVerdict.unclear
    return CheckedFinding(
        item=f.item,
        verdict=verdict,
        model_verdict=model_verdict,
        rationale="; ".join(f"{v.code.value}: {v.explanation}" for v in own) or "no violation",
        concern=held[0].explanation if held else "",
        quotes=quotes,
        downgraded=verdict == MethodsVerdict.unclear and model_verdict != MethodsVerdict.unclear,
        violation_codes=sorted({v.code.value for v in held}),
    )


def structured_problems(out: StructuredMethodsOutput, items: list[ChecklistItemId]) -> list[str]:
    got = [f.item for f in out.findings]
    problems = []
    if missing := [i.value for i in items if i not in got]:
        problems.append(f"missing checklist items: {', '.join(missing)}")
    if dupes := sorted({i.value for i in got if got.count(i) > 1}):
        problems.append(f"items answered more than once: {', '.join(dupes)}")
    if extra := sorted({i.value for i in got if i not in items}):
        problems.append(f"items not on this checklist: {', '.join(extra)}")
    return problems


def check_paper_structured(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    tier: str,
    prompt: Prompt,
    model: str,
    effort: str,
    max_retries: int,
    replicate: int = 0,
) -> PaperMethodsReview:
    items = items_for(prompt.version)
    user = paper_block(paper)
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if problem is None
            else f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: your previous "
            f"answer was rejected: {problem}. Answer again.\n</previous_attempt_error>"
        )
        try:
            out, u = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=StructuredMethodsOutput,
                cache_tag=cache_tag_for(prompt.version, replicate),
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += u
        problem = "; ".join(structured_problems(out, items)) or None
        if problem is None:
            break
        logger.warning("methods %s retry %d: %s", paper.paper_id, attempt + 1, problem)
    else:
        raise MethodsError(f"structured output still invalid: {problem}", usage)
    index = QuoteIndex.for_paper(paper)
    by_item = {f.item: f for f in out.findings}
    findings = [derive(by_item[i], index) for i in items]
    concerns = []
    for c in out.other_concerns[:MAX_OTHER_CONCERNS]:
        checks = check_quotes(index, c.quotes)
        concerns.append(
            CheckedConcern(concern=c.concern, quotes=checks, verified=any_verified(checks))
        )
    return PaperMethodsReview(
        paper_id=paper.paper_id,
        tier=tier,
        model=model,
        prompt_version=prompt.version,
        effort=effort,
        findings=findings,
        other_concerns=concerns,
        token_usage=usage,
    )
