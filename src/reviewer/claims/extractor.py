"""Two-pass claim extraction: extract (pass one) then critic (pass two).

The final PaperClaims is built from the critic's output; the pass-one draft is returned
alongside it for diagnosis. Drafts with the wrong number of claims or keywords, and critic
outputs whose review notes do not describe exactly what changed between the draft and `final`,
are retried with the problems appended to the user message. Nothing is ever padded, truncated,
or silently "fixed".
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from reviewer.claims.grounding import ground_claims
from reviewer.claims.prompt_builder import build_context, render, render_full
from reviewer.claims.prompts import Prompt, load_prompt
from reviewer.claims.verify import VerificationError, verify_claims
from reviewer.config import Settings
from reviewer.llm.client import (
    LLMClient,
    LLMError,
    LLMMalformedOutputError,
    MissingAPIKeyError,
)
from reviewer.models import (
    ClaimGrounding,
    ClaimsDraft,
    ClaimVerdict,
    CriticOutput,
    Defect,
    ExtractionMetadata,
    GroundingReport,
    PaperClaims,
    ParsedPaper,
    TokenUsage,
    VerificationRecord,
)

logger = logging.getLogger(__name__)

MIN_CLAIMS, MAX_CLAIMS = 1, 4
MIN_KEYWORDS, MAX_KEYWORDS = 5, 10


class ExtractionError(RuntimeError):
    """Extraction failed after all retries. `pass1` is the pass-one draft if it succeeded."""

    def __init__(
        self, message: str, pass1: ClaimsDraft | None = None, usage: TokenUsage | None = None
    ) -> None:
        super().__init__(message)
        self.pass1 = pass1
        self.usage = usage


def draft_problems(draft: ClaimsDraft) -> list[str]:
    """Count violations that make a draft unusable; empty when the draft is valid."""
    problems = []
    if not MIN_CLAIMS <= len(draft.claims) <= MAX_CLAIMS:
        problems.append(f"expected {MIN_CLAIMS} to {MAX_CLAIMS} claims, got {len(draft.claims)}")
    if not MIN_KEYWORDS <= len(draft.keywords) <= MAX_KEYWORDS:
        problems.append(
            f"expected {MIN_KEYWORDS} to {MAX_KEYWORDS} keywords, got {len(draft.keywords)}"
        )
    return problems


def defect_problems(where: str, verdict: ClaimVerdict, defects: list[Defect]) -> list[str]:
    """Defect rules for one review note."""
    problems = []
    if Defect.truism in defects and verdict != ClaimVerdict.rejected:
        problems.append(
            f"{where}: a claim with the truism defect must be rejected (removed from final), "
            "never rewritten or kept"
        )
    if verdict == ClaimVerdict.kept and defects:
        problems.append(
            f"{where}: a kept claim must have no defects, got "
            f"{[d.value for d in defects]}; fix it (rewritten) or reject it"
        )
    if verdict in (ClaimVerdict.rewritten, ClaimVerdict.rejected) and not defects:
        problems.append(f"{where}: a {verdict.value} claim must list at least one defect")
    return problems


def uses_defects(critic_prompt_version: str) -> bool:
    """Defect rules exist from critic_v3 on; v1/v2 critics never emit `defects`, so their
    (cached) outputs stay valid when re-run."""
    m = re.search(r"_v(\d+)$", critic_prompt_version)
    return bool(m) and int(m.group(1)) >= 3


def critic_consistency_problems(
    draft: ClaimsDraft, critic: CriticOutput, check_defects: bool = True
) -> list[str]:
    """Ways the critic's review notes misdescribe what changed between `draft` and
    `critic.final`, or break the defect rules (kept has no defects; rewritten and rejected
    list at least one; a truism is always rejected, never rewritten); empty when the notes
    match. See CriticNote for the semantics."""
    n_draft, n_final = len(draft.claims), len(critic.final.claims)
    problems: list[str] = []
    draft_refs: dict[int, int] = {}
    final_refs: dict[int, int] = {}
    for k, note in enumerate(critic.review):
        where = f"review[{k}] ({note.verdict.value})"
        d, f = note.draft_index, note.final_index
        if (d is None) != (note.verdict == ClaimVerdict.added):
            problems.append(
                f"{where}: draft_index must be null exactly when the verdict is 'added', "
                f"got draft_index={d}"
            )
        if (f is None) != (note.verdict == ClaimVerdict.rejected):
            problems.append(
                f"{where}: final_index must be null exactly when the verdict is 'rejected', "
                f"got final_index={f}"
            )
        d_ok = d is not None and 0 <= d < n_draft
        f_ok = f is not None and 0 <= f < n_final
        if d is not None and not d_ok:
            problems.append(f"{where}: draft_index {d} out of range (draft has {n_draft} claims)")
        if f is not None and not f_ok:
            problems.append(f"{where}: final_index {f} out of range (final has {n_final} claims)")
        if d_ok and d is not None:
            draft_refs[d] = draft_refs.get(d, 0) + 1
        if f_ok and f is not None:
            final_refs[f] = final_refs.get(f, 0) + 1
        both = d_ok and f_ok and d is not None and f is not None
        same = both and draft.claims[d].model_dump() == critic.final.claims[f].model_dump()  # type: ignore[index]
        if check_defects:
            if note.verdict == ClaimVerdict.kept and both and not same:
                # reconcile_kept relabels this 'rewritten'; only a truism is still disallowed
                if Defect.truism in note.defects:
                    problems.extend(defect_problems(where, ClaimVerdict.rewritten, note.defects))
            else:
                problems.extend(defect_problems(where, note.verdict, note.defects))
        if both and note.verdict == ClaimVerdict.rewritten and same:
            problems.append(
                f"{where}: final claim {f} is identical to draft claim {d}, so the verdict "
                "must be 'kept'"
            )
    for i in range(n_draft):
        count = draft_refs.get(i, 0)
        if count != 1:
            problems.append(f"draft claim {i} has {count} review notes; it needs exactly one")
    for j in range(n_final):
        count = final_refs.get(j, 0)
        if count != 1:
            problems.append(
                f"final claim {j} is referenced by {count} review notes; it needs exactly one "
                "(its kept, rewritten, or added note)"
            )
    return problems


def reconcile_kept(draft: ClaimsDraft, critic: CriticOutput) -> tuple[CriticOutput, list[int]]:
    """Relabel review notes marked 'kept' whose final claim differs from the draft claim as
    'rewritten' (the critic edited it and said it didn't). Returns the notes' indices."""
    fixed: list[int] = []
    review = []
    for k, note in enumerate(critic.review):
        d, f = note.draft_index, note.final_index
        if (
            note.verdict == ClaimVerdict.kept
            and d is not None
            and f is not None
            and 0 <= d < len(draft.claims)
            and 0 <= f < len(critic.final.claims)
            and draft.claims[d].model_dump() != critic.final.claims[f].model_dump()
        ):
            note = note.model_copy(update={"verdict": ClaimVerdict.rewritten})
            fixed.append(k)
        review.append(note)
    return critic.model_copy(update={"review": review}), fixed


def with_feedback(user: str, problem: str | None, attempt: int) -> str:
    """Append the last validation error. The attempt number makes each retry's cache key
    distinct, so a repeated error does not replay the same cached bad answer."""
    if problem is None:
        return user
    return (
        f"{user}\n\n<previous_attempt_error>\nAttempt {attempt}: your previous answer was "
        f"rejected: {problem}. "
        f"Fix this and answer again, following every rule. Return only as many claims as the "
        f"paper supports ({MIN_CLAIMS} to {MAX_CLAIMS}) and {MIN_KEYWORDS} to {MAX_KEYWORDS} "
        f"keywords, choosing what matters most rather than cutting or padding mechanically.\n"
        f"</previous_attempt_error>"
    )


def critic_user_message(context_text: str, draft: ClaimsDraft) -> str:
    return (
        f"{context_text}\n\n<pass_one_draft>\n{draft.model_dump_json(indent=2)}\n</pass_one_draft>"
    )


def add_usage(total: TokenUsage | None, usage: TokenUsage) -> TokenUsage:
    """Sum usages; starting from None keeps `from_cache` true when every call was cached."""
    return usage if total is None else total + usage


@dataclass
class PassResult[T: BaseModel]:
    output: T
    usage: TokenUsage
    retries: int


def run_pass[T: BaseModel](
    client: LLMClient,
    *,
    name: str,
    system: str,
    user: str,
    model: str,
    schema: type[T],
    cache_tag: str,
    problems_of: Callable[[T], list[str]],
    max_retries: int,
) -> PassResult[T]:
    """Call the LLM until `problems_of(output)` is empty, up to 1 + max_retries attempts.
    Malformed output is retried too; refusals and truncation are not."""
    usage: TokenUsage | None = None
    problem: str | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            logger.warning("%s pass retry %d/%d: %s", name, attempt, max_retries, problem)
        try:
            output, call_usage = client.structured(
                system=system,
                user=with_feedback(user, problem, attempt + 1),
                model=model,
                schema=schema,
                cache_tag=cache_tag,
            )
        except LLMMalformedOutputError as e:
            usage = add_usage(usage, e.usage)
            problem = str(e)
            continue
        except MissingAPIKeyError:
            raise
        except LLMError as e:
            raise ExtractionError(f"{name} pass: {e}", usage=add_usage(usage, e.usage)) from e
        usage = add_usage(usage, call_usage)
        problems = problems_of(output)
        if not problems:
            return PassResult(output=output, usage=usage, retries=attempt)
        problem = "; ".join(problems)
    raise ExtractionError(
        f"{name} pass still invalid after {max_retries} retries: {problem}", usage=usage
    )


def run_critic(
    client: LLMClient,
    *,
    context_text: str,
    draft: ClaimsDraft,
    model: str,
    critic_prompt: Prompt,
    cache_tag: str,
    max_retries: int,
    extra: str = "",
) -> PassResult[CriticOutput]:
    """The critic pass alone: review `draft` against the rendered context, retrying until the
    final set is valid and the review notes are consistent with it. `extra` is appended to the
    user message (e.g. number-grounding errors from a previous critic answer)."""
    return run_pass(
        client,
        name="critic",
        system=critic_prompt.text,
        user=critic_user_message(context_text, draft) + extra,
        model=model,
        schema=CriticOutput,
        cache_tag=cache_tag,
        problems_of=lambda c: (
            draft_problems(c.final)
            + critic_consistency_problems(
                draft, c, check_defects=uses_defects(critic_prompt.version)
            )
        ),
        max_retries=max_retries,
    )


FULL_PAPER_CRITICS = frozenset({"critic_v4"})
"""Critic prompt versions that see the whole paper (render_full) instead of the excerpts."""


def grounding_feedback(final: ClaimsDraft, missing: list[ClaimGrounding]) -> str:
    lines = [
        f'- claim {g.claim_index} ("{final.claims[g.claim_index].statement[:80]}..."): '
        + ", ".join(g.missing)
        for g in missing
    ]
    return (
        "\n\n<number_grounding_errors>\nA code check found numbers in your previous final claims "
        "(statement, setting, or result) that appear nowhere in the paper text, even after "
        "normalizing formats such as 0.5 vs 50% and 1,000 vs 1000:\n"
        + "\n".join(lines)
        + "\nFor each one, copy the number exactly as the paper gives it and attach it to the "
        "result it belongs to, or remove it. Never state a number the paper does not state "
        "(no computed ratios, differences, totals, or rounded summaries).\n"
        f"Your previous final claims were:\n{final.model_dump_json(indent=2)}\n"
        "</number_grounding_errors>"
    )


def ground_with_critic(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    context_text: str,
    draft: ClaimsDraft,
    critic_pass: PassResult[CriticOutput],
    model: str,
    critic_prompt: Prompt,
    cache_tag: str,
    max_retries: int,
    max_rounds: int,
) -> tuple[PassResult[CriticOutput], GroundingReport]:
    """Send claims with ungrounded numbers back to the critic (with the list) up to `max_rounds`
    times. Never fails the paper: what is still ungrounded is recorded."""
    initial = ground_claims(critic_pass.output.final.claims, paper)
    current, missing, rounds = critic_pass, initial, 0
    usage = critic_pass.usage
    while missing and rounds < max_rounds:
        rounds += 1
        logger.warning(
            "%s: %d claim(s) with ungrounded numbers, critic round %d: %s",
            paper.paper_id,
            len(missing),
            rounds,
            [(g.claim_index, g.missing) for g in missing],
        )
        try:
            nxt = run_critic(
                client,
                context_text=context_text,
                draft=draft,
                model=model,
                critic_prompt=critic_prompt,
                cache_tag=cache_tag,
                max_retries=max_retries,
                extra=grounding_feedback(current.output.final, missing),
            )
        except ExtractionError as e:
            logger.warning(
                "%s: grounding round %d failed (%s); keeping the previous claims",
                paper.paper_id,
                rounds,
                e,
            )
            break
        usage = usage + nxt.usage
        current = PassResult(output=nxt.output, usage=usage, retries=current.retries + nxt.retries)
        missing = ground_claims(current.output.final.claims, paper)
    report = GroundingReport(initial=initial, final=missing, rounds=rounds)
    return PassResult(output=current.output, usage=usage, retries=current.retries), report


def build_paper_claims(
    paper: ParsedPaper,
    final: ClaimsDraft,
    critic: CriticOutput,
    model: str,
    prompt_version: str,
    extract: PassResult[ClaimsDraft],
    critic_pass: PassResult[CriticOutput],
    grounding: GroundingReport | None = None,
    verification: tuple[list[VerificationRecord], TokenUsage] | None = None,
    reconciled: list[int] | None = None,
    critic_fallback: str | None = None,
) -> PaperClaims:
    pass_usage = {"extract": extract.usage, "critic": critic_pass.usage}
    total = extract.usage + critic_pass.usage
    if verification is not None:
        pass_usage["verify"] = verification[1]
        total = total + verification[1]
    return PaperClaims(
        paper_id=paper.paper_id,
        title=paper.title,
        one_line_summary=final.one_line_summary,
        claims=final.claims,
        keywords=final.keywords,
        extraction_metadata=ExtractionMetadata(
            model=model,
            prompt_version=prompt_version,
            timestamp=datetime.now(UTC),
            token_usage=total,
            pass_usage=pass_usage,
            retries=extract.retries + critic_pass.retries,
            critic_review=critic.review,
            grounding=grounding,
            verification=verification[0] if verification is not None else None,
            critic_reconciled=reconciled or [],
            critic_fallback=critic_fallback,
        ),
    )


def extract_claims(
    paper: ParsedPaper,
    model: str,
    client: LLMClient,
    settings: Settings,
    extract_prompt: Prompt | None = None,
    critic_prompt: Prompt | None = None,
) -> tuple[PaperClaims, ClaimsDraft]:
    """Run both passes; return (final claims from the critic, pass-one draft).

    Prompts default to `settings.extract_prompt` / `settings.critic_prompt`. Raises
    ExtractionError when either pass fails after `settings.llm_max_retries` retries."""
    extract_prompt = extract_prompt or load_prompt(settings.extract_prompt)
    critic_prompt = critic_prompt or load_prompt(settings.critic_prompt)
    prompt_version = f"{extract_prompt.version}+{critic_prompt.version}"
    context_text = render(build_context(paper, table_char_budget=settings.table_char_budget))

    extract = run_pass(
        client,
        name="extract",
        system=extract_prompt.text,
        user=context_text,
        model=model,
        schema=ClaimsDraft,
        cache_tag=extract_prompt.version,
        problems_of=draft_problems,
        max_retries=settings.llm_max_retries,
    )
    draft = extract.output

    critic_context = (
        render_full(paper) if critic_prompt.version in FULL_PAPER_CRITICS else context_text
    )
    grounding = None
    fallback: str | None = None
    reconciled: list[int] = []
    try:
        critic_pass = run_critic(
            client,
            context_text=critic_context,
            draft=draft,
            model=model,
            critic_prompt=critic_prompt,
            cache_tag=prompt_version,
            max_retries=settings.llm_max_retries,
        )
    except MissingAPIKeyError:
        raise
    except ExtractionError as e:
        # keep the paper's claims: pass one, flagged as not reviewed by the critic
        fallback = str(e)
        logger.warning("%s: critic failed, using pass-one claims unreviewed: %s", paper.paper_id, e)
        critic_pass = PassResult(
            output=CriticOutput(review=[], final=draft),
            usage=e.usage or TokenUsage(),
            retries=settings.llm_max_retries,
        )

    if settings.number_grounding and fallback is None:
        critic_pass, grounding = ground_with_critic(
            client,
            paper,
            context_text=critic_context,
            draft=draft,
            critic_pass=critic_pass,
            model=model,
            critic_prompt=critic_prompt,
            cache_tag=prompt_version,
            max_retries=settings.llm_max_retries,
            max_rounds=settings.grounding_max_rounds,
        )
    critic, reconciled = reconcile_kept(draft, critic_pass.output)
    if reconciled:
        logger.info(
            "%s: critic notes %s said 'kept' but edited the claim; relabelled 'rewritten'",
            paper.paper_id,
            reconciled,
        )
    final = critic.final
    verification = None
    if settings.verify_prompt:
        verify_prompt = load_prompt(settings.verify_prompt)
        prompt_version = f"{prompt_version}+{verify_prompt.version}"
        try:
            kept, records, v_usage = verify_claims(
                client,
                final.claims,
                paper_context=render_full(paper),
                prompt=verify_prompt,
                model=model,
                max_retries=settings.llm_max_retries,
            )
        except VerificationError as e:
            raise ExtractionError(f"verification: {e}", pass1=draft) from e
        final = final.model_copy(update={"claims": kept})
        verification = (records, v_usage)
        logger.info(
            "%s verification: %s",
            paper.paper_id,
            ", ".join(f"{r.input_index}:{r.outcome}" for r in records),
        )
    try:
        claims = build_paper_claims(
            paper,
            final,
            critic,
            model,
            prompt_version,
            extract,
            critic_pass,
            grounding,
            verification,
            reconciled,
            fallback,
        )
    except ValidationError as e:
        raise ExtractionError(f"final claims failed validation: {e}", pass1=draft) from e
    logger.info(
        "%s: %d claims (pass one had %d), %d retries, $%.4f%s%s",
        paper.paper_id,
        len(claims.claims),
        len(draft.claims),
        claims.extraction_metadata.retries,
        claims.extraction_metadata.token_usage.cost_usd,
        " (cached)" if claims.extraction_metadata.token_usage.from_cache else "",
        " [critic failed: pass-one claims]" if fallback else "",
    )
    return claims, draft
