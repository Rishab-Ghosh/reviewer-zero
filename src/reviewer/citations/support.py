"""Does the cited abstract support what the citing sentence attributes to it?

One LLM call per citing sentence judges every resolved reference it cites. References without an
abstract get `cant_tell_from_abstract` without an LLM call, so a verdict can never rest on the
title alone. The verdict must follow from the judge's fields (`apply_overrides`, recorded in
`overridden_from`): `contradicted` needs a property_type other than none and a sentence_span and
abstract_span found word for word; `supported` needs a word-for-word abstract_span;
`cant_tell_from_abstract` is only for abstracts the judge itself called unusable. Every
contradicted verdict that passes is re-judged independently and stands only if both agree.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from xml.sax.saxutils import escape

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError
from reviewer.models import (
    CitationSupport,
    InTextCitation,
    PropertyType,
    ReferenceCheck,
    SupportBatch,
    SupportJudgment,
    SupportVerdict,
    TokenUsage,
)

logger = logging.getLogger(__name__)

NO_ABSTRACT = "no abstract available from arXiv, OpenAlex or Crossref"


def quote_words(text: str) -> list[str]:
    """Lowercased alphanumeric words. OpenAlex abstracts are rebuilt from an inverted index,
    so punctuation and spacing differ from the original; the words and their order do not."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[^\W_]+", text)


def span_in_text(span: str, text: str) -> bool:
    """The span's words appear contiguously, in order, in the text (word for word; case,
    punctuation and spacing are ignored)."""
    q, a = quote_words(span), quote_words(text)
    if not q:
        return False
    n = len(q)
    return any(a[i : i + n] == q for i in range(len(a) - n + 1))


def apply_overrides(
    j: SupportJudgment, sentence: str, abstract: str
) -> tuple[SupportVerdict, str | None]:
    """Code checks on the judge's fields; the verdict must follow from them. Returns the final
    verdict and why it changed (None if unchanged)."""
    verdict = j.verdict
    if not j.abstract_usable:
        if verdict != SupportVerdict.cant_tell_from_abstract:
            return SupportVerdict.cant_tell_from_abstract, "judge called the abstract unusable"
        return verdict, None
    if verdict == SupportVerdict.cant_tell_from_abstract:
        return SupportVerdict.not_in_abstract, "can't-tell on an abstract the judge called usable"
    if verdict == SupportVerdict.contradicted:
        failed = contradiction_check_failures(j, sentence, abstract)
        if failed:
            return SupportVerdict.not_in_abstract, "contradicted failed checks: " + ", ".join(
                failed
            )
    if verdict == SupportVerdict.supported and not span_in_text(j.abstract_span, abstract):
        return SupportVerdict.cant_tell_from_abstract, "abstract_span not found in the abstract"
    return verdict, None


def contradiction_check_failures(j: SupportJudgment, sentence: str, abstract: str) -> list[str]:
    """Why a contradicted verdict cannot stand: property_type none, or a span not verbatim."""
    failed = []
    if j.property_type == PropertyType.none:
        failed.append("property_type is none")
    if not span_in_text(j.sentence_span, sentence):
        failed.append("sentence_span not in the sentence")
    if not span_in_text(j.abstract_span, abstract):
        failed.append("abstract_span not in the abstract")
    return failed


def build_user_message(
    paper_title: str, citation: InTextCitation, refs: list[ReferenceCheck]
) -> str:
    parts = [
        f"<citing_paper>{escape(paper_title)}</citing_paper>",
        f"<section>{escape(citation.section_heading)}</section>",
        f"<sentence>{escape(citation.sentence)}</sentence>",
        "<references>",
    ]
    for ref in refs:
        assert ref.resolved is not None and ref.resolved.abstract
        parts += [
            f'<reference id="{ref.ref_id}">',
            f"<as_printed>{escape(ref.raw_string)}</as_printed>",
            f"<title>{escape(ref.resolved.title)}</title>",
            f"<abstract>{escape(ref.resolved.abstract)}</abstract>",
            "</reference>",
        ]
    parts.append("</references>")
    return "\n".join(parts)


@dataclass
class SentenceResult:
    judgments: list[CitationSupport]
    usage: TokenUsage
    error: str | None = None


VERIFY_TAG = "verify"


def _abstract(checks: dict[str, ReferenceCheck], ref_id: str) -> str:
    c = checks.get(ref_id)
    return (c.resolved.abstract or "") if c and c.resolved else ""


def _verify(
    user: str,
    sentence: str,
    checks: dict[str, ReferenceCheck],
    ref_ids: set[str],
    client: LLMClient,
    prompt: Prompt,
    model: str,
) -> tuple[dict[str, SupportVerdict], TokenUsage, str | None]:
    """Independent re-judgment: the same system prompt and user message, a separate call (its
    own cache entry) that never sees the first verdict. Returns the verifier's verdict (after
    code checks) for each of `ref_ids`."""
    try:
        batch, usage = client.structured(
            system=prompt.text,
            user=user,
            model=model,
            schema=SupportBatch,
            cache_tag=f"{prompt.version}+{VERIFY_TAG}",
        )
    except MissingAPIKeyError:
        raise
    except LLMError as e:
        return {}, TokenUsage(), f"{type(e).__name__}: {e}"
    out = {}
    for j in batch.judgments:
        if j.ref_id in ref_ids:
            out[j.ref_id] = apply_overrides(j, sentence, _abstract(checks, j.ref_id))[0]
    return out, usage, None


def _no_abstract(citation: InTextCitation, ref_id: str) -> CitationSupport:
    return CitationSupport(
        section_heading=citation.section_heading,
        sentence=citation.sentence,
        ref_id=ref_id,
        verdict=SupportVerdict.cant_tell_from_abstract,
        rationale=NO_ABSTRACT,
        judged_by_llm=False,
    )


def judge_sentence(
    citation: InTextCitation,
    checks: dict[str, ReferenceCheck],
    paper_title: str,
    client: LLMClient,
    prompt: Prompt,
    model: str,
) -> SentenceResult:
    """Judge one citing sentence. Unresolved or unknown ref ids are skipped (not judged)."""
    resolved = [checks[r] for r in citation.ref_ids if r in checks and checks[r].resolved]
    out = [_no_abstract(citation, c.ref_id) for c in resolved if not c.resolved.abstract]  # type: ignore[union-attr]
    to_judge = [c for c in resolved if c.resolved and c.resolved.abstract]
    if not to_judge:
        return SentenceResult(out, TokenUsage())
    user = build_user_message(paper_title, citation, to_judge)
    try:
        batch, usage = client.structured(
            system=prompt.text,
            user=user,
            model=model,
            schema=SupportBatch,
            cache_tag=prompt.version,
        )
    except MissingAPIKeyError:
        raise
    except LLMError as e:  # recorded by the caller
        return SentenceResult(out, TokenUsage(), f"{type(e).__name__}: {e}")
    verify_ids = {
        j.ref_id
        for j in batch.judgments
        if apply_overrides(j, citation.sentence, _abstract(checks, j.ref_id))[0]
        == SupportVerdict.contradicted
    }
    verifier: dict[str, SupportVerdict] = {}
    v_error: str | None = None
    if verify_ids:
        verifier, v_usage, v_error = _verify(
            user, citation.sentence, checks, verify_ids, client, prompt, model
        )
        usage = usage + v_usage
    by_id = {j.ref_id: j for j in batch.judgments}
    missing = []
    for c in to_judge:
        j = by_id.get(c.ref_id)
        if j is None:
            missing.append(c.ref_id)
            continue
        abstract = c.resolved.abstract or ""  # type: ignore[union-attr]
        verdict, reason = apply_overrides(j, citation.sentence, abstract)
        second = verifier.get(c.ref_id)
        if verdict == SupportVerdict.contradicted and second != SupportVerdict.contradicted:
            verdict = SupportVerdict.not_in_abstract
            reason = (
                "contradicted not confirmed by independent re-judgment "
                f"(verifier: {second.value if second else 'call failed' if v_error else 'none'})"
            )
            logger.info("%s: %s [%s]", c.ref_id, reason, citation.sentence[:80])
        out.append(
            CitationSupport(
                section_heading=citation.section_heading,
                sentence=citation.sentence,
                ref_id=c.ref_id,
                verdict=verdict,
                attributed=j.attributed,
                quote=j.abstract_span,
                rationale=j.explanation,
                property_type=j.property_type,
                sentence_span=j.sentence_span,
                judged_by_llm=True,
                verifier_verdict=second,
                overridden_from=j.verdict if reason else None,
                override_reason=reason,
            )
        )
    errors = [f"judge returned no verdict for {missing}"] if missing else []
    if v_error:
        errors.append(f"verifier: {v_error}")
    return SentenceResult(out, usage, "; ".join(errors) or None)
