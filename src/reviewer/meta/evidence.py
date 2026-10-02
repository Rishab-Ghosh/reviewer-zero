"""Turn each checker's already-computed output into `EvidenceItem`s with stable ids (plan §2).

Pure: no LLM calls, no network. One function per checker; each degrades gracefully (empty list)
when its input is None, since a missing checker is reported as `not_run`, never silently absent.
"""

from reviewer.models import (
    CitationReport,
    ClaimNovelty,
    EvidenceItem,
    FormatCheck,
    FormatReport,
    MethodsVerdict,
    PaperClaims,
    PaperMethodsReview,
    PaperNovelty,
    SupportVerdict,
    WritingReport,
)
from reviewer.novelty.report import duplicate_flag

NOVELTY_TOP_SHOWN = 10


def claims_evidence(claims: PaperClaims | None) -> list[EvidenceItem]:
    if claims is None:
        return []
    if not claims.claims:
        return [
            EvidenceItem(
                id="claims:none",
                checker="claims",
                text="The claims checker ran and found no claim it could state specifically "
                "and support from the paper's own text.",
            )
        ]
    unreviewed = (
        "\n(Pass-one claim: the critic pass failed, so it was not reviewed.)"
        if claims.extraction_metadata.critic_fallback
        else ""
    )
    out = []
    for i, c in enumerate(claims.claims, start=1):
        text = (
            f"Claim: {c.statement}\nMethod: {c.method}\nSetting: {c.setting}\n"
            f"Result: {c.result}{unreviewed}"
        )
        out.append(EvidenceItem(id=f"claim:{i}", checker="claims", text=text))
    return out


def _novelty_claim_evidence(claim_idx: int, c: ClaimNovelty) -> list[EvidenceItem]:
    out = []
    for rank, j in enumerate(c.ranked[:NOVELTY_TOP_SHOWN], start=1):
        p = j.candidate.paper
        year = p.published_date.year if p.published_date else p.year
        text = (
            f"Claim {claim_idx} statement: {c.query.statement}\n"
            f"Prior work #{rank}: {p.title} ({p.key}, {year or 'year unknown'}), "
            f"label {j.label.value}. Reason: {j.reason}"
        )
        out.append(EvidenceItem(id=f"novelty:{claim_idx}:top{rank}", checker="novelty", text=text))
    flag = duplicate_flag(c)
    if flag:
        out.append(
            EvidenceItem(
                id=f"novelty:{claim_idx}:duplicate",
                checker="novelty",
                text=f"Claim {claim_idx} statement: {c.query.statement}\n{flag}",
            )
        )
    return out


def novelty_evidence(novelty: PaperNovelty | None) -> list[EvidenceItem]:
    """The top-10 prior-work list and the duplicate flag, per claim -- never a numeric verdict
    or severity score (kept internal to calibration/evals only, see PLAN_meta.md decisions)."""
    if novelty is None:
        return []
    out = []
    for c in novelty.claims:
        out += _novelty_claim_evidence(c.claim_index + 1, c)
    return out


def citations_evidence(citations: CitationReport | None) -> list[EvidenceItem]:
    if citations is None:
        return []
    out = []
    for ref in citations.references:
        for i, flag in enumerate(ref.flags):
            id_ = f"cite:{ref.ref_id}" if len(ref.flags) == 1 else f"cite:{ref.ref_id}:{i}"
            text = (
                f"Reference {ref.ref_id} ({ref.raw_string[:150]}): {flag.type.value}. {flag.detail}"
            )
            out.append(EvidenceItem(id=id_, checker="citations", text=text))
    for i, s in enumerate(citations.support, start=1):
        if s.verdict == SupportVerdict.cant_tell_from_abstract:
            continue
        text = (
            f"In '{s.section_heading}', citing [{s.ref_id}]: \"{s.sentence}\"\n"
            f"Attributed: {s.attributed}. Verdict: {s.verdict.value}."
            + (f" {s.rationale}" if s.rationale else "")
        )
        out.append(EvidenceItem(id=f"cite:s{i}", checker="citations", text=text))
    return out


def methods_evidence(methods: PaperMethodsReview | None) -> list[EvidenceItem]:
    if methods is None:
        return []
    out = []
    for f in methods.findings:
        if f.verdict == MethodsVerdict.na:
            continue
        text = f"Methods item {f.item.value}: {f.verdict.value}. {f.rationale}"
        if f.concern:
            text += f" Concern: {f.concern}"
        out.append(EvidenceItem(id=f"methods:{f.item.value}", checker="methods", text=text))
    for i, oc in enumerate(methods.other_concerns):
        if not oc.verified:
            continue
        out.append(
            EvidenceItem(id=f"methods:other:{i}", checker="methods", text=f"Concern: {oc.concern}")
        )
    return out


STATUS_REVEALING_FORMAT_CHECKS = frozenset(
    {
        FormatCheck.author_block,
        FormatCheck.email,
        FormatCheck.self_citation,
        FormatCheck.identifying_url,
        FormatCheck.acknowledgments,
        FormatCheck.pdf_metadata,
        FormatCheck.page_limit,
    }
)
"""On OpenReview dev papers, accepted ones come as camera-ready PDFs: these findings (broken
anonymity, camera-ready page limits) would reveal the decision to the meta-reviewer and
penalize accepted papers for being camera-ready. Dropped when blinding decision status."""


def format_evidence(
    fmt: FormatReport | None, drop_checks: frozenset[FormatCheck] = frozenset()
) -> list[EvidenceItem]:
    if fmt is None:
        return []
    findings = [f for f in fmt.findings if f.check not in drop_checks]
    by_check: dict[str, int] = {}
    for f in findings:
        by_check[f.check.value] = by_check.get(f.check.value, 0) + 1
    seen: dict[str, int] = {}
    out = []
    for f in findings:
        seen[f.check.value] = seen.get(f.check.value, 0) + 1
        id_ = (
            f"format:{f.check.value}"
            if by_check[f.check.value] == 1
            else f"format:{f.check.value}:{seen[f.check.value] - 1}"
        )
        text = f"Format check {f.check.value} ({f.severity.value}): {f.detail}"
        out.append(EvidenceItem(id=id_, checker="format", text=text))
    return out


def writing_evidence(writing: WritingReport | None) -> list[EvidenceItem]:
    if writing is None:
        return []
    out = []
    for i, t in enumerate(writing.abstract_trace):
        text = (
            f'Abstract claim: "{t.abstract_span}"\nVerdict: {t.verdict.value}.'
            + (f' Evidence in the paper: "{t.result_span}"' if t.result_span else "")
            + (f" Gap: {t.gap}" if t.gap else "")
        )
        out.append(EvidenceItem(id=f"writing:abstract_claim:{i}", checker="writing", text=text))
    if writing.contribution is not None:
        c = writing.contribution
        text = f'Stated contribution ({c.verdict.value}): "{c.intro_span}"' + (
            f" Problem: {c.problem}" if c.problem else ""
        )
        out.append(EvidenceItem(id="writing:contribution", checker="writing", text=text))
    if writing.related_work is not None:
        out.append(
            EvidenceItem(
                id="writing:related_work",
                checker="writing",
                text=f"Related work discussion: {writing.related_work.value}",
            )
        )
    by_check: dict[str, int] = {}
    for f in writing.findings:
        by_check[f.check.value] = by_check.get(f.check.value, 0) + 1
    seen = dict.fromkeys(by_check, 0)
    for f in writing.findings:
        idx = seen[f.check.value]
        seen[f.check.value] += 1
        text = f'{f.check.value} ({f.location}): "{f.quote}". {f.detail}'
        out.append(EvidenceItem(id=f"writing:{f.check.value}:{idx}", checker="writing", text=text))
    return out
