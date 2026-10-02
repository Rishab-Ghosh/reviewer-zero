"""Assemble one `MetaInput` per paper from whatever checkers have already run (plan §2). Pure
disk reads plus `blind_text`; no LLM calls, no network."""

import logging
from collections.abc import Callable
from pathlib import Path

from reviewer.citations.commands import report_dir as citations_dir
from reviewer.config import Settings
from reviewer.format.commands import report_dir as format_dir
from reviewer.meta.blinding import blind_text, strip_status
from reviewer.meta.evidence import (
    STATUS_REVEALING_FORMAT_CHECKS,
    citations_evidence,
    claims_evidence,
    format_evidence,
    methods_evidence,
    novelty_evidence,
    writing_evidence,
)
from reviewer.models import (
    CheckerName,
    CheckerRunStatus,
    CheckerStatus,
    CitationReport,
    FormatReport,
    MetaInput,
    PaperClaims,
    PaperMethodsReview,
    PaperNovelty,
    ParsedPaper,
    WritingReport,
)
from reviewer.writing.commands import report_dir as writing_dir

logger = logging.getLogger(__name__)


def _load[T](path: Path, model: type[T]) -> T | None:
    if not path.exists():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))  # type: ignore[attr-defined]


def _load_private_or_public(
    settings: Settings, paper_id: str, dir_fn: Callable[[Settings, bool], Path], model: type
) -> object | None:
    """citations/format/writing write to reports/<checker>/ or reports/private/<checker>/
    depending on tier; try private first (only own/lab papers live there)."""
    for private in (True, False):
        out = _load(dir_fn(settings, private) / f"{paper_id}.json", model)
        if out is not None:
            return out
    return None


def load_checkers(
    settings: Settings, paper_id: str
) -> tuple[
    PaperClaims | None,
    PaperNovelty | None,
    CitationReport | None,
    PaperMethodsReview | None,
    FormatReport | None,
    WritingReport | None,
]:
    claims = _load(settings.claims_dir / f"{paper_id}.json", PaperClaims)
    novelty = _load(settings.novelty_dir / f"{paper_id}.json", PaperNovelty)
    citations = _load_private_or_public(settings, paper_id, citations_dir, CitationReport)
    methods = _load(settings.methods_dir / f"{paper_id}.json", PaperMethodsReview)
    fmt = _load_private_or_public(settings, paper_id, format_dir, FormatReport)
    writing = _load_private_or_public(settings, paper_id, writing_dir, WritingReport)
    return claims, novelty, citations, methods, fmt, writing


def checker_status(
    name: CheckerName, output: object | None, failure: str | None, detail: str = ""
) -> CheckerStatus:
    if output is not None:
        return CheckerStatus(checker=name, status=CheckerRunStatus.ok, detail=detail)
    if failure:
        return CheckerStatus(checker=name, status=CheckerRunStatus.failed, detail=failure)
    return CheckerStatus(checker=name, status=CheckerRunStatus.not_run, detail=detail)


def claims_detail(claims: PaperClaims | None) -> str:
    if claims is None:
        return ""
    if not claims.claims:
        return "found 0 claims"
    if claims.extraction_metadata.critic_fallback:
        return "pass-one claims; the critic pass failed, so they were not reviewed"
    return ""


def novelty_not_run_reason(claims: PaperClaims | None) -> str:
    """Novelty checks claims, so without any it cannot run; say why rather than a bare not_run."""
    if claims is None:
        return "no claims to check: the claims checker did not produce any"
    if not claims.claims:
        return "no claims to check: the claims checker found none"
    return ""


def build_meta_input(
    settings: Settings,
    paper: ParsedPaper,
    *,
    venue: str,
    venue_year: int | None = None,
    failures: dict[str, str] | None = None,
) -> MetaInput:
    """`failures` maps checker name -> recorded error, for checkers that ran but failed (as
    opposed to never having been run at all)."""
    failures = failures or {}
    claims, novelty, citations, methods, fmt, writing = load_checkers(settings, paper.paper_id)

    evidence = (
        claims_evidence(claims)
        + novelty_evidence(novelty)
        + citations_evidence(citations)
        + methods_evidence(methods)
        + format_evidence(
            fmt,
            STATUS_REVEALING_FORMAT_CHECKS if settings.meta_blind_decision_status else frozenset(),
        )
        + writing_evidence(writing)
    )
    evidence = [e.model_copy(update={"text": strip_status(e.text)}) for e in evidence]
    status = [
        checker_status(
            CheckerName.claims,
            claims,
            failures.get("claims"),
            claims_detail(claims),
        ),
        checker_status(
            CheckerName.novelty, novelty, failures.get("novelty"), novelty_not_run_reason(claims)
        ),
        checker_status(CheckerName.citations, citations, failures.get("citations")),
        checker_status(CheckerName.methods, methods, failures.get("methods")),
        checker_status(CheckerName.format, fmt, failures.get("format")),
        checker_status(CheckerName.writing, writing, failures.get("writing")),
    ]
    for s in status:
        if s.status != CheckerRunStatus.ok:
            logger.info(
                "meta input %s: checker %s is %s", paper.paper_id, s.checker.value, s.status.value
            )

    return MetaInput(
        paper_id=paper.paper_id,
        title=strip_status(paper.title),
        venue=venue,
        venue_year=venue_year,
        blinded_text=blind_text(paper),
        evidence=evidence,
        checker_status=status,
    )
