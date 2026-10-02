"""`reviewer methods review-eval`: the checker against OpenReview reviewers' methodology
concerns, dev split only (test is locked). Outputs are private (reports/private/).

Per submission: parse (existing or-<id>.json; else the PDF reviewers saw, downloaded from
OpenReview and parsed with GROBID) -> checker (methods_v1, unchanged) -> matcher (match_v1,
unchanged). Then the precision sample. Spend stops at --max-cost; --dry-run only estimates.
"""

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import typer

from reviewer.cache import sha256_file
from reviewer.config import MODEL_PRICES_PER_MTOK, Settings, get_settings
from reviewer.eval.judge import judge_context_full
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError
from reviewer.methods.checker import MethodsError, check_paper, load_methods_prompt
from reviewer.methods.commands import MATCH_PROMPT, REPAIR_PROMPT, VERIFY_PROMPT, methods_client
from reviewer.methods.evaluate import match_paper, outcomes, precision_sample, verify_paper
from reviewer.methods.openreview_pdf import OpenReviewError, OpenReviewPdfClient
from reviewer.methods.report import render_precision_grading
from reviewer.methods.review_eval import (
    Submission,
    cases_path,
    concern_outcomes,
    group_submissions,
    load_rows,
    paired,
    render_compare,
    render_review_eval,
    review_supported,
)
from reviewer.models import ConcernOutcome, PaperMethodsReview, ParsedPaper
from reviewer.novelty.commands import git_state
from reviewer.novelty.evaluate import venue_cutoff
from reviewer.parse.commands import docling_tables_or_none, parse_pdf
from reviewer.parse.docling_tables import DoclingTableExtractor
from reviewer.parse.grobid_client import GrobidClient
from reviewer.privacy import append_private, write_private

logger = logging.getLogger(__name__)

SAMPLE_SEED = 20260926
CHARS_PER_TOKEN = 3.5
EST_OUTPUT_TOKENS = 3500
"""Checker output tokens per paper at effort medium (31-paper run: 2.5-3.5k)."""
EST_MATCH_USD = 0.03


class OpenReviewParser:
    """Lazily opens OpenReview and GROBID; never starts a server."""

    def __init__(self, settings: Settings, fetch: bool) -> None:
        self.settings = settings
        self.fetch = fetch
        self._pdf: OpenReviewPdfClient | None = None
        self._grobid: GrobidClient | None = None
        self._docling: DoclingTableExtractor | None = None
        self.provenance: dict[str, str] = {}

    def existing(self, sub: Submission) -> ParsedPaper | None:
        """A saved parse, or None. An unreadable one (e.g. from an interrupted run) is removed
        so it is parsed again."""
        path = self.settings.openreview_parsed_dir / f"{sub.paper_id}.json"
        if not path.exists():
            return None
        try:
            paper = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as e:
            logger.warning("unreadable parse %s (%s); parsing again", path.name, str(e)[:80])
            path.unlink()
            return None
        note = self.settings.openreview_pdf_dir / f"{sub.submission_id}.revision.json"
        try:
            prov = json.loads(note.read_text(encoding="utf-8"))["provenance"]
        except (OSError, ValueError, KeyError):
            prov = "existing_parse"
        self.provenance[sub.submission_id] = prov
        return paper

    def get(self, sub: Submission) -> ParsedPaper | None:
        """The parse, or None (failure recorded)."""
        if (paper := self.existing(sub)) is not None:
            return paper
        if not self.fetch:
            record_failure(
                self.settings.failures_file, "methods", sub.paper_id, "no parse and --no-fetch"
            )
            return None
        try:
            if self._pdf is None:
                self._pdf = OpenReviewPdfClient(self.settings)
            deadline = sub.cutoff_date or venue_cutoff(sub.venue)
            fetched = self._pdf.download(sub.submission_id, sub.venue, deadline)
            pdf, reason = fetched.path, fetched.reason
            self.provenance[sub.submission_id] = fetched.provenance
            if self._grobid is None:
                self._grobid = GrobidClient(self.settings)
                if not self._grobid.is_alive():
                    raise OpenReviewError(
                        f"GROBID not reachable at {self.settings.grobid_url} (`make grobid-up`)"
                    )
                self._docling = DoclingTableExtractor(self.settings)
            assert self._docling is not None
            docling = docling_tables_or_none(self._docling, pdf, sub.paper_id, self.settings)
            paper = parse_pdf(self._grobid, pdf, sub.paper_id, sha256_file(pdf), docling=docling)
        except (OpenReviewError, OSError, ValueError, RuntimeError) as e:
            record_failure(
                self.settings.failures_file, "methods", sub.paper_id, f"{type(e).__name__}: {e}"
            )
            return None
        out = self.settings.openreview_parsed_dir / f"{sub.paper_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(out)
        logger.info("parsed %s (%s)", sub.paper_id, reason)
        return paper

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
        if self._grobid is not None:
            self._grobid.close()


def estimate_usd(settings: Settings, papers: list[ParsedPaper], n_unparsed: int) -> float:
    """Checker + matcher cost; unparsed submissions priced at the mean parsed size (or 60k
    characters)."""
    in_rate, out_rate = MODEL_PRICES_PER_MTOK[settings.methods_model]
    sizes = [len(judge_context_full(p)) for p in papers]
    mean = sum(sizes) / len(sizes) if sizes else 60_000
    chars = sum(sizes) + mean * n_unparsed
    n = len(papers) + n_unparsed
    return (chars / CHARS_PER_TOKEN * in_rate + n * EST_OUTPUT_TOKENS * out_rate) / 1e6 + (
        n * EST_MATCH_USD
    )


def sample_submissions(
    subs: list[Submission],
    limit: int | None,
    seed: int = SAMPLE_SEED,
    exclude_sample: int | None = None,
) -> list[Submission]:
    """A seeded sample of `limit` submissions (all when None). `exclude_sample` N first removes
    the submissions the default-seed sample of N picks (the tuning sample), so a second sample
    is disjoint from it."""
    if exclude_sample:
        taken = {s.submission_id for s in sample_submissions(subs, exclude_sample)}
        subs = [s for s in subs if s.submission_id not in taken]
    if limit is None or limit >= len(subs):
        return subs
    picked = random.Random(seed).sample(subs, limit)
    return sorted(picked, key=lambda s: s.submission_id)


def run_review_eval(
    split: str = "dev",
    cases: Path | None = None,
    limit: int | None = None,
    max_cost: float = 5.0,
    fetch: bool = True,
    dry_run: bool = False,
    precision_n: int = 20,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    seed: int = SAMPLE_SEED,
    exclude_sample: int | None = None,
    tag: str = "",
) -> list[ConcernOutcome]:
    """`tag` names the outputs (methods_review_eval_dev_<tag>.md/.jsonl); empty = the old
    names."""
    settings = settings or get_settings()
    path = cases_path(settings.methods_eval_dir, split, cases)
    if not path.exists():
        logger.error("no %s yet (the index session writes it)", path)
        raise typer.Exit(code=1)
    subs = sample_submissions(group_submissions(load_rows(path)), limit, seed, exclude_sample)
    n_concerns = sum(len(s.concerns) for s in subs)
    logger.info("methods review-eval [dev]: %d submissions, %d concerns", len(subs), n_concerns)
    parser = OpenReviewParser(settings, fetch)
    if dry_run:
        have = [p for s in subs if (p := parser.existing(s)) is not None]
        est = estimate_usd(settings, have, len(subs) - len(have))
        typer.echo(
            f"{len(subs)} dev submissions, {n_concerns} concerns; {len(have)} already parsed, "
            f"{len(subs) - len(have)} to download+parse; estimated LLM cost ${est:.2f} "
            "(plus the precision sample, ~$0.15 per sampled paper)"
        )
        return []

    client = client or methods_client(settings)
    prompt = load_methods_prompt(settings.methods_prompt)
    repair = load_methods_prompt(REPAIR_PROMPT) if settings.methods_repair else None
    match_prompt = load_methods_prompt(MATCH_PROMPT)
    out_dir = settings.methods_review_eval_dir
    items: list[ConcernOutcome] = []
    reviews: list[PaperMethodsReview] = []
    papers: dict[str, ParsedPaper] = {}
    stopped = None
    try:
        for sub in subs:
            if client.totals.fresh.cost_usd >= max_cost:
                stopped = f"stopped at ${client.totals.fresh.cost_usd:.2f} (--max-cost {max_cost})"
                logger.warning("%s; %s and later not run", stopped, sub.submission_id)
                break
            paper = parser.get(sub)
            if paper is None:
                continue
            papers[sub.paper_id] = paper
            review_file = out_dir / f"{sub.paper_id}.json"
            try:
                if review_file.exists():
                    review = PaperMethodsReview.model_validate_json(review_file.read_text())
                else:  # written atomically (write_private) only after the paper is complete
                    review = check_paper(
                        client,
                        paper,
                        tier="openreview",
                        prompt=prompt,
                        repair_prompt=repair,
                        model=settings.methods_model,
                        effort=settings.methods_effort,
                        max_retries=settings.llm_max_retries,
                    )
                    write_private(review_file, review.model_dump_json(indent=2) + "\n")
                crit = sub.critiques()
                matches, _ = match_paper(
                    client,
                    crit,
                    review,
                    prompt=match_prompt,
                    model=settings.methods_model,
                    max_retries=settings.llm_max_retries,
                )
            except MissingAPIKeyError as e:
                logger.error("%s", e)
                raise typer.Exit(code=1) from e
            except (MethodsError, LLMError) as e:
                record_failure(
                    settings.failures_file, "methods", sub.paper_id, f"{type(e).__name__}: {e}"
                )
                continue
            reviews.append(review)
            prov = parser.provenance.get(sub.submission_id, "")
            asked = {f.item.value for f in review.findings}
            items += [
                o.model_copy(update={"checker": review.prompt_version})
                for o in concern_outcomes(sub, outcomes(crit, review, matches), prov, asked)
            ]
    finally:
        parser.close()

    precision_line = "Precision sample: not run (no reviews)."
    if reviews and precision_n:
        precision_line = _precision(settings, client, reviews, papers, precision_n, max_cost, tag)

    commit, dirty = git_state()
    header = [
        "# Methodology checker vs OpenReview reviewers (dev split, private)",
        "",
        f"{path} · {len(reviews)}/{len(subs)} submissions reviewed, {len(items)} concerns · "
        f"checker {prompt.version} ({settings.methods_model}, effort {settings.methods_effort}), "
        f"matcher {match_prompt.version} · commit {commit[:10]}{' (dirty)' if dirty else ''} · "
        f"new spend ${client.totals.fresh.cost_usd:.2f}" + (f" · {stopped}" if stopped else ""),
        "",
        "Strict: only verified `no` findings and verified other concerns count. Lenient: also "
        "`unclear`. Partial: the finding touches the concern but misses its main point.",
    ]
    base = settings.private_reports_dir / f"methods_review_eval_dev{'_' + tag if tag else ''}"
    report = render_review_eval(items, reviews, precision_line, header)
    write_private(base.with_suffix(".md"), report)
    write_private(base.with_suffix(".jsonl"), "".join(o.model_dump_json() + "\n" for o in items))
    append_private(
        settings.private_reports_dir / "methods_review_eval_runs.jsonl",
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "split": "dev",
                "tag": tag,
                "checker": prompt.version,
                "seed": seed,
                "limit": limit,
                "exclude_sample": exclude_sample,
                "submission_ids": [s.submission_id for s in subs],
                "cases": str(path),
                "commit": commit,
                "dirty": dirty,
                "submissions": len(subs),
                "reviewed": len(reviews),
                "concerns": len(items),
                "spend_usd": round(client.totals.fresh.cost_usd, 4),
                "stopped": stopped,
            }
        ),
    )
    typer.echo(
        f"{len(reviews)}/{len(subs)} submissions, {len(items)} concerns; new spend "
        f"${client.totals.fresh.cost_usd:.2f}; wrote {base.with_suffix('.md')}"
    )
    return items


def _precision(
    settings: Settings,
    client: LLMClient,
    reviews: list[PaperMethodsReview],
    papers: dict[str, ParsedPaper],
    n: int,
    max_cost: float,
    tag: str = "",
) -> str:
    sample = precision_sample(reviews, n)
    write_private(
        settings.private_reports_dir
        / f"methods_review_eval_precision_dev{'_' + tag if tag else ''}.md",
        render_precision_grading(sample),
    )
    prompt = load_methods_prompt(VERIFY_PROMPT)
    labels: list[str] = []
    skipped = 0
    for pid in sorted({r.paper_id for r in sample}):
        concerns = [r for r in sample if r.paper_id == pid]
        if client.totals.fresh.cost_usd >= max_cost:
            skipped += len(concerns)
            continue
        try:
            checks, _ = verify_paper(
                client,
                papers[pid],
                concerns,
                prompt=prompt,
                model=settings.methods_model,
                max_retries=settings.llm_max_retries,
            )
        except (MethodsError, LLMError) as e:
            record_failure(settings.failures_file, "methods", pid, f"verify: {e}")
            continue
        labels += [c.validity.value for c in checks]
    if skipped:
        logger.warning("precision sample: %d findings not audited (--max-cost reached)", skipped)
    if not labels:
        return "Precision sample: verifier produced nothing."
    k = len(labels)
    valid, partly = labels.count("valid"), labels.count("partly_valid")
    return (
        f"Precision sample ({k} strict findings, seeded; Opus verifier with the full paper, "
        f"likely lenient): valid {valid}, partly valid {partly}, invalid {k - valid - partly} "
        f"-> {valid / k:.0%} valid, {(valid + partly) / k:.0%} incl. partly valid"
        + (f" ({skipped} sampled findings not audited: --max-cost reached)" if skipped else "")
    )


def _load_outcomes(settings: Settings, tag: str) -> list[ConcernOutcome]:
    path = settings.private_reports_dir / f"methods_review_eval_dev_{tag}.jsonl"
    if not path.exists():
        logger.error("no %s (run review-eval --tag %s)", path, tag)
        raise typer.Exit(code=1)
    return [
        ConcernOutcome.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _support(settings: Settings, items: list[ConcernOutcome], subs: set[str]) -> tuple[int, int]:
    """Precision lower bound over `subs`, from the saved reviews of the run's checker."""
    checker = items[0].checker.split("+")[0] if items else settings.methods_prompt
    review_dir = settings.private_reports_dir / "methods_review_eval" / checker
    reviews = [
        PaperMethodsReview.model_validate_json(p.read_text(encoding="utf-8"))
        for sid in sorted(subs)
        if (p := review_dir / f"or-{sid}.json").exists()
    ]
    return review_supported(reviews, [o for o in items if o.submission_id in subs])


def run_review_compare(pairs: list[str], settings: Settings | None = None) -> Path:
    """reports/private/methods_review_compare.md: each BASE:NEW pair of review-eval tags on
    the submissions both reviewed."""
    settings = settings or get_settings()
    lines = [
        "# Methods checker versions compared on the same dev submissions (private)",
        "",
        "Strict = verified `no` findings and verified other concerns. Paired: only submissions "
        "both runs reviewed, so a failure in one run does not change the other's denominator.",
        "",
    ]
    for pair in pairs:
        base_tag, _, new_tag = pair.partition(":")
        base, new = _load_outcomes(settings, base_tag), _load_outcomes(settings, new_tag)
        b, n = paired(base, new)
        subs = {o.submission_id for o in b}
        lines += render_compare(
            f"{base_tag} → {new_tag}",
            b,
            n,
            _support(settings, b, subs),
            _support(settings, n, subs),
        )
    out = settings.private_reports_dir / "methods_review_compare.md"
    write_private(out, "\n".join(lines) + "\n")
    typer.echo(f"wrote {out}")
    return out
