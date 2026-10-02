"""Command-line entry point. Each command delegates to its workstream's `commands` module."""

from pathlib import Path
from typing import Annotated

import typer

from reviewer.logging_setup import configure_logging

app = typer.Typer(no_args_is_help=True, help="Review AI research papers. Week 1: the spine.")
eval_app = typer.Typer(help="Render the grading report, or summarize filled-in grades.")
app.add_typer(eval_app, name="eval")
judge_app = typer.Typer(help="LLM judge for extracted claims (specificity, correctness).")
app.add_typer(judge_app, name="judge")
novelty_app = typer.Typer(help="Novelty check of extracted claims against reviewer-index.")
app.add_typer(novelty_app, name="novelty")
methods_app = typer.Typer(help="Methodology checker: ML checklist with verified quotes.")
app.add_typer(methods_app, name="methods")
meta_app = typer.Typer(help="Meta-reviewer: one review + score from every checker's output.")
app.add_typer(meta_app, name="meta")


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    configure_logging(verbose)


@app.command()
def fetch(
    ids: Annotated[list[str] | None, typer.Argument(help="arXiv ids; default: corpus")] = None,
) -> None:
    """Download corpus PDFs from arXiv into data/papers/arxiv/."""
    from reviewer.corpus.commands import run_fetch

    run_fetch(ids)


@app.command()
def parse(
    path: Annotated[Path, typer.Argument(help="A PDF or a directory of PDFs")] = Path(
        "data/papers"
    ),
    force: Annotated[bool, typer.Option(help="Re-parse even if output exists")] = False,
) -> None:
    """Parse PDFs with GROBID into data/parsed/<paper_id>.json."""
    from reviewer.parse.commands import run_parse

    run_parse(path, force=force)


@app.command()
def extract(
    model: Annotated[str | None, typer.Option(help="Model id; default LLM_MODEL")] = None,
    ids: Annotated[list[str] | None, typer.Option("--id", help="Only these paper ids")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Default data/claims/")] = None,
    force: Annotated[bool, typer.Option(help="Re-extract even if output exists")] = False,
) -> None:
    """Extract 2-4 claims per parsed paper into data/claims/<paper_id>.json."""
    from reviewer.claims.commands import run_extract

    run_extract(model=model, ids=ids, out_dir=out_dir, force=force)


@app.command()
def compare(
    ids: Annotated[list[str] | None, typer.Option("--id", help="Paper ids; default 5")] = None,
    model: Annotated[str | None, typer.Option(help="Default LLM_MODEL_STRONG")] = None,
    force: Annotated[bool, typer.Option(help="Re-extract even if output exists")] = False,
) -> None:
    """Extract with the strong model and write reports/model_comparison.md."""
    from reviewer.eval.compare import run_compare

    run_compare(ids, strong_model=model, force=force)


@app.command("format")
def format_check(
    path: Annotated[Path, typer.Argument(help="A parsed paper JSON or a directory of them")],
    pdf_root: Annotated[
        Path | None, typer.Option(help="Directory that parsed source_path is relative to")
    ] = None,
) -> None:
    """Formatting/anonymization check (no LLM): reports/format/<paper_id>.md (private papers:
    reports/private/format/)."""
    from reviewer.format.commands import run_format

    run_format(path, pdf_root)


@app.command("format-summary")
def format_summary(
    seed: Annotated[int, typer.Option(help="Random seed for the sample")] = 7,
    n: Annotated[int, typer.Option(help="Findings to sample")] = 10,
) -> None:
    """Per-paper counts and a random sample of findings (reports/format/SUMMARY.md)."""
    from reviewer.format.commands import run_summary

    run_summary(seed, n)


@app.command()
def citations(
    path: Annotated[Path, typer.Argument(help="A parsed paper JSON or a directory of them")],
) -> None:
    """Check references and citation support; write reports/citations/<paper_id>.md
    (private papers: reports/private/citations/)."""
    from reviewer.citations.commands import run_citations

    run_citations(path)


@app.command("citations-summary")
def citations_summary(
    seed: Annotated[int, typer.Option(help="Random seed for the sample")] = 7,
    n: Annotated[int, typer.Option(help="Flagged items to sample")] = 10,
    n_not_in: Annotated[int, typer.Option(help="not_in_abstract items to sample")] = 4,
) -> None:
    """Per-paper counts and a random sample of flagged items (reports/citations/SUMMARY.md;
    private papers summarized separately under reports/private/citations/)."""
    from reviewer.citations.commands import run_summary

    run_summary(seed, n, n_not_in)


@app.command()
def gold() -> None:
    """Build gold claims (public famous gold; private author/lab gold) and private critiques."""
    from reviewer.gold.commands import run_gold_build

    run_gold_build()


@eval_app.callback(invoke_without_command=True)
def eval_report(ctx: typer.Context) -> None:
    """Write the private grading report + grades.yaml and the public week1_claims.md."""
    if ctx.invoked_subcommand is None:
        from reviewer.eval.commands import run_eval

        run_eval()


@eval_app.command("summarize")
def eval_summarize(
    grades: Annotated[Path | None, typer.Option(help="Default reports/private/grades.yaml")] = None,
) -> None:
    """Print average grades by contribution type and by paper tier."""
    from reviewer.eval.commands import run_summarize

    run_summarize(grades)


@eval_app.command("parse-quality")
def eval_parse_quality() -> None:
    """Print lost tables and suspicious authors per parsed paper (terminal only)."""
    from reviewer.eval.commands import run_parse_quality

    run_parse_quality()


@novelty_app.command("check")
def novelty_check(
    ids: Annotated[list[str] | None, typer.Argument(help="Paper ids; default: --tier")] = None,
    tier: Annotated[
        list[str] | None, typer.Option(help="Corpus tiers to check; default lab and own")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-check even if output exists")] = False,
    detail: Annotated[
        bool, typer.Option(help="Also write novelty_detail_<id>.md: every judged candidate")
    ] = False,
    claims_dir: Annotated[
        Path | None, typer.Option(help="Claim set to check (default data/claims/)")
    ] = None,
) -> None:
    """Closest prior work + verdict per claim into reports/private/ (needs reviewer-index serve)."""
    from reviewer.models import PaperTier
    from reviewer.novelty.commands import run_check

    run_check(
        ids,
        [PaperTier(t) for t in tier] if tier else None,
        force=force,
        detail=detail,
        claims_dir=claims_dir,
    )


@novelty_app.command("eval")
def novelty_eval(
    split: Annotated[str, typer.Option(help="dev (default; all tuning) or test")] = "dev",
    confirm_test: Annotated[
        bool, typer.Option(help="Required for --split test: only when the user asks for it")
    ] = False,
    limit: Annotated[
        int | None, typer.Option(help="Deterministic sample of N submissions (default: all)")
    ] = None,
    cases: Annotated[
        Path | None, typer.Option(help="Override the split file (default <eval dir>/<split>.jsonl)")
    ] = None,
    replay_index: Annotated[
        str | None,
        typer.Option(help="Replay retrieval from the cache of this index version (no live index)"),
    ] = None,
    retrieval_only: Annotated[
        bool, typer.Option(help="Union/pool recall only: no rerank, no new LLM calls")
    ] = False,
) -> None:
    """Recall@5/@20 of reviewer-named prior work; every run is logged with its commit hash."""
    from reviewer.config import get_settings
    from reviewer.novelty.commands import run_eval

    if replay_index:
        get_settings().index_replay_version = replay_index

    if split not in ("dev", "test"):
        raise typer.BadParameter("--split must be dev or test")
    run_eval(
        split,  # type: ignore[arg-type]
        limit=limit,
        confirm_test=confirm_test,
        cases_path=cases,
        retrieval_only=retrieval_only,
    )


@novelty_app.command("severity-eval")
def novelty_severity_eval(
    limit: Annotated[int, typer.Option(help="Positives and negatives each")] = 50,
    replay_index: Annotated[
        str | None, typer.Option(help="Replay retrieval from this index version's cache")
    ] = None,
) -> None:
    """Dev only: AUC of paper-level severity, positives vs negatives."""
    from reviewer.config import get_settings
    from reviewer.novelty.commands import run_severity_eval

    if replay_index:
        get_settings().index_replay_version = replay_index
    run_severity_eval(limit=limit)


@novelty_app.command("negatives")
def novelty_negatives(
    limit: Annotated[int, typer.Option(help="Seeded sample of N dev negatives")] = 50,
) -> None:
    """Dev only: verdicts on accepted submissions reviewers found novel (false-alarm check)."""
    from reviewer.novelty.commands import run_negatives

    run_negatives(limit=limit)


@novelty_app.command("first-stage-eval")
def novelty_first_stage_eval(
    limit: Annotated[
        int | None, typer.Option(help="Deterministic sample of N dev submissions (default: all)")
    ] = None,
    pool_sizes: Annotated[
        str, typer.Option(help="Extra pool sizes to report, comma-separated (e.g. 200,300,400)")
    ] = "",
) -> None:
    """Dev only, retrieval only: pool recall of current vs rule vs learned first stage."""
    from reviewer.novelty.commands import run_first_stage_eval

    sizes = [int(x) for x in pool_sizes.split(",") if x.strip()]
    run_first_stage_eval(limit=limit, pool_sizes=sizes)


@judge_app.command("calibration")
def judge_calibration(
    force: Annotated[bool, typer.Option(help="Overwrite even if grades were entered")] = False,
    fresh: Annotated[
        int, typer.Option(help="Clean check: N new claims not in calibration_10.md")
    ] = 0,
) -> None:
    """Write reports/private/calibration_10.md: 10 claims for you to grade (no judge scores)."""
    from reviewer.eval.judge_commands import run_calibration_file

    run_calibration_file(force=force, fresh=fresh)


@judge_app.command("run")
def judge_run(
    ids: Annotated[list[str] | None, typer.Argument(help="Paper ids; default all claimed")] = None,
    force: Annotated[bool, typer.Option(help="Re-judge even if output exists")] = False,
    claims_dir: Annotated[
        Path | None, typer.Option(help="Claim set to judge (default data/claims/)")
    ] = None,
) -> None:
    """Judge every claim (one claude-opus-5-5 call per paper) into reports/private/judge/."""
    from reviewer.eval.judge_commands import run_judge

    run_judge(ids, force=force, claims_dir=claims_dir)


@judge_app.command("agreement")
def judge_agreement(
    file: Annotated[
        Path | None, typer.Option(help="Graded file (default reports/private/calibration_10.md)")
    ] = None,
) -> None:
    """Exact and within-1 agreement between your calibration grades and the judge."""
    from reviewer.eval.judge_commands import run_agreement

    run_agreement(path=file)


@judge_app.command("compare")
def judge_compare(
    a: Annotated[Path, typer.Argument(help="First claim set, e.g. data/claims_v4")],
    b: Annotated[Path, typer.Argument(help="Second claim set, e.g. data/claims_v5")],
) -> None:
    """Correctness by tier for two judged claim sets."""
    from reviewer.eval.judge_commands import run_compare

    run_compare(a, b)


@judge_app.command("spotcheck")
def judge_spotcheck(
    before: Annotated[Path, typer.Argument(help="Claim set verification started from")],
    after: Annotated[Path, typer.Argument(help="Verified claim set")],
    n: Annotated[int, typer.Option(help="Claims to sample")] = 5,
) -> None:
    """Hand spot-check file of claims verification fixed or dropped (no judge scores)."""
    from reviewer.eval.judge_commands import run_spotcheck

    run_spotcheck(before, after, n)


@judge_app.command("summary")
def judge_summary(
    claims_dir: Annotated[
        Path | None, typer.Option(help="Claim set that was judged (default data/claims/)")
    ] = None,
) -> None:
    """By-tier summary and the 10 lowest-scoring claims (reports/private/judge_summary_*.md)."""
    from reviewer.eval.judge_commands import run_summary

    run_summary(claims_dir=claims_dir)


if __name__ == "__main__":
    app()


@app.command()
def writing(
    path: Annotated[Path, typer.Argument(help="A ParsedPaper JSON or a directory of them")] = Path(
        "data/parsed"
    ),
    force: Annotated[bool, typer.Option(help="Re-check even if a report exists")] = False,
) -> None:
    """Writing check (one LLM call per paper, quotes verified in code) ->
    reports/writing/<id>.md (own papers: reports/private/writing/)."""
    from reviewer.writing.commands import run_writing

    run_writing(path, force=force)


@app.command("writing-summary")
def writing_summary(
    seed: Annotated[int, typer.Option(help="Seed for the precision sample")] = 7,
    n: Annotated[int, typer.Option(help="Findings in the precision sample")] = 10,
) -> None:
    """Counts per paper and check plus a precision sample -> reports/writing/SUMMARY.md."""
    from reviewer.writing.commands import run_summary

    run_summary(seed, n)


@methods_app.command("check")
def methods_check(
    ids: Annotated[list[str] | None, typer.Argument(help="Paper ids; default all parsed")] = None,
    force: Annotated[bool, typer.Option(help="Re-check even if output exists")] = False,
) -> None:
    """Checklist review of each paper's methodology (private: reports/private/methods/)."""
    from reviewer.methods.commands import run_check

    run_check(ids, force=force)


@methods_app.command("eval")
def methods_eval(
    precision_n: Annotated[int, typer.Option(help="Concerns in the precision sample")] = 20,
) -> None:
    """Recall against data/private/critiques/ and an audited precision sample."""
    from reviewer.methods.commands import run_eval

    run_eval(precision_n=precision_n)


@methods_app.command("report")
def methods_report() -> None:
    """Re-render reports/private/methods_report.md from existing reviews."""
    from reviewer.methods.commands import run_report

    run_report()


@methods_app.command("review-eval")
def methods_review_eval(
    cases: Annotated[
        Path | None, typer.Option(help="Dev concerns file (default METHODS_EVAL_DIR/dev.jsonl)")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Seeded sample of N dev submissions")] = None,
    max_cost: Annotated[float, typer.Option(help="Stop starting papers past this spend")] = 5.0,
    fetch: Annotated[bool, typer.Option(help="Download + parse missing PDFs")] = True,
    dry_run: Annotated[bool, typer.Option(help="Only count and estimate cost")] = False,
    precision_n: Annotated[int, typer.Option(help="Findings in the precision sample")] = 20,
    seed: Annotated[int, typer.Option(help="Sample seed")] = 20260926,
    exclude_sample: Annotated[
        int | None, typer.Option(help="Exclude the default-seed sample of N (the tuning set)")
    ] = None,
    tag: Annotated[str, typer.Option(help="Output name suffix, e.g. B_methods_v2")] = "",
    prompt: Annotated[
        str | None, typer.Option(help="Checker prompt (default METHODS_PROMPT)")
    ] = None,
) -> None:
    """Checker vs OpenReview reviewers' methodology concerns. Dev only; test is locked."""
    from reviewer.config import get_settings
    from reviewer.methods.review_commands import run_review_eval

    settings = get_settings()
    if prompt:
        settings = settings.model_copy(update={"methods_prompt": prompt})
    run_review_eval(
        "dev",
        cases,
        limit=limit,
        max_cost=max_cost,
        fetch=fetch,
        dry_run=dry_run,
        precision_n=precision_n,
        settings=settings,
        seed=seed,
        exclude_sample=exclude_sample,
        tag=tag,
    )


@methods_app.command("review-compare")
def methods_review_compare(
    pairs: Annotated[
        list[str], typer.Argument(help="BASE:NEW tag pairs, e.g. A_methods_v1:A_methods_v2")
    ],
) -> None:
    """Compare checker versions on the same submissions (paired) from review-eval outputs."""
    from reviewer.methods.review_commands import run_review_compare

    run_review_compare(pairs)


@methods_app.command("stability")
def methods_stability(
    condition: Annotated[str, typer.Argument(help="current | structured")],
    ids: Annotated[list[str], typer.Option("--id", help="OpenReview submission ids (parsed)")],
    replicates: Annotated[int, typer.Option(help="Runs per paper")] = 2,
    max_cost: Annotated[float, typer.Option(help="Stop starting calls past this spend")] = 3.0,
) -> None:
    """Verdict flip rate: check each paper `replicates` times under one condition."""
    from reviewer.methods.stability import run_stability

    run_stability(condition, ids, replicates, max_cost)


@meta_app.command("review")
def meta_review(
    ids: Annotated[list[str], typer.Argument(help="Parsed paper ids")],
    venue: Annotated[str, typer.Option(help="Target venue (sets the rating scale)")] = "ICLR",
    venue_year: Annotated[int | None, typer.Option(help="Venue year")] = None,
    lite: Annotated[bool, typer.Option(help="Novelty on top-2 claims, meta k=1")] = True,
    force: Annotated[bool, typer.Option(help="Re-run checkers and the review")] = False,
) -> None:
    """Run every checker (if missing) then the meta-reviewer, for one or more papers."""
    from reviewer.meta.commands import run_batch

    run_batch(ids, venue=venue, venue_year=venue_year, lite=lite, force=force)


@meta_app.command("dev-fetch")
def meta_dev_fetch() -> None:
    """Sample the 60-paper ICLR dev set, download each PDF from OpenReview and parse it."""
    from reviewer.meta.dev_commands import run_dev_fetch

    run_dev_fetch()


@meta_app.command("dev-run")
def meta_dev_run(
    rejudge_from: Annotated[
        str | None,
        typer.Option(help="Judge only papers reviewed under this earlier meta prompt"),
    ] = None,
) -> None:
    """Every checker (lite) + meta judgment + baselines + probe on the dev set, then the 5-fold
    CV calibration and the report (spend capped at META_BUDGET_USD)."""
    from reviewer.meta.dev_commands import run_dev_run

    run_dev_run(rejudge_from=rejudge_from)


@app.command("review-caps")
def review_caps() -> None:
    """Recompute per-step max_tokens from the dev LLM cache (2x the largest observed output)."""
    from reviewer.config import get_settings
    from reviewer.review.caps import OBSERVED_FILE, load_caps, observed_max, write_observed

    settings = get_settings()
    write_observed(observed_max(settings.paper_cache / "llm", settings))
    for step, cap in load_caps().items():
        typer.echo(f"{step.value}\t{cap}")
    typer.echo(f"wrote {OBSERVED_FILE}")


@app.command("review")
def review(
    pdf: Annotated[Path, typer.Argument(help="The paper PDF")],
    out: Annotated[Path, typer.Option("--out", help="Where to write review.json")],
    work_dir: Annotated[
        Path | None, typer.Option(help="Per-paper work dir (default: next to --out)")
    ] = None,
    venue: Annotated[str, typer.Option(help="Target venue for the meta-review scale")] = "ICLR",
    venue_year: Annotated[int | None, typer.Option(help="Venue year")] = None,
    limit_usd: Annotated[float, typer.Option(help="Hard spend limit for this review")] = 6.0,
) -> None:
    """Run every checker on one PDF with the web profile (task 1.9); writes checker outputs
    and per-step status/cost to --out. Findings and anchors are added in W2-W3."""
    from reviewer.review.job import review_pdf

    review_pdf(pdf, out, work_dir, venue, venue_year, limit_usd)
