"""`reviewer judge ...`: claim-grading judge, human calibration, agreement, summary.

All outputs are private (reports/private/): they cover own and lab papers."""

import json
import logging
import random
from pathlib import Path
from statistics import mean

import typer

from reviewer.config import Settings, get_settings
from reviewer.corpus.corpus import load_corpus
from reviewer.eval.judge import (
    CALIBRATION_SEED,
    LOW_SCORE,
    ClaimRef,
    JudgeError,
    agreement,
    best_excerpt,
    calibration_sample,
    fresh_sample,
    judge_paper,
    load_judge_prompt,
    parse_calibration,
    render_calibration,
)
from reviewer.eval.loader import load_all_gold
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError, MissingAPIKeyError
from reviewer.models import PaperClaims, PaperJudgment, ParsedPaper
from reviewer.privacy import write_private

logger = logging.getLogger(__name__)

TIER_ORDER = ("own", "lab", "ordinary", "famous")


def _tiers(settings: Settings) -> dict[str, str]:
    return {e.paper_id: e.tier.value for e in load_corpus(settings)}


def judge_out_dir(settings: Settings, claims_dir: Path | None) -> Path:
    """judge/<prompt>/ for the default claims; judge/<prompt>__<dir name>/ for another set."""
    if claims_dir is None or claims_dir.resolve() == settings.claims_dir.resolve():
        return settings.judge_dir
    return settings.judge_dir.with_name(f"{settings.judge_prompt}__{claims_dir.name}")


def _claims(settings: Settings, paper_id: str, claims_dir: Path | None = None) -> PaperClaims:
    path = (claims_dir or settings.claims_dir) / f"{paper_id}.json"
    return PaperClaims.model_validate_json(path.read_text(encoding="utf-8"))


def _parsed(settings: Settings, paper_id: str) -> ParsedPaper:
    path = settings.parsed_dir / f"{paper_id}.json"
    return ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))


def _claimed_ids(
    settings: Settings, tiers: dict[str, str], claims_dir: Path | None = None
) -> list[str]:
    return sorted(
        p.stem
        for p in (claims_dir or settings.claims_dir).glob("*.json")
        if not p.name.endswith(".pass1.json") and p.stem in tiers
    )


# --------------------------------------------------------------------------- calibration


def run_calibration_file(
    settings: Settings | None = None, force: bool = False, fresh: int = 0
) -> None:
    """Write reports/private/calibration_10.md, or with `fresh` N a clean-check file of N new
    claims not in it (calibration_fresh_N.md). Never overwrites grades already entered."""
    settings = settings or get_settings()
    out = (
        settings.private_reports_dir / f"calibration_fresh_{fresh}.md"
        if fresh
        else settings.calibration_file
    )
    if out.exists() and not force:
        filled = any(
            v is not None for s in parse_calibration(out.read_text()).values() for v in s.values()
        )
        if filled:
            logger.error("%s already has grades; not overwriting (use --force)", out)
            raise typer.Exit(code=1)
    tiers = _tiers(settings)
    refs = [
        ClaimRef(pid, i, tiers[pid])
        for pid in _claimed_ids(settings, tiers)
        for i in range(len(_claims(settings, pid).claims))
    ]
    if fresh:
        if not settings.calibration_file.exists():
            logger.error(
                "no %s to exclude; run the 10-claim calibration first", settings.calibration_file
            )
            raise typer.Exit(code=1)
        done = set(parse_calibration(settings.calibration_file.read_text(encoding="utf-8")))
        picked = fresh_sample(refs, done, fresh)
        title = f"Claim grading clean check ({len(picked)} fresh claims)"
    else:
        picked = calibration_sample(refs)
        title = "Claim grading calibration (10 claims)"
    items = []
    for ref in picked:
        claim = _claims(settings, ref.paper_id).claims[ref.claim_index]
        items.append((ref, claim, best_excerpt(_parsed(settings, ref.paper_id), claim)))
    write_private(out, render_calibration(items, title))
    typer.echo(f"wrote {out} ({len(items)} claims)")


# --------------------------------------------------------------------------- judging


def run_judge(
    ids: list[str] | None = None,
    force: bool = False,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    claims_dir: Path | None = None,
) -> list[PaperJudgment]:
    """Judge every claimed corpus paper (or `ids`); one Opus call per paper, cached.
    `claims_dir` judges another claim set (default data/claims/)."""
    settings = settings or get_settings()
    tiers = _tiers(settings)
    paper_ids = ids or _claimed_ids(settings, tiers, claims_dir)
    out_dir = judge_out_dir(settings, claims_dir)
    client = client or LLMClient(settings)
    prompt = load_judge_prompt(settings.judge_prompt)
    gold, _ = load_all_gold(settings.gold_dir, settings.private_gold_dir, tiers)
    results: list[PaperJudgment] = []
    for pid in paper_ids:
        out = out_dir / f"{pid}.json"
        if out.exists() and not force:
            results.append(PaperJudgment.model_validate_json(out.read_text(encoding="utf-8")))
            continue
        try:
            result = judge_paper(
                client,
                _parsed(settings, pid),
                _claims(settings, pid, claims_dir),
                gold.get(pid),
                tier=tiers[pid],
                prompt=prompt,
                model=settings.judge_model,
                max_retries=settings.llm_max_retries,
            )
        except MissingAPIKeyError as e:
            logger.error("%s", e)
            raise typer.Exit(code=1) from e
        except (JudgeError, LLMError, OSError, ValueError) as e:
            record_failure(settings.failures_file, "judge", pid, f"{type(e).__name__}: {e}")
            continue
        write_private(out, result.model_dump_json(indent=2) + "\n")
        results.append(result)
    logger.info(
        "judge: %d/%d papers; new spend $%.2f (%d calls, %d cached)",
        len(results),
        len(paper_ids),
        client.totals.fresh.cost_usd,
        client.totals.n_fresh,
        client.totals.n_cached,
    )
    return results


def _judge_scores(results: list[PaperJudgment]) -> dict[str, dict[str, int]]:
    return {
        f"{r.paper_id}#{j.claim_index + 1}": {
            "specificity": j.specificity,
            "correctness": j.correctness,
        }
        for r in results
        for j in r.judgments
    }


# --------------------------------------------------------------------------- agreement


def run_agreement(
    settings: Settings | None = None, client: LLMClient | None = None, path: Path | None = None
) -> dict:
    settings = settings or get_settings()
    path = path or settings.calibration_file
    if not path.exists():
        logger.error("no %s; run `reviewer judge calibration` first", path)
        raise typer.Exit(code=1)
    human = parse_calibration(path.read_text(encoding="utf-8"))
    graded = {cid: s for cid, s in human.items() if any(v is not None for v in s.values())}
    if not graded:
        logger.error("no grades entered in %s yet", path)
        raise typer.Exit(code=1)
    papers = sorted({cid.rsplit("#", 1)[0] for cid in graded})
    judged = run_judge(papers, settings=settings, client=client)
    result = agreement(graded, _judge_scores(judged))
    reasons = {f"{r.paper_id}#{j.claim_index + 1}": j for r in judged for j in r.judgments}
    lines = [f"# Judge vs human agreement ({path.name}, {settings.judge_prompt})", ""]
    lines += ["| dimension | n | exact | within 1 |", "|---|---|---|---|"]
    for dim, row in result.items():
        lines.append(f"| {dim} | {row['n']} | {row['exact']:.0%} | {row['within_1']:.0%} |")
    lines += ["", "## Differences of 2 or more", ""]
    diffs = [(dim, d) for dim, row in result.items() for d in row["differ_2plus"]]
    if not diffs:
        lines.append("None.")
    for dim, (cid, h, j) in diffs:
        reason = getattr(reasons[cid], f"{dim}_reason") or "(no reason: judge scored 4-5)"
        lines.append(f"- **{cid}** {dim}: you {h}, judge {j}. Judge: {reason}")
    write_private(
        settings.private_reports_dir / f"judge_agreement_{path.stem}_{settings.judge_prompt}.md",
        "\n".join(lines) + "\n",
    )
    for dim, row in result.items():
        typer.echo(
            f"{dim}: n={row['n']}, exact {row['exact']:.0%}, within 1 {row['within_1']:.0%}, "
            f"{len(row['differ_2plus'])} differ by 2+"
        )
    return result


# --------------------------------------------------------------------------- summary


def run_summary(settings: Settings | None = None, claims_dir: Path | None = None) -> None:
    """By-tier summary of existing judgments plus the 10 lowest-scoring claims."""
    settings = settings or get_settings()
    out_dir = judge_out_dir(settings, claims_dir)
    results = [
        PaperJudgment.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(out_dir.glob("*.json"))
    ]
    if not results:
        logger.error("no judgments in %s; run `reviewer judge run`", out_dir)
        raise typer.Exit(code=1)
    rows = [(r, j) for r in results for j in r.judgments]
    lines = [
        "# Claim judge summary (private)",
        "",
        f"{len(rows)} claims in {len(results)} papers; judge {results[0].model}, "
        f"{results[0].prompt_version}. Scores 1-5.",
        "",
        "| tier | papers | claims | specificity mean | correctness mean | spec ≤3 | corr ≤3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier in TIER_ORDER:
        tr = [(r, j) for r, j in rows if r.tier == tier]
        if not tr:
            continue
        spec = [j.specificity for _, j in tr]
        corr = [j.correctness for _, j in tr]
        lines.append(
            f"| {tier} | {len({r.paper_id for r, _ in tr})} | {len(tr)} | {mean(spec):.2f} "
            f"| {mean(corr):.2f} | {sum(s <= LOW_SCORE for s in spec)} "
            f"| {sum(c <= LOW_SCORE for c in corr)} |"
        )
    lowest = sorted(
        rows,
        key=lambda rj: (
            rj[1].specificity + rj[1].correctness,
            min(rj[1].specificity, rj[1].correctness),
            rj[0].paper_id,
            rj[1].claim_index,
        ),
    )[:10]
    lines += ["", "## 10 lowest-scoring claims", ""]
    for r, j in lowest:
        claim = _claims(settings, r.paper_id, claims_dir).claims[j.claim_index]
        lines += [
            f"- **{r.paper_id}#{j.claim_index + 1}** ({r.tier}): specificity {j.specificity}, "
            f"correctness {j.correctness}",
            f"  - Claim: {claim.statement}",
        ]
        if j.specificity_reason:
            lines.append(f"  - Specificity: {j.specificity_reason}")
        if j.correctness_reason:
            lines.append(f"  - Correctness: {j.correctness_reason}")
    write_private(
        settings.private_reports_dir / f"judge_summary_{out_dir.name}.md",
        "\n".join(lines) + "\n",
    )
    typer.echo(json.dumps({"claims": len(rows), "papers": len(results)}))


# --------------------------------------------------------------------------- compare, spot check


def _tier_stats(results: list[PaperJudgment]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for tier in (*TIER_ORDER, "all"):
        js = [j for r in results if tier in ("all", r.tier) for j in r.judgments]
        if not js:
            continue
        out[tier] = {
            "claims": len(js),
            "correctness_mean": mean(j.correctness for j in js),
            "correctness_le3": sum(j.correctness <= LOW_SCORE for j in js),
            "specificity_mean": mean(j.specificity for j in js),
        }
    return out


def _judged(settings: Settings, claims_dir: Path) -> list[PaperJudgment]:
    return [
        PaperJudgment.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(judge_out_dir(settings, claims_dir).glob("*.json"))
    ]


def _pair(x: dict[str, float], y: dict[str, float], key: str, fmt: str) -> str:
    return f"{format(x.get(key, 0), fmt)} → {format(y.get(key, 0), fmt)}"


def run_compare(a: Path, b: Path, settings: Settings | None = None) -> dict[str, dict]:
    """Correctness by tier for two judged claim sets (same judge prompt)."""
    settings = settings or get_settings()
    sa, sb = _tier_stats(_judged(settings, a)), _tier_stats(_judged(settings, b))
    lines = [
        f"# Claim correctness: {a.name} vs {b.name} ({settings.judge_prompt})",
        "",
        "| tier | claims | correctness mean | correctness ≤3 | specificity mean |",
        "|---|---|---|---|---|",
    ]
    for tier in [t for t in (*TIER_ORDER, "all") if t in sa or t in sb]:
        x, y = sa.get(tier, {}), sb.get(tier, {})
        lines.append(
            f"| {tier} | {_pair(x, y, 'claims', '.0f')} | {_pair(x, y, 'correctness_mean', '.2f')} "
            f"| {_pair(x, y, 'correctness_le3', '.0f')} "
            f"| {_pair(x, y, 'specificity_mean', '.2f')} |"
        )
    write_private(
        settings.private_reports_dir / f"judge_compare_{a.name}_vs_{b.name}.md",
        "\n".join(lines) + "\n",
    )
    typer.echo("\n".join(lines))
    return {"a": sa, "b": sb}


def run_spotcheck(before: Path, after: Path, n: int = 5, settings: Settings | None = None) -> Path:
    """A hand spot-check file of `n` seeded claims that verification fixed or dropped: the claim
    before, what verification did and why, and its verified quotes. Judge scores are hidden."""
    settings = settings or get_settings()
    tiers = _tiers(settings)
    changed: list[tuple[str, int, str]] = []
    for pid in _claimed_ids(settings, tiers, after):
        meta = _claims(settings, pid, after).extraction_metadata
        for rec in meta.verification or []:
            if rec.outcome in ("fixed", "dropped"):
                changed.append((pid, rec.input_index, rec.outcome))
    picked = random.Random(CALIBRATION_SEED).sample(sorted(changed), min(n, len(changed)))
    lines = [
        f"# Spot check: {len(picked)} claims changed by verification ({after.name})",
        "",
        f"Seeded sample of {len(changed)} fixed or dropped claims. For each, say whether the "
        "change is right (the old claim really was wrong or unsupported, and a fix matches the "
        "paper). Judge scores are not in this file.",
    ]
    for k, (pid, idx, outcome) in enumerate(picked, start=1):
        after_claims = _claims(settings, pid, after)
        rec = next(
            r for r in after_claims.extraction_metadata.verification or [] if r.input_index == idx
        )
        before_claim = _critic_final_claim(settings, pid, idx, before, after)
        lines += [
            "",
            f"## S{k} · {tiers[pid]} · {pid} · critic claim {idx + 1} · **{outcome}**",
            "",
            f"**Before:** {before_claim}",
            "",
        ]
        if outcome == "fixed":
            kept_pos = [
                r.input_index
                for r in after_claims.extraction_metadata.verification or []
                if r.outcome != "dropped"
            ].index(idx)
            lines += [f"**After:** {after_claims.claims[kept_pos].statement}", ""]
        lines += [f"**Verifier's reason:** {rec.reason}", ""]
        if rec.problems:
            lines += [
                "**Code-check problems (why unverifiable):**",
                *[f"- {p}" for p in rec.problems],
                "",
            ]
        if rec.items:
            lines += [
                "**Verified quotes:**",
                *[f'- {it.element} ({it.source}): "{it.quote}"' for it in rec.items],
                "",
            ]
        lines += ["- change is right (y/n): ", "- notes: "]
    out = settings.private_reports_dir / f"spotcheck_{after.name}_{len(picked)}.md"
    if out.exists():
        raise typer.BadParameter(f"{out} exists; not overwriting a spot-check file")
    write_private(out, "\n".join(lines) + "\n")
    typer.echo(f"wrote {out}")
    return out


def _critic_final_claim(settings: Settings, pid: str, idx: int, before: Path, after: Path) -> str:
    """The critic's claim `idx` that verification saw: the before-set's claim when that set came
    from the same critic output (same prompts up to the critic), else a note."""
    b = _claims(settings, pid, before)
    a_pv = _claims(settings, pid, after).extraction_metadata.prompt_version
    if a_pv.startswith(b.extraction_metadata.prompt_version) and idx < len(b.claims):
        return b.claims[idx].statement
    return "(critic output not stored separately; see the verification record)"
