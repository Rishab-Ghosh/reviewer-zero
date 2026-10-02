"""Verdict stability: run the checker k times on the same papers and count verdict flips.

Conditions:
  current     methods_v2 + repair_v1, free yes/no/unclear/na verdicts. Replicate 0 is the
              ordinary cache entry (served from cache when the paper was already checked).
  structured  methods_v2s: verdicts derived in code from structured fields.
(Temperature 0 is not a condition: claude-opus-5-5 rejects `temperature` with a 400.)

Metrics compare replicate 0 with replicate 1 on papers both have. Pure except run_stability.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import typer

from reviewer.config import Settings, get_settings
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient, LLMError
from reviewer.methods.checker import MethodsError, check_paper, load_methods_prompt
from reviewer.methods.commands import REPAIR_PROMPT, methods_client
from reviewer.methods.structured import check_paper_structured
from reviewer.models import MethodsVerdict, PaperMethodsReview, ParsedPaper
from reviewer.privacy import write_private

logger = logging.getLogger(__name__)

CONDITIONS = {"current": "methods_v2", "structured": "methods_v2s"}


@dataclass(frozen=True)
class Flips:
    n: int
    flips: int

    @property
    def rate(self) -> float:
        return self.flips / self.n if self.n else 0.0

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson interval (treats verdicts as independent; they cluster by paper)."""
        if not self.n:
            return (0.0, 0.0)
        p, n = self.rate, self.n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return (max(0.0, centre - half), min(1.0, centre + half))


Pairs = list[tuple[PaperMethodsReview, PaperMethodsReview]]


def _pairs(a: dict[str, PaperMethodsReview], b: dict[str, PaperMethodsReview]) -> Pairs:
    return [(a[k], b[k]) for k in sorted(a.keys() & b.keys())]


def verdict_flips(pairs: Pairs, attr: str = "verdict") -> Flips:
    n = flips = 0
    for r1, r2 in pairs:
        v2 = {f.item: getattr(f, attr) for f in r2.findings}
        for f in r1.findings:
            n += 1
            flips += getattr(f, attr) != v2.get(f.item)
    return Flips(n, flips)


def strict_flips(pairs: Pairs) -> Flips:
    """Flips of the finding the product surfaces: is the item a `no` or not."""
    n = flips = 0
    for r1, r2 in pairs:
        v2 = {f.item: f.verdict == MethodsVerdict.no for f in r2.findings}
        for f in r1.findings:
            n += 1
            flips += (f.verdict == MethodsVerdict.no) != v2.get(f.item)
    return Flips(n, flips)


def code_agreement(pairs: Pairs) -> Flips:
    """Structured only: items that are `no` in either run whose verified violation-code sets
    differ (counted as flips)."""
    n = flips = 0
    for r1, r2 in pairs:
        c2 = {f.item: (f.verdict, set(f.violation_codes)) for f in r2.findings}
        for f in r1.findings:
            v2, codes2 = c2[f.item]
            if MethodsVerdict.no not in (f.verdict, v2):
                continue
            n += 1
            flips += set(f.violation_codes) != codes2
    return Flips(n, flips)


def other_concern_delta(pairs: Pairs) -> float:
    diffs = [
        abs(sum(c.verified for c in a.other_concerns) - sum(c.verified for c in b.other_concerns))
        for a, b in pairs
    ]
    return sum(diffs) / len(diffs) if diffs else 0.0


def _fmt(f: Flips) -> str:
    lo, hi = f.wilson()
    return f"{f.flips}/{f.n} = **{f.rate:.0%}** [{lo:.0%}, {hi:.0%}]"


def render_stability(runs: dict[str, dict[int, dict[str, PaperMethodsReview]]]) -> str:
    lines = [
        "# Checker verdict stability (private)",
        "",
        "Each paper checked twice per condition (replicate 0 vs 1). Flip = the item's verdict "
        "differs between the two runs. 95% Wilson intervals (verdicts cluster by paper, so "
        "they are optimistic). Temperature 0 is not testable: claude-opus-5-5 returns 400 "
        "'`temperature` is deprecated for this model'.",
        "",
        "| condition | papers | verdict flips (final) | `no` vs not-`no` flips | verdict the "
        "model's output implied, flips | cost/paper/run | Δ verified other concerns |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    extra = []
    for cond, reps in runs.items():
        pairs = _pairs(reps.get(0, {}), reps.get(1, {}))
        if not pairs:
            continue
        cost = [r.token_usage.cost_usd for r in reps.get(1, {}).values()]
        lines.append(
            f"| {cond} | {len(pairs)} | {_fmt(verdict_flips(pairs))} | {_fmt(strict_flips(pairs))} "
            f"| {_fmt(verdict_flips(pairs, 'model_verdict'))} "
            f"| ${sum(cost) / len(cost):.3f} | {other_concern_delta(pairs):.1f} |"
        )
        if cond == "structured":
            extra.append(
                f"- structured: among items that are `no` in either run, the verified "
                f"violation-code sets differ in {_fmt(code_agreement(pairs))}."
            )
        per_item: dict[str, list[int]] = {}
        for r1, r2 in pairs:
            v2 = {f.item: f.verdict for f in r2.findings}
            for f in r1.findings:
                per_item.setdefault(f.item.value, []).append(int(f.verdict != v2[f.item]))
        worst = sorted(per_item.items(), key=lambda kv: -sum(kv[1]))[:4]
        extra.append(
            f"- {cond}: most unstable items: "
            + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in worst)
        )
    return "\n".join([*lines, "", *extra]) + "\n"


def stability_dir(settings: Settings, condition: str, replicate: int) -> Path:
    return settings.private_reports_dir / "methods_stability" / condition / f"rep{replicate}"


def load_runs(settings: Settings) -> dict[str, dict[int, dict[str, PaperMethodsReview]]]:
    out: dict[str, dict[int, dict[str, PaperMethodsReview]]] = {}
    for cond in CONDITIONS:
        for rep in (0, 1):
            d = stability_dir(settings, cond, rep)
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                r = PaperMethodsReview.model_validate_json(p.read_text(encoding="utf-8"))
                out.setdefault(cond, {}).setdefault(rep, {})[r.paper_id] = r
    return out


def run_stability(
    condition: str,
    ids: list[str],
    replicates: int = 2,
    max_cost: float = 3.0,
    settings: Settings | None = None,
    client: LLMClient | None = None,
) -> Path:
    settings = settings or get_settings()
    if condition not in CONDITIONS:
        raise typer.BadParameter(f"condition must be one of {sorted(CONDITIONS)}")
    client = client or methods_client(settings)
    prompt = load_methods_prompt(CONDITIONS[condition])
    repair = load_methods_prompt(REPAIR_PROMPT) if condition == "current" else None
    for rep in range(replicates):
        for sid in ids:
            pid = sid if sid.startswith("or-") else f"or-{sid}"
            out = stability_dir(settings, condition, rep) / f"{pid}.json"
            if out.exists():
                continue
            if client.totals.fresh.cost_usd >= max_cost:
                spent = client.totals.fresh.cost_usd
                logger.warning("stopped at $%.2f (--max-cost %.2f)", spent, max_cost)
                break
            paper = ParsedPaper.model_validate_json(
                (settings.openreview_parsed_dir / f"{pid}.json").read_text(encoding="utf-8")
            )
            kw = dict(
                tier="openreview",
                prompt=prompt,
                model=settings.methods_model,
                effort=settings.methods_effort,
                max_retries=settings.llm_max_retries,
                replicate=rep,
            )
            try:
                review = (
                    check_paper(client, paper, repair_prompt=repair, **kw)
                    if condition == "current"
                    else check_paper_structured(client, paper, **kw)
                )
            except (MethodsError, LLMError) as e:
                record_failure(settings.failures_file, "methods", pid, f"stability: {e}")
                continue
            write_private(out, review.model_dump_json(indent=2) + "\n")
    report = settings.private_reports_dir / "methods_stability.md"
    write_private(report, render_stability(load_runs(settings)))
    typer.echo(f"{condition}: new spend ${client.totals.fresh.cost_usd:.2f}; wrote {report}")
    return report
