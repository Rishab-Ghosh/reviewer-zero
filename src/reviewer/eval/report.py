"""Render reports/week1_claims.md. Pure: takes records, returns markdown."""

import re
from datetime import datetime

from reviewer.models import Claim, Failure, Flag, GoldClaim, PaperTier

from .checks import (
    DEFAULT_DUPLICATE_THRESHOLD,
    MAX_STATEMENT_WORDS,
    VAGUE_PHRASES,
    similarity_method,
)
from .gold_compare import NUMBER_RULE
from .schemas import GoldComparison, PaperRecord, SimilarityHit

GRADES_PATH = "reports/private/grades.yaml"
PRIVATE_REPORT_PATH = "reports/private/week1_claims.md"

TITLE_WIDTH = 60
REASON_WIDTH = 50
BLANK = "___"


# --------------------------------------------------------------------------- helpers


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _cell(text: str) -> str:
    """Make text safe for a single markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _quote(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _readable(pattern: str) -> str:
    """A vague-phrase regex as plain words, for the report header."""
    text = re.sub(r"\(\?:[^)]*\)\?", "", pattern).replace(r"\s+", " ").replace("?", "")
    return " ".join(text.split())


def _paper_link(paper_id: str, tier: PaperTier) -> str:
    if tier == PaperTier.own or paper_id.startswith("own-"):
        return f"`{paper_id}`"
    return f"[arXiv:{paper_id}](https://arxiv.org/abs/{paper_id})"


def _failure(record: PaperRecord, *stages: str) -> Failure | None:
    matches = [f for f in record.failures if f.stage in stages]
    return matches[-1] if matches else None


def parse_status(record: PaperRecord) -> tuple[bool, str]:
    """(ok, short human-readable reason) for the parse stage."""
    if record.parsed is not None:
        return True, ""
    for stage in ("parse", "fetch"):
        f = _failure(record, stage)
        if f is not None:
            return False, f"{stage} failed: {f.error}"
    parse_errors = [e for e in record.load_errors if e.startswith("parsed/")]
    if parse_errors:
        return False, parse_errors[0]
    return False, "not processed"


def extract_status(record: PaperRecord) -> tuple[bool, str]:
    if record.claims is not None:
        return True, ""
    f = _failure(record, "extract")
    if f is not None:
        return False, f"extract failed: {f.error}"
    claims_errors = [e for e in record.load_errors if e.startswith("claims/")]
    if claims_errors:
        return False, claims_errors[0]
    if record.parsed is None:
        return False, "not run (no parsed paper)"
    return False, "not processed"


def _tokens(record: PaperRecord) -> tuple[int, int, float] | None:
    if record.claims is None:
        return None
    u = record.claims.extraction_metadata.token_usage
    tokens_in = u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
    return tokens_in, u.output_tokens, u.cost_usd


# --------------------------------------------------------------------------- sections


def _recall_all_cell(comparison: GoldComparison | None) -> str:
    if comparison is None or comparison.recall_all is None:
        return "—"
    return f"{comparison.n_found_all}/{comparison.n_numbers_all} ({comparison.recall_all:.0%})"


def _recall_cell(comparison: GoldComparison | None) -> str:
    if comparison is None or comparison.recall is None:
        return "—"
    return f"{comparison.n_found}/{comparison.n_numbers} ({comparison.recall:.0%})"


def _gold_label(record: PaperRecord) -> str:
    if record.gold is None:
        return "no"
    label = record.gold.source.value
    return f"{label} (private)" if record.gold_private else label


def render_public_note() -> str:
    return (
        "> **Public version.** Own papers and private gold (gold from `data/private/gold/`, or "
        "with source `author` or `lab_summary`) are omitted: papers with private gold are shown "
        "as having no gold. The full grading report is "
        f"`{PRIVATE_REPORT_PATH}` and the grades file is `{GRADES_PATH}` (both gitignored). "
        "Similarity scores here use an IDF over the public papers' claims only, so they can "
        "differ slightly from the private report."
    )


def render_header(
    generated_at: datetime,
    n_papers: int,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    public: bool = False,
) -> str:
    phrases = ", ".join(f"`{_readable(p)}`" for p in VAGUE_PHRASES)
    note = f"\n{render_public_note()}\n" if public else ""
    return f"""# Week 1 — extracted claims for grading

Generated {generated_at.isoformat(timespec="seconds")} · {n_papers} papers.
{note}
## How to grade

Enter scores in `{GRADES_PATH}` (one entry per extracted claim, same order as this report;
`claim_index` there is 0-based, "Claim 1" here is `claim_index: 0`). The grading lines under each
paper are a visual aid only. Re-running `reviewer eval` keeps every grade you have entered.

- **specificity (1-5)** — could a novelty search built on this claim work?
  1 = could describe a hundred papers; 3 = names the method or the setting but not both, or omits
  reported numbers; 5 = names method, setting and headline number, falsifiable and searchable.
- **correctness (1-5)** — does the paper actually make this claim, as scoped?
  1 = not in the paper / wrong; 3 = roughly, but misstates scope, numbers, or novelty;
  5 = faithful to the paper.
- **matches_gold** — only for papers with a gold file: 0-based index of the gold claim this
  extracted claim matches, `-1` if it matches none (leave `null` until graded).
- Then run `reviewer eval summarize`.

## Automated checks (hints, not gates)

- `vague_phrase` — statement contains a stock phrase ({phrases}); "state of the art"/SOTA only
  when the statement has no number.
- `missing_numbers` — no number in the statement or result, although the abstract reports
  numbers. Numbers are percentages, decimals, multipliers (3x), scaled counts (175B), numbers
  followed by a unit/metric word, and integers ≥ 2 that are not years (1900–2099), not part of a
  model name (GPT-4, T5, Llama-3), and not a reference (Table 2).
- `too_long` — statement longer than {MAX_STATEMENT_WORDS} words.
- `near_duplicate` — two claims of the same paper have TF-IDF cosine ≥ {threshold:.2f}. Every
  claim also shows its highest similarity to another claim of the same paper ("max
  similarity"), flagged or not.

**Similarity method:** {similarity_method(threshold)}

## Gold comparison

For every paper with a gold file, each gold claim is shown next to its closest extracted claim
(highest TF-IDF cosine between the two statements, same IDF as above) with that score, and each
number in the gold `result` field is marked ✓ (found) or ✗ (not found) in the extracted claims.
The gold source is shown in the summary table and in each gold block heading
(`paper_contributions` = copied from the paper; `author` = written by the paper's author;
`lab_summary` = copied from the lab's literature review).

**Number matching rule:** {NUMBER_RULE}
"""


def render_summary_table(records: list[PaperRecord]) -> str:
    lines = [
        "## Summary",
        "",
        "| # | paper_id | tier | title | parse ok | #claims | contributions stmt | #flags "
        "| gold | number recall (result) | recall (result+setting) | tokens in | tokens out "
        "| cost USD |",
        "|---:|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    tot_claims = tot_flags = tot_in = tot_out = n_gold = 0
    tot_numbers = tot_found = tot_all = tot_found_all = 0
    tot_cost = 0.0
    for n, r in enumerate(records, start=1):
        ok, reason = parse_status(r)
        parse_cell = "✓" if ok else f"✗ {_cell(_truncate(reason, REASON_WIDTH))}"
        if r.claims is not None:
            n_claims = str(len(r.claims.claims))
            tot_claims += len(r.claims.claims)
        else:
            _, why = extract_status(r)
            n_claims = f"— ({_cell(_truncate(why, 30))})" if ok else "—"
        contrib = "—" if r.parsed is None else ("yes" if r.parsed.contributions else "no")
        tok = _tokens(r)
        if tok is not None:
            tot_in, tot_out, tot_cost = tot_in + tok[0], tot_out + tok[1], tot_cost + tok[2]
            tok_cells = f"{tok[0]:,} | {tok[1]:,} | {tok[2]:.4f}"
        else:
            tok_cells = "— | — | —"
        n_gold += r.gold is not None
        tot_flags += len(r.flags)
        if r.gold is not None and r.gold_comparison is not None:
            tot_numbers += r.gold_comparison.n_numbers
            tot_found += r.gold_comparison.n_found
            tot_all += r.gold_comparison.n_numbers_all
            tot_found_all += r.gold_comparison.n_found_all
        recall = _recall_cell(r.gold_comparison if r.gold is not None else None)
        recall_all = _recall_all_cell(r.gold_comparison if r.gold is not None else None)
        lines.append(
            f"| {n} | `{r.entry.paper_id}` | {r.entry.tier.value} "
            f"| {_cell(_truncate(r.title, TITLE_WIDTH))} | {parse_cell} | {n_claims} | {contrib} "
            f"| {len(r.flags)} | {_gold_label(r)} | {recall} | {recall_all} | {tok_cells} |"
        )
    n_parsed = sum(r.parsed is not None for r in records)
    micro = f"{tot_found}/{tot_numbers} ({tot_found / tot_numbers:.0%})" if tot_numbers else "—"
    micro_all = f"{tot_found_all}/{tot_all} ({tot_found_all / tot_all:.0%})" if tot_all else "—"
    lines.append(
        f"| | **total** | | {len(records)} papers | {n_parsed}/{len(records)} "
        f"| {tot_claims} | | {tot_flags} | {n_gold} | {micro} | {micro_all} | {tot_in:,} "
        f"| {tot_out:,} "
        f"| {tot_cost:.4f} |"
    )
    lines.append("")
    lines.append(
        "_tokens in = input + cache-read + cache-write tokens over both passes. "
        "number recall (result) = numbers in gold `result` fields found in the extracted claims "
        "(see rule above); recall (result+setting) also counts numbers in gold `setting` fields. "
        "Totals are pooled over all gold numbers (micro)._"
    )
    return "\n".join(lines)


def render_failures(failures: list[Failure]) -> str:
    lines = ["## Failures", ""]
    if not failures:
        lines.append("_No failures recorded in data/failures.jsonl._")
        return "\n".join(lines)
    lines.append("Latest failure per paper and stage (from `data/failures.jsonl`):")
    lines.append("")
    for f in failures:
        lines.append(
            f"- `{f.paper_id}` — **{f.stage}** ({f.timestamp.isoformat(timespec='seconds')}): "
            f"{' '.join(f.error.split())}"
        )
    return "\n".join(lines)


def _render_flags(flags: list[Flag]) -> str:
    if not flags:
        return "_No automated flags._"
    return "\n".join(f"- ⚠ {f.kind.value}: {f.detail}" for f in flags)


def render_similarity(
    hit: SimilarityHit | None, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> str:
    if hit is None or hit.other_index is None:
        return "- **max similarity:** — (only claim)"
    line = f"- **max similarity:** {hit.score:.2f} (vs claim {hit.other_index + 1})"
    if hit.score >= threshold:
        line += f" — ⚠ near_duplicate (≥ {threshold:.2f})"
    return line


def render_claim(
    k: int,
    claim: Claim,
    flags: list[Flag],
    similarity: SimilarityHit | None = None,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> str:
    evidence = "; ".join(claim.evidence) if claim.evidence else "_none given_"
    return "\n".join(
        [
            f"### Claim {k} — {claim.contribution_type.value}",
            "",
            f"- **statement:** {claim.statement}",
            f"- **contribution_type:** {claim.contribution_type.value}",
            f"- **method:** {claim.method}",
            f"- **problem:** {claim.problem}",
            f"- **setting:** {claim.setting}",
            f"- **result:** {claim.result}",
            f"- **novelty_assertion:** {claim.novelty_assertion}",
            f"- **evidence:** {evidence}",
            f"- **confidence:** {claim.confidence:.2f}",
            render_similarity(similarity, threshold),
            "",
            "Flags:",
            "",
            _render_flags(flags),
        ]
    )


_GOLD_EXTRA_FIELDS = ("method", "problem", "setting", "novelty_assertion", "notes")


def _gold_extras(g: GoldClaim) -> str:
    parts = [
        f"*{name}:* {' '.join(value.split())}"
        for name in _GOLD_EXTRA_FIELDS
        if (value := getattr(g, name))
    ]
    return " · ".join(parts)


def render_gold_block(record: PaperRecord) -> str:
    """Each gold claim next to its closest extracted claim, with number checks."""
    assert record.gold is not None
    gold = record.gold
    extracted = record.claims.claims if record.claims is not None else []
    alignments = (
        {a.gold_index: a for a in record.gold_comparison.alignments}
        if (record.gold_comparison is not None)
        else {}
    )
    heading = f"### Gold vs extracted — source: {gold.source.value}"
    if gold.author:
        heading += f" (by {gold.author})"
    if record.gold_private:
        heading += " · private"
    lines = [heading, ""]
    if gold.source_document:
        lines += [f"_Source document: `{gold.source_document}`_", ""]
    for i, g in enumerate(gold.claims):
        ctype = f" ({g.contribution_type.value})" if g.contribution_type else ""
        lines.append(f"**Gold [{i}]**{ctype}: {' '.join(g.statement.split())}")
        lines.append("")
        extras = _gold_extras(g)
        if extras:
            lines.append(f"- gold fields: {extras}")
        if g.result:
            lines.append(f"- gold result: {' '.join(g.result.split())}")
        a = alignments.get(i)
        if a is None or a.closest_index is None:
            lines.append("- closest extracted: _no extracted claims_")
        else:
            lines.append(
                f"- closest extracted: Claim {a.closest_index + 1} (cosine {a.score:.2f}): "
                f"{' '.join(extracted[a.closest_index].statement.split())}"
            )
        for fld in ("result", "setting"):
            checks = [n for n in (a.numbers if a is not None else []) if n.field == fld]
            if checks:
                marks = " · ".join(f"`{n.number}` {'✓' if n.found else '✗'}" for n in checks)
                lines.append(f"- numbers in gold {fld}: {marks}")
            elif getattr(g, fld):
                lines.append(f"- numbers in gold {fld}: _none_")
        lines.append("")
    lines.append(
        f"Number recall (result): {_recall_cell(record.gold_comparison)}; "
        f"(result+setting): {_recall_all_cell(record.gold_comparison)}."
    )
    lines += [
        "",
        f"Fill `matches_gold` in `{GRADES_PATH}` with the bracketed gold index each extracted "
        "claim matches, or `-1` if it matches none. The closest claim is a lexical hint only.",
    ]
    return "\n".join(lines)


def render_grading(record: PaperRecord) -> str:
    assert record.claims is not None
    lines = ["### Grading", ""]
    for k in range(1, len(record.claims.claims) + 1):
        parts = [
            f"- Claim {k} — specificity (1-5): {BLANK}",
            f"correctness (1-5): {BLANK}",
        ]
        if record.gold is not None:
            parts.append(f"matches gold: {BLANK}")
        parts.append(f"notes: {BLANK}")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def render_paper(
    n: int, record: PaperRecord, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
) -> str:
    e = record.entry
    out: list[str] = [f"## {n}. {record.title}", ""]
    meta = f"{_paper_link(e.paper_id, e.tier)} · tier: **{e.tier.value}**"
    if e.reason:
        meta += f" · {e.reason}"
    out += [meta, ""]

    parse_ok, parse_reason = parse_status(record)
    extract_ok, extract_reason = extract_status(record)
    if not parse_ok:
        out += [f"**Status: ✗ {parse_reason}**", ""]
    elif not extract_ok:
        out += [f"**Status: ✗ {extract_reason}**", ""]
    for err in record.load_errors:
        out += [f"**Load error:** {err}", ""]
    stale = [
        f
        for f in record.failures
        if (f.stage in ("fetch", "parse") and parse_ok) or (f.stage == "extract" and extract_ok)
    ]
    for f in stale:
        out += [
            f"_Earlier {f.stage} failure recorded ({f.timestamp.isoformat(timespec='seconds')}), "
            f"but output now exists: {' '.join(f.error.split())}_",
            "",
        ]

    if record.parsed is not None:
        p = record.parsed
        out += ["**Abstract**", "", _quote(p.abstract) if p.abstract else "_No abstract._", ""]
        if p.contributions:
            out += ["**Contributions statement (from the paper)**", "", _quote(p.contributions)]
        else:
            out += ["_No contributions statement detected._"]
        out.append("")

    if record.claims is not None:
        c = record.claims
        md = c.extraction_metadata
        out += [
            f"**One-line summary:** {c.one_line_summary}",
            "",
            f"**Keywords:** {', '.join(c.keywords)}",
            "",
            f"_Extracted by `{md.model}` with `{md.prompt_version}`, {md.retries} retries._",
            "",
        ]
        for k, claim in enumerate(c.claims, start=1):
            flags = [f for f in record.flags if f.claim_index == k - 1]
            sim = next((h for h in record.similarities if h.claim_index == k - 1), None)
            out += [render_claim(k, claim, flags, sim, threshold), ""]
        notes = md.critic_review
        out += [f"<details><summary>Critic review ({len(notes)} notes)</summary>", ""]
        if notes:
            for note in notes:
                idx = "added" if note.draft_index is None else f"draft {note.draft_index + 1}"
                out.append(f"- **{idx}** — {note.verdict.value}: {note.reason}")
        else:
            out.append("_No critic notes recorded._")
        out += ["", "</details>", ""]
    if record.gold is not None:
        out += [render_gold_block(record), ""]
    if record.claims is not None:
        out += [render_grading(record), ""]
    return "\n".join(out)


def render_report(
    records: list[PaperRecord],
    failures: list[Failure],
    generated_at: datetime,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    public: bool = False,
) -> str:
    """The markdown report. `records` must already be in report order and annotated.

    With `public=True` the caller must pass `public_records(...)` (re-annotated) and failures
    without own papers; this function only adds the header note.
    """
    parts = [
        render_header(generated_at, len(records), threshold, public),
        render_summary_table(records),
        "",
        render_failures(failures),
        "",
        "---",
        "",
    ]
    parts += [render_paper(n, r, threshold) for n, r in enumerate(records, start=1)]
    return "\n".join(parts).rstrip() + "\n"
