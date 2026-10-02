"""Markdown rendering for novelty results and the recall eval. Pure."""

from reviewer.models import (
    CandidateSource,
    ClaimNovelty,
    JudgedCandidate,
    NoveltyEvalCaseResult,
    PaperNovelty,
    TokenUsage,
)
from reviewer.novelty.evaluate import different_examples, label_counts


def _cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def _year(j: JudgedCandidate) -> str:
    p = j.candidate.paper
    y = p.published_date.year if p.published_date else p.year
    return str(y) if y else "?"


def _source(j: JudgedCandidate) -> str:
    if j.candidate.hits:
        hits = sorted(j.candidate.hits, key=lambda h: h.rank)
        via = ["x"] if CandidateSource.expansion in j.candidate.sources else []
        return ", ".join([f"{h.query_id}#{h.rank}" for h in hits[:4]] + via)
    srcs = j.candidate.sources
    if srcs == [CandidateSource.expansion]:
        return "exp via " + ", ".join(j.candidate.via)
    tags = {
        CandidateSource.search_statement: "stmt",
        CandidateSource.search_method_problem: "m+p",
        CandidateSource.expansion: "exp",
    }
    return "+".join(tags[s] for s in srcs)


def counts_line(c: ClaimNovelty) -> str:
    n = c.counts
    return (
        f"Candidates: statement search {n.search_statement}, method+problem search "
        f"{n.search_method_problem}, union {n.search_union}; expansion added {n.expansion_new} "
        f"({n.expansion_nodes_empty}/{n.expansion_nodes} expanded hits had no neighbors); "
        f"LLM pool {n.pool}, of which {n.pool_from_expansion} from expansion only and "
        f"{n.pool_from_generated_only} from generated queries only ({n.n_queries} queries). "
        f"Dropped: {n.filtered_self} self; guard (should be 0): {n.filtered_future} past cutoff, "
        f"{n.filtered_undated} undated, {n.filtered_no_metadata} without title/abstract."
    )


LOW_EVIDENCE_POOL = 10
"""Below this many judged candidates, a label (especially 'different' across the board) mostly
reflects thin retrieval rather than a considered judgment."""

TOP_SHOWN = 10
"""Candidates shown in the product report and used for the top-label summary column."""


def duplicate_flag(c: ClaimNovelty) -> str | None:
    """'Possible duplicate' only when the internal severity check judges a single prior work to
    already contain this claim's core contribution (severity == 3), naming that work. The 0-3
    score itself, and any lower value, is never shown here; severity stays internal (evals only).
    None when there is nothing to flag."""
    if c.severity is None or c.severity.score < 3:
        return None
    driver_key = c.severity.drivers[0] if c.severity.drivers else None
    driver = next((j for j in c.ranked if j.candidate.paper.key == driver_key), None)
    name = f"{driver.candidate.paper.title} ({driver_key})" if driver else driver_key or "?"
    return f"**Possible duplicate — {name}:** {c.severity.reason}"


def low_evidence_note(c: ClaimNovelty) -> str | None:
    if c.counts.pool < LOW_EVIDENCE_POOL:
        return f"*Low evidence: only {c.counts.pool} candidates were retrieved and judged.*"
    return None


def render_claim(c: ClaimNovelty) -> list[str]:
    lines = [f"### Claim {c.claim_index + 1}", ""]
    for note in (duplicate_flag(c), low_evidence_note(c)):
        if note:
            lines += [note, ""]
    lines += [
        f"> {_cell(c.query.statement)}",
        "",
        counts_line(c),
        "",
        "| # | label | year | key | sim | source | title | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    top = c.ranked[:TOP_SHOWN]
    for i, j in enumerate(top, start=1):
        p = j.candidate.paper
        lines.append(
            f"| {i} | {j.label.value} | {_year(j)} | {p.key} | {j.candidate.similarity:.3f} "
            f"| {_source(j)} | {_cell(p.title)} | {_cell(j.reason)} |"
        )
    if not top:
        lines.append("| – | no candidates | | | | | | |")
    if c.queries:
        lines += ["", "Queries (source column: query id#rank; x = expansion):", ""]
        lines += [f"- `{q.id}` {q.origin}/{q.kind}: {_cell(q.text)}" for q in c.queries]
    return [*lines, ""]


def _usage(u: TokenUsage) -> str:
    return f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens, ${u.cost_usd:.2f}"


def render_report(papers: list[PaperNovelty], failures: list[str]) -> str:
    total = TokenUsage()
    for p in papers:
        total += p.token_usage
    all_claims = [c for p in papers for c in p.claims]
    tops = [c.ranked[:TOP_SHOWN] for c in all_claims]
    exp_pool = sum(c.counts.pool_from_expansion for c in all_claims)
    pool = sum(c.counts.pool for c in all_claims)
    exp_top = sum(j.candidate.sources == [CandidateSource.expansion] for t in tops for j in t)
    n_flagged = sum(duplicate_flag(c) is not None for c in all_claims)
    lines = [
        "# Novelty check v0 (private)",
        "",
        f"Per claim: the {TOP_SHOWN} closest prior works found in reviewer-index, each labelled",
        "by the LLM with a one-line reason. A claim is flagged **possible duplicate** only when",
        "an internal check judges a single prior work to already contain its core contribution,",
        "naming that work; otherwise nothing is flagged and it is on you to read the labels and",
        "reasons. The index returns only papers published before the paper's arXiv v1 date",
        "(`before`); own papers have no cutoff. Labels and flags are the model's, not graded.",
        "",
        f"Papers: {len(papers)}, claims: {len(all_claims)}. Possible duplicates flagged: "
        f"{n_flagged}/{len(all_claims)}. Pool candidates from expansion only: {exp_pool}/{pool}; "
        f"in the shown top {TOP_SHOWN}: {exp_top}/{sum(len(t) for t in tops)}.",
        f"LLM usage (incl. cached originals): {_usage(total)}.",
        "",
        "| paper | tier | claim | possible duplicate | pool | from expansion | top label |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in papers:
        for c in p.claims:
            top_label = c.ranked[0].label.value if c.ranked else "–"
            dup = "yes" if duplicate_flag(c) else "–"
            lines.append(
                f"| {p.paper_id} | {p.tier} | {c.claim_index + 1} | {dup} "
                f"| {c.counts.pool} | {c.counts.pool_from_expansion} | {top_label} |"
            )
    if failures:
        lines += ["", "## Failures", ""] + [f"- {f}" for f in failures]
    for p in papers:
        lines += [
            "",
            f"## {p.paper_id}: {_cell(p.title)}",
            "",
            f"Tier {p.tier}; cutoff {p.cutoff or 'none (as of today)'}"
            f"{f' ({p.cutoff_source})' if p.cutoff and p.cutoff_source else ''}; "
            f"index {p.index_version}; "
            f"{p.model}, {p.prompt_version}; {_usage(p.token_usage)}.",
            "",
        ]
        for c in p.claims:
            lines += render_claim(c)
    return "\n".join(lines).rstrip() + "\n"


def _group_table(groups: dict[str, dict[str, int]]) -> list[str]:
    if not groups:
        return []
    lines = [
        "Which queries retrieved the named works (a work counts once per query group that found",
        "it; 'only' = no other group found it; pooled/top5 = of those finds):",
        "",
        "| query group | found | only | pooled | top5 |",
        "|---|---|---|---|---|",
    ]
    for g, row in groups.items():
        lines.append(f"| {g} | {row['found']} | {row['only']} | {row['pooled']} | {row['top5']} |")
    return [*lines, ""]


def _diagnostics(results: list[NoveltyEvalCaseResult]) -> list[str]:
    counts = label_counts(results)
    if not any(counts.values()):
        return []
    lines = [
        "## Labels given to named works in the pool",
        "",
        "| label | named works |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in counts.items()],
        "",
    ]
    examples = different_examples(results)
    if examples:
        lines += [f"### {len(examples)} named works labelled `different`", ""]
        for i, (case_id, statement, nj) in enumerate(examples, start=1):
            lines += [
                f"**{i}. case {case_id}; named work {nj.key} (rank {nj.rank})**",
                "",
                f"- Claim (submission): {_cell(statement)}",
                f"- Named work: *{_cell(nj.title)}*. {_cell(nj.abstract)}",
                f"- Reranker reason: {_cell(nj.reason)}",
                "",
            ]
    return lines


def render_eval(
    results: list[NoveltyEvalCaseResult],
    summary: dict[str, float],
    cases_path: str,
    groups: dict[str, dict[str, int]] | None = None,
) -> str:
    ok = [r for r in results if r.error is None]
    lines = [
        "# Novelty eval (private)",
        "",
        f"Cases: {cases_path} ({len(results)} cases, {len(ok)} scored, "
        f"{len(results) - len(ok)} failed). Named prior works resolved in the index: "
        f"{sum(r.n_named for r in ok)}; not in the index (excluded from recall): "
        f"{sum(r.n_unresolved for r in ok)}; resolved but newer than the cutoff (excluded): "
        f"{sum(r.n_after_cutoff for r in ok)}.",
        "",
        "Query per case: submission title + abstract, and the title alone (no claim extraction).",
        "Cutoff: the row's cutoff_date (the venue's submission deadline).",
        "",
        "Recall is micro-averaged over all reviewer-named works in scored cases. Union recall =",
        "any query (or expansion) retrieved the named work; pool recall = it passed the first-pass",
        "filter and was judged by the LLM (the ceiling for recall@k).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| recall@5 | {summary['recall_at_5']:.3f} |",
        f"| recall@10 | {summary.get('recall_at_10', 0.0):.3f} |",
        f"| recall@20 | {summary['recall_at_20']:.3f} |",
        f"| pool recall | {summary['pool_recall']:.3f} |",
        f"| union recall | {summary.get('union_recall', 0.0):.3f} |",
        "",
        *_group_table(groups or {}),
        *_diagnostics(results),
        "| case | venue | named | unresolved | ranks | R@5 | R@10 | R@20 | pool | verdict "
        "| error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ranks = ", ".join(str(x) if x else "–" for x in r.found_at)
        lines.append(
            f"| {r.case_id} | {r.venue} | {r.n_named} | {r.n_unresolved} | {ranks} "
            f"| {r.recall_at_5:.2f} | {r.recall_at_10:.2f} | {r.recall_at_20:.2f} "
            f"| {r.pool_recall:.2f} | {r.verdict.value if r.verdict else '–'} "
            f"| {_cell(r.error)} |"
        )
    return "\n".join(lines) + "\n"


def _full_source(j: JudgedCandidate) -> str:
    hits = ", ".join(
        f"{h.query_id}#{h.rank}" for h in sorted(j.candidate.hits, key=lambda h: h.rank)
    )
    via = j.candidate.via
    parts = [hits] if hits else []
    if via:
        parts.append("expansion via " + ", ".join(via))
    return "; ".join(parts) or "–"


def render_detail(p: PaperNovelty) -> str:
    """Every judged candidate of every claim with full provenance (private)."""
    lines = [
        f"# Novelty detail: {p.paper_id}",
        "",
        f"*{_cell(p.title)}*",
        "",
        f"Cutoff {p.cutoff or 'none (as of today)'}; index {p.index_version}; {p.model}; "
        f"prompts {p.prompt_version}; {_usage(p.token_usage)}.",
        "",
        "Source = every query that returned the candidate as `query id#rank` (rank within that",
        "query's results), plus expansion seeds. LW = position in the listwise ranking.",
    ]
    for c in p.claims:
        lines += ["", f"## Claim {c.claim_index + 1}", ""]
        for note in (duplicate_flag(c), low_evidence_note(c)):
            if note:
                lines += [note, ""]
        lines += [
            f"> {_cell(c.query.statement)}",
            "",
            counts_line(c),
            "",
            "### Queries",
            "",
            "| id | origin | kind | component | query |",
            "|---|---|---|---|---|",
            *[
                f"| {q.id} | {q.origin} | {q.kind} | {_cell(q.component)} | {_cell(q.text)} |"
                for q in c.queries
            ],
            "",
            f"### Judged candidates ({len(c.ranked)}; {len(c.unpooled)} more retrieved but not "
            "pooled)",
            "",
            "| # | label | LW | year | key | sim | source | title | reason | listwise reason |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, j in enumerate(c.ranked, start=1):
            pp = j.candidate.paper
            lines.append(
                f"| {i} | {j.label.value} | {j.listwise_rank or '–'} | {_year(j)} | {pp.key} "
                f"| {j.candidate.similarity:.3f} | {_full_source(j)} | {_cell(pp.title)} "
                f"| {_cell(j.reason)} | {_cell(j.listwise_reason)} |"
            )
    return "\n".join(lines) + "\n"


def _size_table(by_size: dict) -> list[str]:
    if len(by_size) < 2:
        return []
    lines = [
        "| pool size | rule pool recall | learned (CV) pool recall | rerank calls |",
        "|---|---|---|---|",
    ]
    for n, row in sorted(by_size.items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {n} | {row['rule_pool_recall']:.3f} | {row['learned_pool_recall_cv']:.3f} "
            f"| {row['rerank_calls']} |"
        )
    return [*lines, ""]


def render_first_stage(run: dict) -> str:
    """Pool recall of the current pool vs the first-stage ranker (retrieval only, dev)."""
    s = run["summary"]
    model = run["learned_model_all_cases"]
    weights = ", ".join(
        f"{f} {w:+.2f}" for f, w in zip(model["features"], model["weights"], strict=True)
    )
    return "\n".join(
        [
            "# First-stage ranker (private, dev, retrieval only)",
            "",
            f"Commit {run['commit'][:10]}{' (dirty)' if run['dirty'] else ''}; index "
            f"{run['config']['index_version']}; {run['n_cases']} submissions "
            f"({run['n_failed']} failed), {run['n_named']} named works; pool size "
            f"{run['pool_size']}. No LLM calls (generated queries from cache).",
            "",
            f"Candidates per submission (search + /expand): median {run['median_candidates']:.0f}; "
            f"new via /expand: median {run['median_expand_new']:.0f} "
            f"(seeds: top {run['config']['expand_seeds']} search results).",
            "",
            "| pool | pool recall |",
            "|---|---|",
            f"| current (round-robin, /neighbors expansion) | {s['current_pool_recall']:.3f} |",
            f"| rule-based on search + /expand | {s['rule_pool_recall']:.3f} |",
            f"| logistic regression, 5-fold CV by submission | {s['learned_pool_recall_cv']:.3f} |",
            "",
            "| candidate set | union recall |",
            "|---|---|",
            f"| current (search + /neighbors) | {s['current_union_recall']:.3f} |",
            f"| search + /expand | {s['stage_union_recall']:.3f} |",
            "",
            *_size_table(run.get("by_pool_size", {})),
            f"Logistic weights (standardised features, fitted on all cases): {weights}.",
            "The submission's own reference list is never used.",
            "",
        ]
    )


def render_negatives(run: dict, results: list) -> str:
    """Verdicts on submissions reviewers found novel; every already_done spelled out."""
    v = run["verdicts"]
    lines = [
        "# Novelty checker on negatives (private, dev)",
        "",
        f"{run['n_cases']} accepted submissions whose reviews raise no overlap or novelty concern "
        f"and praise novelty ({run['n_failed']} failed). Commit {run['commit'][:10]}"
        f"{' (dirty)' if run['dirty'] else ''}; index {run['config']['index_version']}; first "
        f"stage {run['config']['first_stage']}, pool {run['config']['pool_size']}, "
        f"{run['config']['rerank_prompt']}, listwise cap {run['config']['listwise_cap']}. "
        "Query: title + abstract plus paper-level generated queries; before = the venue deadline.",
        "",
        "| verdict | submissions |",
        "|---|---|",
        *[f"| {k} | {n} |" for k, n in v.items()],
        "",
        "## Every already_done verdict",
        "",
        "Each: the submission (what was checked), the top prior work, and why it was judged "
        "same/subsumes. Decide whether it is a false alarm or a real find the reviewers missed.",
    ]
    done = [(c, r) for c, r in results if r.verdict.value == "already_done"]
    if not done:
        lines += ["", "None."]
    for k, (case, r) in enumerate(done, start=1):
        overlap = [j for j in r.top if j.label.value in ("same", "subsumes")]
        top = r.top[0]
        p = top.candidate.paper
        lines += [
            "",
            f"### {k}. {_cell(case.title)} ({case.case_id}, {case.venue})",
            "",
            f"> {_cell(case.abstract)}",
            "",
            f"**Top prior work:** {_cell(p.title)} ({p.key}, {_year(top)}), label "
            f"**{top.label.value}**, listwise #{top.listwise_rank or '–'}.",
            "",
            f"- Reranker reason: {_cell(top.reason)}",
            f"- Listwise reason: {_cell(top.listwise_reason) or '–'}",
            f"- Abstract of the prior work: {_cell(p.abstract)}",
        ]
        for j in overlap:
            if j is top:
                continue
            q = j.candidate.paper
            lines.append(
                f"- Also {j.label.value} in the top 5: {_cell(q.title)} ({q.key}, {_year(j)}): "
                f"{_cell(j.reason)}"
            )
    return "\n".join(lines) + "\n"


def render_severity(s: dict) -> str:
    d = s["distributions"]
    lines = [
        "# Novelty severity on dev: positives vs negatives (private)",
        "",
        f"Commit {s['commit'][:10]}{' (dirty)' if s['dirty'] else ''}; index replay "
        f"{s['index_replay']}. Positives: {s['n_positives']} dev submissions whose reviewers named "
        f"prior work; negatives: {s['n_negatives']} accepted submissions whose reviewers praised "
        "novelty. Paper-level severity = max over claims (one query per submission here).",
        "",
        f"**AUC of paper-level severity (positives vs negatives): {s['auc']:.3f}** (ties count "
        "half).",
        "",
        "| severity | positives | negatives |",
        "|---|---|---|",
        *[
            f"| {k} | {d['positives'].get(k, d['positives'].get(str(k), 0))} "
            f"| {d['negatives'].get(k, d['negatives'].get(str(k), 0))} |"
            for k in range(4)
        ],
        "",
        f"Positive-set recall with this pipeline: {s['positive_recall']}.",
        "",
        "## Negatives scoring 3",
    ]
    if not s["negatives_scoring_3"]:
        lines += ["", "None."]
    for x in s["negatives_scoring_3"]:
        lines += [
            "",
            f"- **{_cell(x['title'])}** ({x['case_id']}): driven by {'; '.join(x['drivers'])}.",
            f"  Reason: {_cell(x['reason'])}",
        ]
    return "\n".join(lines) + "\n"
