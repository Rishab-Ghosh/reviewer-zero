# Demo paper: SPINDLE (fictional)

`spindle.tex` / `spindle.pdf` is a **fictional** 4-page ML paper written for a public demo
recording: the method name (SPINDLE), the authors (Nemo Quillfeather, Ada Placeholder), the
institution and every number are invented, and page 1 says so in a footnote. The 9 cited works
are real papers in our index. Compile with `tectonic spindle.tex`.

## What was planted, and what the pipeline caught

Run on 2026-09-29 with `reviewer review docs/demo/spindle.pdf --venue ICLR --venue-year 2027`
(task 1.9 CLI, web profile) against the local stack: GROBID, index-api on index version
`3ffb61aba1b4a051`. Output: `review.json` (see "What is in review.json").

| # | Planted problem | Caught? | By | What it said |
|---|---|---|---|---|
| 1 | SPINDLE is LoRA (Hu et al., arXiv 2106.09685) renamed, and LoRA is not cited | **Yes** | Prior work (novelty) | LoRA ranked **1st** and labelled "check first" for the method claim; the **possible-prior-claim flag fired**: "LoRA already introduced the identical frozen W0+(α/r)BA update with zero init on query/value projections of RoBERTa-base (r=8, ~0.24% params) and showed it matching or beating full fine-tuning on GLUE". For the rank-ablation claim LoRA is 1st, labelled "builds on" (no flag). |
| 2 | Vaswani et al. cited as 2015 (published 2017) | **No** | Citations | The reference resolved correctly (OpenAlex/arXiv 1706.03762, 2017) but was not flagged: by design, a year printed with a venue ("In Advances in Neural Information Processing Systems, 2015") is only checked against a held version of *that venue*, and the index holds only the arXiv/OpenAlex record (`citations/checks.py::year_flag`: "ML venues are thinly covered"). A year printed on an arXiv-only citation would have been checked. |
| 3 | "Outperforms full fine-tuning on every GLUE task", but Table 1 shows it losing on MNLI and QQP | **Yes, twice** | Methodology (`claims_support`, **confirmed** by both runs); Writing (`abstract_overclaim`) | Methodology: "claim SPINDLE outperforms full fine-tuning on every GLUE task, but Table 1 shows it is worse on MNLI (86.9 vs 87.6) and QQP (90.6 vs 91.9)". Writing: the same two numbers, quoted against the abstract sentence. |
| 4 | Single seed, no variance, called "statistically robust" | **Yes** | Methodology (`seeds_variance` and `significance`, both **confirmed**); Writing (`abstract_overclaim`) | "All results in Tables 1 and 2 come from a single run with seed 0. No standard deviation or confidence interval is reported". "The paper calls the improvements statistically robust without any test, CI, or multiple runs". |
| 5 | Author names and affiliation on page 1 of an "Under review … ICLR 2027" submission | **Yes** | Format (`author_block`, **violation**) | "names on page 1: Nemo Quillfeather, Ada Placeholder; affiliation line: Institute for Imaginary Learning, University of Nowhere". |

**4 of 5 caught.** The miss is a deliberate precision rule of the citation checker, not an error.

Things it also found that were not planted (true of the paper): the missing ICLR 2027 AI-use
statement (format violation), baselines not tuned in the same setup, rank selected on the dev
sets it reports, no ablation of the zero-init "contribution", undisclosed hyperparameter search,
evaluation scope narrower than the "large language models" framing, the abstract saying "each
attention projection" when only query/value are edited, and table captions without metrics.

## Spend

$1.92 at most (the per-call ledger charges a call that fails without usage at its full hold):
$1.548 for the first run, whose novelty step timed out mid-call while the Mac slept and whose
meta call then failed on the network, and $0.37 for the rerun, which reused every cached call
and finished novelty and meta. Budget for this demo: under $5.

## What is in review.json

Every checker's output from the final run: format, citations (references, flags and support
verdicts), claims, novelty (top 10 per claim with display labels and the possible-prior-claim
flag), methodology (both runs' consensus) and writing. **Omitted on purpose:** the
meta-review, because its stored rating and sub-scores are never shown to users (MASTER_PLAN
D11) and the leak filter that makes its text safe lands in W3; and the internal novelty
severity score (D14).

## Live `review_paper` run (MCP server, 2026-10-02)

The full review through the real `reviewer-zero` stdio server (`review_paper`, venue
`iclr2027`), fresh cache, local GROBID and local public-mode index (`3ffb61aba1b4a051`).

| | |
|---|---|
| Plan call (`confirm=false`) | $0; estimate **about $1.46**, hard limit **$2.90** (`review_paper_live_plan.json`) |
| Confirmed run | **$1.4835**, 451 s, 8 of 8 steps done; per step: citations $0.082, claims $0.089, methods $0.154, writing $0.071, novelty $0.965, meta $0.122 |
| Progress | 8 MCP progress notifications (`review_paper_live_progress.json`): writing at 24 s, citations 30 s, methods 47 s, claims 54 s, novelty 424 s, meta 451 s |
| Resume | calling again on the same PDF: plan $0 with all 8 steps already done; the confirmed call returned in 2 s for $0 |

What it reported (`review_paper_live.json`): LoRA (2106.09685) **ranked 1st for both claims**,
with the possible-prior-claim flag on the method claim; methodology `seeds_variance`,
`significance` and `claims_support` (plus baseline tuning, dev-set selection, missing
ablations, hyperparameter disclosure, evaluation scope), all confirmed by both runs; four
abstract over-claims and two captions without metrics; the format author-block violation and
the missing ICLR 2027 AI-use statement. No citation flags, as in the first demo run (the year
on a venue citation is not checked, see above). The meta-review comes back as the filtered
allowlist: summary, three "most likely to sink this paper" points, strengths, questions and
what would change the reviewer's mind — no rating, score or verdict anywhere in the output.

**A leak the filter caught only after this run.** The model's first "what would change my
mind" said "Only a demonstrated, substantive methodological difference from LoRA would make
this publishable." No D12 pattern covered "publishable"; a publication-verdict pattern was
added (DECISIONS M8-4), and the saved output is the same run re-served from its cache ($0)
with that sentence removed and the left-out note shown.
