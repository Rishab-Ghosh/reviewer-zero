# Measured claims for public copy

Aggregate numbers only: no paper ids, titles or review text. **Every number here is from the
development set** (ICLR 2024 and 2025 submissions and reviews from OpenReview). The held-out test
set has not been run (MASTER_PLAN Q7). Copy that quotes these numbers must say "development set".

## 1. Novelty: recall@10 of reviewer-named prior work

**Suggested wording:** "On 50 ICLR 2024–2025 submissions from our development set, the prior work
that reviewers named appeared in our top 10 for 32 of 74 named papers (43%). Development set, not
a held-out test."

| | |
|---|---|
| recall@10 | **0.432 = 32 / 74** |
| recall@5, recall@20 | 0.365 (27 / 74), 0.459 (34 / 74) |
| cases | 50 submissions (33 ICLR 2025, 17 ICLR 2024), 0 failed |
| denominator | 74 reviewer-named prior works found in the index; 15 more named works are not in the index and are excluded (counting them, 32 / 89 = 36%); 0 excluded as newer than the cutoff |
| cutoff | each submission's deadline: works published later are not eligible |
| queries | submission title + abstract, and the title alone. **No claim extraction:** the product queries from the paper's top 2 extracted claims, so this measures retrieval and reranking, not claim extraction |
| pipeline | rerank `rerank_v3` (Sonnet 5), queries `queries_v1` and listwise `listwise_v1` (Opus 5.5), learned first stage, pool 150 |
| run commit | `85a4bff69fe29394c6700f353a2364327bf1ac72` (clean), 2026-09-27 |
| index | 47 cases replayed on index `967e0c4e9906a5aa`; 3 cases whose expansion response was not cached ran live on index `ae2b6b9221c8ad1a` |
| reproduced | 2026-09-28 at `29154171c4894fd8bcb4e95f7202ae977de75207` (current `main`) from cached responses only, $0: 31 / 71 on the 47 replayed cases + 1 / 3 on the 3 others = 32 / 74. The novelty scoring code has not changed since the run commit. |

### 1.1 Lite novelty (`review_paper` with `novelty="lite"`)

Measured 2026-10-02 (M11) on the same 50 cases and 74 named works, replayed on the same index
versions as §1 (47 cases on `967e0c4e9906a5aa`, 3 on `ae2b6b9221c8ad1a`), same prompts and
models, with the reranker pool cut from 150 to 60 candidates per claim, so the only difference
is the pool:

| | full (pool 150, the default) | lite (pool 60) |
|---|---|---|
| recall@10 | 0.432 = 32 / 74 (§1) | **0.365 = 27 / 74** |
| named works in the reranker pool | 0.581 = 43 / 74 | 0.405 = 30 / 74 |
| `review_paper` on the demo paper, fresh cache | 172 s, $1.61 | ≈ 2 min, ≈ $0.88 |

The lite eval spent $4.27 (50 cases). Times and costs are one run each on the 4-page demo
paper (DECISIONS M7-5, M11-2).

## 2. "Possible prior claim" flag

The flag (`novelty/report.py::duplicate_flag`) fires when the severity check scores a claim 3,
"a single prior work already contains this claim's core contribution" (D14).

**Suggested wording:** "On 50 accepted ICLR papers whose reviewers praised their novelty, the
'possible prior claim' flag fired on none (0 of 50). Development set."

Use it only as a false-alarm number. The flag fired on only 3 of the 50 positives, so no copy
should say or suggest that it catches prior work in general. The top-10 list in §1 is what finds
prior work.

| | fired (severity 3) | not fired | n |
|---|---|---|---|
| negatives: accepted submissions whose reviewers praised novelty | **0** | 50 | 50 (32 ICLR 2025, 18 ICLR 2024) |
| positives: submissions whose reviewers named specific prior work | **3** | 47 | 50 (the §1 cases) |

| | |
|---|---|
| severity version | `severity_v2` (Opus 5.5); paper-level severity = the maximum over queries |
| severity distribution | positives 1: 33, 2: 14, 3: 3; negatives 1: 36, 2: 14, 3: 0 (AUC 0.538) |
| uncertainty | 0 of 50 still allows a false-alarm rate of up to about 6% (95% upper bound, rule of three) |
| run commit | scores from `85a4bff69fe29394c6700f353a2364327bf1ac72` (where `severity_v2` landed); the flag rule and the merged severity report come from `719cd883f215c641b52153be0ca8a97825b9ba11` (clean), 2026-09-27 |
| index | as in §1: 97 of 100 cases on `967e0c4e9906a5aa`, 3 positives on `ae2b6b9221c8ad1a` |
| reproduced | 2026-09-28 at `29154171c4894fd8bcb4e95f7202ae977de75207` from cached responses only, $0: the 97 replayed cases give positives 1: 31, 2: 13, 3: 3 and negatives 1: 36, 2: 14, 3: 0; the other 3 positives give 1, 1, 2. The merged result is identical to the recorded one. |

Both results are **reproducible from cached responses**. A replay needs index version
`967e0c4e9906a5aa` for 97 cases and `ae2b6b9221c8ad1a` for the other 3, because the 3 cases'
expansion responses exist only under the second.

## 3. index_size: what the prior-work search covers

**Suggested wording:** "We search 1.09 million papers: 577,000 arXiv papers in ML and adjacent
fields since 2017, plus 516,000 other works: papers at least 5 of them cite, and papers from
venues not on arXiv. They are linked by 28.3 million citation edges." Say "about" if the copy is not updated when the index grows: the
index adds papers nightly.

| | |
|---|---|
| arXiv papers | **577,299**, published 2017-01-01 to 2026-09-25; largest primary categories cs.CV 153,356, cs.LG 139,544, cs.CL 85,068, cs.AI 40,428 |
| other searchable works | **515,589** = 500,944 works outside the arXiv set, each cited by ≥ 5 of its papers (median 9; 255,037 published before 2017), + 14,645 papers from venues not on arXiv (largest: DATE, DAC, VLSI, ISSCC, ICCAD) |
| total searchable documents | **1,092,888** |
| citation edges | **28,261,886** distinct citing→cited pairs (29,460,289 rows across sources: Semantic Scholar and OpenAlex) |
| index_version | `5b5c4c45a10ef9f9` |
| read at | 2026-09-29 04:38 UTC, one read-only snapshot of the live index (the development machine's copy; W1 task 1.3 restores it to the server) |

"Searchable" means embedded with the search model (`BAAI/bge-base-en-v1.5`), which is what
`/search` queries. Of those documents, 211,764 external works and 678 venue papers are embedded
from the title alone (no abstract available). None of the searchable external works is excluded
as a mis-merged record or an arXiv duplicate. The table has 17,019 more external and 1,579 more
venue rows that are not embedded and are not searchable, so they are not counted. MASTER_PLAN
and BP say "710k papers": the arXiv count today is 577,299.

## 4. check_citations: GROBID vs the text-layer fallback

Measured 2026-10-02 on the 30 public dev papers (`data/parsed/*.json`, arXiv PDFs), both
reference lists resolved the same way: a local copy of the index (`3ffb61aba1b4a051`), then
OpenAlex, Crossref and arXiv, with no LLM. OpenAlex's paid title search was at its daily budget
floor during the run and arXiv rate-limited the IP (HTTP 429), so both rates are lower bounds;
the bias between the two parsers is small (second half of the papers: GROBID 88.5%, fallback
87.6% resolved; first half 93.2% / 86.1%).

| | GROBID (local container) | Fallback (PDF text layer, pypdf) |
|---|---|---|
| References found | 1,646 | 1,375 (83.5% of GROBID's) |
| Resolved | 1,509 (91.7%) | 1,190 (86.5% of its own; **72.3% of GROBID's count**) |
| Papers with no reference list found | 0 | 1 of 30 |
| Flags: wrong_year / duplicate / wrong_authors / unresolved | 5 / 14 / 1 / 14 | 22 / 46 / 10 / 24 |

**Suggested wording:** "Without GROBID, Reviewer Zero reads the reference list from the PDF's
text and resolves about 72% as many references as with GROBID on our dev papers, with noisier
flags; run GROBID locally for full results." The fallback's extra flags are most likely
splitting errors (merged or broken entries); they were counted, not individually verified.

## 5. The MCP path: prior-work recall when Claude writes the queries

Measured 2026-10-02 (M11, MCP_PLAN §6). Question: what recall does a user get when Claude
writes the queries from the `find_prior_work` tool description, compared with the pipeline's
43.2% (§1)?

| | Sonnet 5 (`claude-sonnet-5`) | Opus 5.5 (`claude-opus-5-5`) |
|---|---|---|
| recall@10, the tool's own order (no LLM ranking) | 0.162 = 12 / 74 | 0.095 = 7 / 74 |
| **recall@10 after the model ranks the 40 candidates** | **0.257 = 19 / 74** | **0.270 = 20 / 74** |
| recall@40 (everything the model sees) | 0.365 = 27 / 74 | 0.311 = 23 / 74 |
| cases scored, errors | 50, 0 | 50, 0 |
| spend (100 calls / 96 calls) | $2.55 | $4.42 |

- **Cases:** the same 50 dev cases, 74 reviewer-named works and cutoffs as §1. Dev only.
- **Queries:** one plain Messages API call per case, system prompt = the `find_prior_work` tool
  description, input = the submission's title and abstract (what §1's pipeline queried from).
  Not our `queries_v1` prompt.
- **Retrieval:** `find_prior_work` itself (k = 40, `before` = the cutoff), on today's local copy
  of the index **`3ffb61aba1b4a051`**. §1 ran on `967e0c4e9906a5aa`; cutoffs still exclude later
  work, but the index grew in between, so the two are not the same corpus.
- **Ranking:** one plain call (same model) that drops the paper itself and labels and orders
  the 40 candidates as SKILL.md step 4 says (same, close, builds on, different).
- **Code:** at `4017159` (find_prior_work unchanged since M3-5). Scripts are not in the repo
  (scratch measurement harnesses); rows recorded per case.

**What it means.** The MCP path finds about 26–27% of reviewer-named prior work in its top 10,
against 43% for the measured pipeline. Most of the gap is retrieval, not ranking: the pipeline's
first stage sends 150 candidates per claim to an LLM reranker (58% of the named works are in
that pool, §1.1), while the tool returns 40 and the model sees only those (recall@40 31–37%). Ranking roughly doubles the
tool's own top-10 recall for both models. Opus wrote queries whose raw retrieval was worse than
Sonnet's here (7 vs 12 at 10, 23 vs 27 at 40) but ranked better; with 74 works, a difference
of a few hits is within noise.

**Suggested wording:** none for the landing page yet. If one is needed: "In Claude, Reviewer
Zero's prior-work search found about a quarter of the prior work real reviewers named, in its
top 10 (dev set); our measured pipeline (`review_paper`) finds 43%." Do not quote the MCP number
as the product's recall without saying which path it is.

### 5.1 The skill path end to end

Measured 2026-10-02: the whole no-key workflow (`skills/reviewer-zero/SKILL.md`) run by
**Opus 5.5** through the Messages API with the six tools as real tool calls (executed locally
against the same local index) and the paper's PDF as a document, on dev cases from §1's 50
that have a downloaded OpenReview PDF. The $6 cap for this arm stopped it after **6 finished
papers**; a 7th was cut off before writing its review and is not counted.

| per finished paper (n = 6) | |
|---|---|
| reviewer-named prior work among the first 10 works reported for some claim | **4 of 6** named works (1 per paper) |
| … of the named works the PDF does **not** already cite | 1 of 3 |
| … of the named works the PDF already cites | 3 of 3 |
| quotes submitted to `verify_quotes` (all calls) | 188: 184 verified, 3 not found, 1 too short (97.9%) |
| score, rating or verdict sentences in the review (D12 leak filter over the review text) | **0** in 6 reviews |
| claims checked, `find_prior_work` calls | 2–4 claims, 2–3 calls |
| API cost (Opus 5.5, prompt caching) | median **$1.00**, range $0.63–1.17 |
| time | median **234 s**, range 181–255 s |

**Caveats.** n = 6, one named work each: this shows the workflow runs end to end with
verified quotes and no verdict language; it is not a recall estimate. The skill reads the whole
paper, references included, while §1 and §5 query from the title and abstract, so the numbers
are not comparable; most hits were works the paper already cites. 4 of the 6 PDFs are the
current OpenReview version, revised after the reviews (2 are dated at the deadline), so a
revision may cite what reviewers named. On a Claude plan the user pays no API cost; the cost
row shows the scale of the work (3–6 model turns, each rereading the cached paper and tool results).

### 5.2 More candidates per call: 40 vs 100 (and the free ceiling to 150)

Measured 2026-10-02. **Free ceiling** (no LLM; the Sonnet and Opus arms' cached queries, same 50
cases, the tool's own order): named works among the first k candidates.

| k | 10 | 20 | 40 | 60 | 80 | 100 | 150 |
|---|---|---|---|---|---|---|---|
| Sonnet 5 queries (of 74) | 12 | 15 | 27 | 31 | 34 | 35 | 41 |
| Opus 5.5 queries (of 74) | 7 | 15 | 23 | 28 | 31 | 33 | 40 |

**A/B with ranking** (Sonnet 5, the decision rule fixed before the run: 100 becomes the
default only if its recall@10 beats 40 by ≥ 4 named works). Same cached queries, a compact
listing (key, title, year, first 300 characters of the abstract) and one ranking call that
returns only overlapping works (same, close, builds on). The $3 cap ran out after **39 of the
50 cases** (59 named works), scored paired:

| | k = 40 | k = 100 |
|---|---|---|
| named works in the candidate list | 18 | 26 |
| **recall@10 after ranking** | **13 / 59** | **13 / 59** |
| recall@5, recall@20 | 9, 17 | 11, 18 |
| listing size per call (median) | ≈ 15.8k chars | ≈ 39.6k chars |

**Decision: 40 stays the default.** The extra candidates contain 8 more named works, but one
ranking pass does not bring them into the top 10. On the same 39 cases the earlier full listing
(abstract[:600], every candidate labelled; §5) also gave 13 / 59, so the compact listing loses
nothing at 40. Spend $2.97.

