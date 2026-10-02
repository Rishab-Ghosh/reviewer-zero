# Reviewer Zero

An open-source MCP server that reviews your ML paper before submission the way a strict PI
would: format and anonymization, every reference, prior work for each claim, a methodology
checklist and a writing check, with every quote checked against your PDF. It runs inside
Claude Code or Claude Desktop, on your own Claude plan. It never gives a score, a rating or an
accept/reject verdict, and it never rewrites your text.

MIT licensed. Package `reviewer-zero` (import name `reviewer`).

## Install

You need [uv](https://docs.astral.sh/uv/) and a free index key. Get a free key at
https://tryreviewerzero.com. Questions: hello@tryreviewerzero.com. The key meters use of the
hosted search index; it is not a payment method.

**Claude Code**

```bash
claude mcp add reviewer-zero -e REVIEWER_ZERO_INDEX_KEY=<your key> -- uvx reviewer-zero
# the review workflow (the skill), so Claude knows the order and the rules:
git clone https://github.com/rishab-ghosh/reviewer-zero
mkdir -p ~/.claude/skills && cp -r reviewer-zero/skills/reviewer-zero ~/.claude/skills/
```

Then ask: "Review my paper at ~/papers/draft.pdf for ICLR 2027."

**Claude Desktop**: add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "reviewer-zero": {
      "command": "uvx",
      "args": ["reviewer-zero"],
      "env": { "REVIEWER_ZERO_INDEX_KEY": "<your key>" }
    }
  }
}
```

and add `skills/reviewer-zero` as a skill (zip the folder and upload it in Claude's skills
settings), or paste `skills/reviewer-zero/SKILL.md` into a project's instructions.

**Optional, recommended: GROBID** for the best reference parsing (and required by
`review_paper`). It runs locally; your PDF never leaves your machine for it:

```bash
docker run --rm -p 8070:8070 grobid/grobid:0.9.1-crf
```

Without it, `check_citations` reads the reference list from the PDF's text layer and resolves
about 72% as many references (see "Measured" below).

## Two ways to review

**The default: the skill, on your Claude plan, no API key.** Claude reads your paper and
writes the review following `skills/reviewer-zero/SKILL.md`; the tools do retrieval and the
checks that need code. Every quote goes through `verify_quotes` before Claude may use it.

**The measured pipeline: `review_paper`, on your own `ANTHROPIC_API_KEY`.** Our prompts and
models, the pipeline we measured on the dev set. It needs a local GROBID and the key in the
server's environment (`-e ANTHROPIC_API_KEY=...`). It always shows an estimate and a hard spending limit first
(`confirm=false` spends nothing) and runs only after you agree. Two novelty settings:

| `novelty=` | Candidates per claim sent to the reranker | Dev recall@10 | Demo paper, end to end | Demo paper, cost |
|---|---|---|---|---|
| `full` (default) | 150 | **43.2%** (32 / 74) | **172 s** | **$1.61** |
| `lite` | 60 | **36.5%** (27 / 74) | **≈ 2 min** | **≈ $0.88** |

Recall is on the same 50 dev cases (74 prior works named by real reviewers); times and costs
are one run each on our 4-page demo paper (`docs/demo`), so a long paper takes longer.

**Timeouts and the cheap resume.** Claude Code stops waiting for a tool after its MCP tool
timeout; start it with a longer one for `review_paper`:

```bash
MCP_TOOL_TIMEOUT=900000 claude
```

If a call still times out, call `review_paper` again on the same PDF: finished steps are
served from the local cache (`~/.cache/reviewer-zero`) and only the unfinished ones are paid
for.

## Tools

| Tool | What it does | Needs | Sends off your machine |
|---|---|---|---|
| `check_format(pdf_path, venue, stage)` | desk-reject risks: names, emails, self-citations, identifying URLs, PDF metadata, page limits, required sections | nothing | nothing |
| `check_citations(pdf_path)` | resolves every reference; flags unresolved, wrong year, wrong authors, arXiv-now-published, duplicates | index key | titles, DOIs and arXiv ids of the works you cite |
| `find_prior_work(queries, before, k)` | 40 candidates for one claim from 6–10 component queries, with BibTeX | index key | the queries Claude writes |
| `paper(key)`, `citations(key)` | one paper's details; one hop of its citation graph | index key | the paper key |
| `verify_quotes(pdf_path, quotes)` | checks each quote appears verbatim in your PDF, with its page | nothing | nothing |
| `review_paper(pdf_path, venue, confirm, novelty)` | the measured pipeline (above) | index key, `ANTHROPIC_API_KEY`, local GROBID | your paper's text to Anthropic under your key; queries and cited titles as above |

**Quota.** A key allows 600 units an hour and 5,000 a day. One `find_prior_work` call costs
about 17 units (6–10 searches plus two expansion calls), so about 35 calls an hour.

## Privacy

- Reviewer Zero's tools never send your PDF or its text anywhere, except to Anthropic under
  your own key in `review_paper`. (In the skill path, Claude itself reads the paper through
  your Claude client, as with any file you give it.)
- Only search queries, paper keys and the titles, DOIs and arXiv ids of works you cite go to
  our index; cited works' identifiers may also go to OpenAlex, Crossref and arXiv to resolve
  them.
- GROBID is used only at a local address (`localhost`); any other `GROBID_URL` is ignored.
- No telemetry. The index counts requests per key and stores no query text.
- `tests/test_mcp_privacy.py` runs every tool on a test paper and fails if any request goes to
  another host or carries a sentence of the paper's body.

## Measured

Dev-set numbers, measured on 2026-10-02 unless noted; details, caveats and suggested wording
in [`docs/MEASURED_CLAIMS.md`](https://github.com/rishab-ghosh/reviewer-zero/blob/main/docs/MEASURED_CLAIMS.md).

- **Prior-work recall@10, measured pipeline** (`review_paper`): 43.2% full, 36.5% lite, on 50
  dev cases with 74 reviewer-named prior works.
- **Prior-work recall, MCP path** (Claude writes the queries from the tool description, then
  ranks the 40 candidates): **25.7%** with Sonnet 5, **27.0%** with Opus 5.5
  (recall@10, same 50 cases; today's index). The gap to the pipeline is mostly retrieval: the
  pipeline reranks 150 candidates per claim, the tool returns 40. For the most thorough
  prior-work search, use `review_paper`.
- **Skill path, end to end** (Opus 5.5, 6 dev papers): about 4 minutes per paper; 184 of 188
  quotes Claude submitted were verified (it must fix or drop the rest); no score or verdict
  sentence in any review. Too few papers for a recall number.
- **`find_prior_work` latency**: median **5.2 s** for 6 queries against the hosted index
  (16 calls from Austin, TX, 4.7–9.3 s): about 2.4 s of searches, 2.3 s of citation-graph
  expansion and 0.3 s of metadata, with at most 4 index requests in flight (the per-key limit).
  On a local copy of the index the same call takes a median 1.6 s.
- **`check_citations` without GROBID**: 72.3% as many references resolved as with GROBID
  (30 dev papers).
- **Index**: about 1.09 million papers (577k arXiv papers in ML and adjacent fields since
  2017, plus works they cite and venue papers not on arXiv), 28.3 million citation edges,
  updated nightly.

## Licence

MIT (`LICENSE`). Runtime dependencies are permissively licensed: `mcp`, `anthropic`,
`pydantic`, `pydantic-settings`, `typer`, `pyyaml`, `pdfplumber` (MIT); `httpx`, `lxml`,
`pypdf`, `numpy` (BSD); the optional `docling` extra (MIT). GROBID, run separately as a
container, is Apache-2.0.

## Development

```bash
make setup        # uv sync, pre-commit install
make lint test    # ruff and pytest; tests never touch the network
```

The pipeline behind `review_paper` is in `src/reviewer/`: `parse/` (GROBID), `claims/`,
`novelty/`, `methods/`, `writing/`, `citations/`, `format/`, `meta/` (including the leak
filter that keeps scores and verdicts out of every output), `review/` (orchestration, budget)
and `mcp/` (the server and its tools). Prompts are versioned files next to each step.
