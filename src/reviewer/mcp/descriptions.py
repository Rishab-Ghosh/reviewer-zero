"""Tool descriptions. They are instructions to the calling model; edit them like prompts."""

PRIVACY = (
    "Privacy: runs on your machine. Your PDF and its text never leave it, except to Anthropic "
    "under your own API key when you call review_paper. What is sent: search queries and paper "
    "keys to the Reviewer Zero index (it counts requests per API key and stores nothing else), "
    "and, for check_citations, the titles, DOIs and arXiv ids of the works the paper cites, to "
    "the index and to OpenAlex, Crossref and arXiv. No telemetry."
)

FIND_PRIOR_WORK = f"""Find prior work that may already contain a claim of an ML paper.

Before calling: decompose ONE claim into 6 to 10 component queries, one per technique, \
objective, data-construction step or problem framing. Phrase each the way a prior paper's \
abstract would describe that component: a declarative sentence of 10 to 40 words, not a \
question, not naming this paper or its method name. Set `before` to the paper's submission \
date so later work is excluded. Call once per claim.

Returns up to `k` (default 40) candidates in a cheap first-stage order: title, year, venue, \
authors, abstract, link, seed_count, the queries that found it and a BibTeX entry. The order \
is not a judgement. You must read the abstracts and label each candidate yourself: same (it \
already contains the claim), close, builds on, or different, with a one-line reason; show \
same and close first. Never declare the paper novel or not novel overall.

{PRIVACY}"""

PAPER = f"""Look up one paper in the Reviewer Zero index by key: a bare arXiv id \
(2106.09685) or a key returned by find_prior_work (openalex:W..., s2:...). Returns title, \
authors, year, venue, DOI, arXiv id, abstract, a link and a BibTeX entry. Fails with a clear \
message if the key is unknown.

{PRIVACY}"""

CITATIONS = f"""List the papers one paper cites (direction "cites") or the papers in the \
Reviewer Zero index that cite it (direction "cited_by"), with metadata and BibTeX for each. \
Use it to follow a candidate from find_prior_work one hop. `limit` is 1 to 200 (default 50).

{PRIVACY}"""

CHECK_FORMAT = f"""Check a paper PDF against a venue's format and anonymization rules, \
locally and for free (no LLM, nothing sent anywhere). venue: iclr2027, neurips2026, \
icml2026, cvpr2027, acl or other; stage: submission, camera_ready or preprint. At \
submission, anonymization problems (author names or affiliations on page 1, emails, \
first-person self-citations, identifying URLs, PDF metadata) are violations. Page limits \
and required sections are checked only for venues whose rules are built (ICLR 2027, \
NeurIPS 2026, ICML 2026); for the others the result says so. Report every violation with \
its page and evidence; never rewrite the paper.

{PRIVACY}"""

CHECK_CITATIONS = f"""Check every reference of a paper PDF: resolve it (the Reviewer Zero \
index first, then OpenAlex, Crossref and arXiv) and flag references that cannot be found, \
have the wrong year or wrong authors, are cited as arXiv preprints although now published, \
or are listed twice. No LLM is used and no API key other than the index key is needed.

Sent off your machine: the titles, DOIs and arXiv ids of the works the paper cites (to the \
index and to OpenAlex, Crossref and arXiv) — never the paper's own text. The reference list \
is read by a local GROBID container if one is running, otherwise from the PDF's text layer; \
the result says which (`parser`). Report each flag with the reference as printed and what it \
resolved to; a reference that did not resolve is not proof it does not exist.

{PRIVACY}"""

REVIEW_PAPER = f"""Run the full Reviewer Zero review of a paper PDF (claims, prior work, \
citations with support checks, methodology, writing, format) with our prompts, using the \
user's own ANTHROPIC_API_KEY. It costs real money (usually about $1.50 to $2.00) and takes \
7 to 12 minutes.

ALWAYS call it first with confirm=false: that spends nothing and returns the estimate and the \
hard limit. Show both to the user and call again with confirm=true only after they agree. If a \
call times out, call again with the same PDF: finished steps are served from the cache and \
only what is left is paid for.

novelty="full" (default) sends the 150 best candidates per claim to the reranker; \
novelty="lite" sends 60 (quicker and cheaper, lower recall; see the README for the measured \
recall and time of each).

Report findings evidence-first, each with its quote. Never give a score, rating or \
accept/reject verdict, and never rewrite the user's text. Without ANTHROPIC_API_KEY or a \
local GROBID container it explains what is missing; check_format, check_citations, \
find_prior_work, paper and citations work without a key.

Sent off your machine by this tool: the paper's text to Anthropic under your own key; search \
queries and cited works' titles/DOIs/arXiv ids to the index, OpenAlex, Crossref and arXiv.

{PRIVACY}"""

VERIFY_QUOTES = f"""Check that quotes you intend to cite from a paper appear in its PDF word \
for word, and on which page. Use it on every quote before putting it in a review: a quote that \
is not verified must not be presented as the paper's text. Whitespace, line-break hyphenation, \
ligatures and punctuation are forgiven; added, dropped or changed words are not. Join separate \
fragments with "..." only when they appear in order, close together. Quotes need at least 5 \
words. Up to 50 quotes per call. Local only: nothing is sent anywhere.

{PRIVACY}"""
