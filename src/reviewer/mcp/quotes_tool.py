"""verify_quotes: are these quotes in the PDF, word for word, and where? Local, no LLM.

Reuses the pipeline's verifier (`methods/quotes.py`: squash, fragments, MIN_QUOTE_WORDS), so a
quote the skill verifies is verified in exactly the sense the pipeline's checkers use.
"""

from reviewer.format.pdf import load_pdf
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import local_pdf
from reviewer.mcp.local_parse import grobid_paper
from reviewer.mcp.prior_work import QueryError
from reviewer.methods.quotes import MIN_QUOTE_WORDS, WORD_RE, QuoteIndex
from reviewer.models import VerifiedQuote, VerifyQuotesResult

MAX_QUOTES = 50
MAX_QUOTE_CHARS = 1000


def run_verify_quotes(
    pdf_path: str, quotes: list[str], config: McpConfig | None = None
) -> VerifyQuotesResult:
    config = config or McpConfig()
    pdf = local_pdf(pdf_path)
    if not quotes or len(quotes) > MAX_QUOTES:
        raise QueryError(f"Pass 1 to {MAX_QUOTES} quotes.")
    if any(len(q) > MAX_QUOTE_CHARS for q in quotes):
        raise QueryError(f"Each quote must be at most {MAX_QUOTE_CHARS} characters.")
    doc = load_pdf(pdf)
    pages = [QuoteIndex(t) for t in doc.pages]
    pairs = [QuoteIndex(a + "\n" + b) for a, b in zip(doc.pages, doc.pages[1:], strict=False)]
    paper = grobid_paper(pdf, config)
    whole = QuoteIndex.for_paper(paper) if paper else None
    out = []
    for q in quotes:
        if len(WORD_RE.findall(q)) < MIN_QUOTE_WORDS:
            out.append(VerifiedQuote(quote=q, status="too_short"))
            continue
        hit = next((i for i, idx in enumerate(pages, 1) if idx.verify(q)), None)
        if hit is not None:
            out.append(VerifiedQuote(quote=q, status="verified", pages=[hit]))
            continue
        pair = next((i for i, idx in enumerate(pairs, 1) if idx.verify(q)), None)
        if pair is not None:
            out.append(VerifiedQuote(quote=q, status="verified", pages=[pair, pair + 1]))
        elif whole is not None and whole.verify(q):
            out.append(VerifiedQuote(quote=q, status="verified"))
        else:
            out.append(VerifiedQuote(quote=q, status="not_found"))
    return VerifyQuotesResult(
        quotes=out, n_verified=sum(c.status == "verified" for c in out), text_engine=doc.engine
    )
