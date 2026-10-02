"""check_citations: resolve and flag every reference (MCP_PLAN §3.2). No LLM."""

import logging

from reviewer.cache import sha256_file
from reviewer.citations.commands import Checker
from reviewer.citations.fallback_refs import fallback_references
from reviewer.format.pdf import load_pdf
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import local_pdf
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.local_parse import grobid_paper, paper_settings
from reviewer.models import CitationCheckResult, CitationRefResult, ParsedPaper, ReferenceCheck

logger = logging.getLogger(__name__)

RAW_CHARS = 400
KEY_PROBE = "1706.03762"
"""A paper every index has: one 1-unit lookup that surfaces a missing or bad key up front,
instead of every reference silently failing to resolve through the index."""


def _ref_result(c: ReferenceCheck) -> CitationRefResult:
    r = c.resolved
    years = [v.year for v in (r.versions if r else []) if v.year]
    link = None
    if r and r.arxiv_id:
        link = f"https://arxiv.org/abs/{r.arxiv_id}"
    elif r and r.doi:
        link = f"https://doi.org/{r.doi}"
    return CitationRefResult(
        ref_id=c.ref_id,
        raw_string=c.raw_string[:RAW_CHARS],
        resolved_title=r.title if r else None,
        resolved_year=min(years) if years else None,
        arxiv_id=r.arxiv_id if r else None,
        doi=r.doi if r else None,
        link=link,
        source=str(r.source) if r else None,
        flags=[str(f.type) for f in c.flags],
    )


def run_check_citations(pdf_path: str, config: McpConfig | None = None) -> CitationCheckResult:
    config = config or McpConfig()
    pdf = local_pdf(pdf_path)
    HostedIndex(config).paper(KEY_PROBE)  # raises IndexAuthError / IndexRateLimitedError early
    sha = sha256_file(pdf)
    notes: list[str] = []
    paper = grobid_paper(pdf, config)
    engine = None
    if paper is not None and paper.references:
        parser = "grobid"
    else:
        doc = load_pdf(pdf)
        engine = doc.engine
        parser = "fallback"
        refs = fallback_references(doc.pages)
        paper = ParsedPaper(
            paper_id=f"pdf-{sha[:16]}",
            source_path=str(pdf),
            pdf_sha256=sha,
            title="",
            references=refs,
        )
        notes.append(
            "GROBID is not running, so the reference list was split from the PDF's text layer, "
            "which finds fewer references than GROBID (the README gives the measured rates). For "
            "best results: docker run --rm -p 8070:8070 grobid/grobid:0.9.1-crf"
        )
    if not paper.references:
        notes.append("No reference list was found in this PDF.")
    checker = Checker(paper_settings(config, sha))
    report = checker.check(paper, support=False)
    if not report.index_available:
        notes.append(
            "The index was not reachable; references were resolved via OpenAlex, "
            "Crossref and arXiv only."
        )
    refs = [_ref_result(c) for c in report.references]
    return CitationCheckResult(
        parser=parser,
        text_engine=engine,
        n_references=len(refs),
        n_resolved=sum(r.resolved_title is not None for r in refs),
        index_available=report.index_available,
        flags=[f for c in report.references for f in c.flags],
        references=refs,
        notes=notes,
    )
