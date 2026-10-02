"""`reviewer parse` implementation. Owned by the parse workstream."""

import logging
from dataclasses import dataclass
from pathlib import Path

import typer

from reviewer.cache import sha256_file
from reviewer.config import PROJECT_ROOT, Settings, get_settings
from reviewer.corpus.arxiv_abstracts import ArxivAbstractClient
from reviewer.corpus.ids import is_arxiv_id, strip_version
from reviewer.failures import record_failure
from reviewer.models import ParsedPaper, Table
from reviewer.parse.contributions import find_contributions
from reviewer.parse.docling_tables import (
    DoclingError,
    DoclingTableExtractor,
    choose_tables,
    usable_tables,
)
from reviewer.parse.grobid_client import GrobidClient
from reviewer.parse.ids import paper_id_for
from reviewer.parse.pages import (
    PdfToTextError,
    count_chars,
    fallback_sections,
    page_coverage,
    pdftotext_pages,
)
from reviewer.parse.tei import UNATTRIBUTED_PAGE, page_texts, parse_tei

logger = logging.getLogger(__name__)


@dataclass
class ParseSummary:
    ok: int = 0
    skipped: int = 0
    failed: int = 0


def collect_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    return []


def display_path(pdf: Path) -> str:
    """Project-relative path when possible, so outputs don't embed a machine-specific root."""
    resolved = pdf.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def with_contributions(paper: ParsedPaper) -> ParsedPaper:
    contributions = find_contributions(paper.sections)
    report = paper.parse_report.model_copy(update={"has_contributions": contributions is not None})
    if contributions is None:
        report.warnings.append("no contributions statement found")
    return paper.model_copy(update={"contributions": contributions, "parse_report": report})


def with_pages(paper: ParsedPaper, tei: str, pdf: Path) -> ParsedPaper:
    """Per-page coverage (GROBID coordinates vs pdftotext) and pdftotext fallback sections for
    pages under 50%. Fallback sections are appended after GROBID's sections and marked."""
    report = paper.parse_report.model_copy(deep=True)
    try:
        pdf_pages = pdftotext_pages(pdf)
    except PdfToTextError as exc:
        report.warnings.append(f"page coverage unavailable: {exc}")
        return paper.model_copy(update={"parse_report": report})
    parse_pages = page_texts(tei)
    unattributed = count_chars(parse_pages.get(UNATTRIBUTED_PAGE, ""))
    if unattributed:
        report.warnings.append(f"{unattributed} parsed characters had no page coordinates")
    coverage = page_coverage(parse_pages, pdf_pages)
    fallback = fallback_sections(coverage, pdf_pages)
    if fallback:
        pages = ", ".join(str(s.fallback_page) for s in fallback)
        report.warnings.append(f"pages under 50% coverage, pdftotext fallback spliced in: {pages}")
    return paper.model_copy(
        update={
            "pages": coverage,
            "fallback_pages": [s.fallback_page for s in fallback if s.fallback_page is not None],
            "sections": [*paper.sections, *fallback],
            "parse_report": report,
        }
    )


def with_abstract(paper: ParsedPaper, arxiv_abstract: str | None) -> ParsedPaper:
    """arXiv's metadata abstract is the source of truth for arXiv papers; GROBID's is kept in
    `grobid_abstract`. Papers without an arXiv id (or a failed lookup) keep GROBID's."""
    if arxiv_abstract:
        return paper.model_copy(
            update={
                "abstract": arxiv_abstract,
                "abstract_source": "arxiv",
                "grobid_abstract": paper.abstract,
            }
        )
    if is_arxiv_id(paper.paper_id):
        report = paper.parse_report.model_copy(deep=True)
        report.warnings.append("arXiv abstract unavailable; using GROBID's abstract")
        return paper.model_copy(update={"parse_report": report})
    return paper


def with_tables(paper: ParsedPaper, docling: list[Table] | None) -> ParsedPaper:
    """Docling's tables replace GROBID's when Docling has more usable tables; otherwise GROBID's
    tables stay as they are (caption-only ones included)."""
    chosen, source = choose_tables(paper.tables, docling)
    tables = chosen if source == "docling" else paper.tables
    report = paper.parse_report.model_copy(update={"n_tables": len(tables)})
    return paper.model_copy(
        update={"tables": tables, "table_source": source, "parse_report": report}
    )


def parse_pdf(
    client: GrobidClient,
    pdf: Path,
    paper_id: str,
    sha: str,
    arxiv_abstract: str | None = None,
    docling: list[Table] | None = None,
) -> ParsedPaper:
    tei = client.process(pdf)
    paper = parse_tei(tei, paper_id=paper_id, source_path=display_path(pdf), pdf_sha256=sha)
    paper = with_contributions(paper)
    paper = with_pages(paper, tei, pdf)
    paper = with_abstract(paper, arxiv_abstract)
    return with_tables(paper, docling)


def docling_tables_or_none(
    extractor: DoclingTableExtractor, pdf: Path, paper_id: str, settings: Settings
) -> list[Table] | None:
    if not settings.parse_docling_tables:
        return None
    try:
        return extractor.extract(pdf)
    except DoclingError as exc:
        record_failure(settings.failures_file, "parse", paper_id, f"Docling tables: {exc}")
        return None


def run_parse(path: Path, force: bool = False, settings: Settings | None = None) -> None:
    """Parse one PDF or every PDF in a directory into data/parsed/<paper_id>.json."""
    settings = settings or get_settings()
    pdfs = collect_pdfs(path)
    if not pdfs:
        logger.error("no PDFs found at %s", path)
        raise typer.Exit(1)

    client = GrobidClient(settings)
    if not client.is_alive():
        logger.error(
            "GROBID is not reachable at %s (try `make grobid-up`); aborting parse",
            settings.grobid_url,
        )
        client.close()
        raise typer.Exit(1)

    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    summary = ParseSummary()
    arxiv_ids = [strip_version(p.stem) for p in pdfs if is_arxiv_id(p.stem)]
    abstracts = ArxivAbstractClient(settings).abstracts(arxiv_ids) if arxiv_ids else {}
    logger.info("arXiv abstracts: %d/%d", len(abstracts), len(arxiv_ids))
    extractor = DoclingTableExtractor(settings)
    try:
        for pdf in pdfs:
            paper_id = pdf.stem
            try:
                sha = sha256_file(pdf)
                paper_id = paper_id_for(pdf, sha)
                out = settings.parsed_dir / f"{paper_id}.json"
                if out.exists() and not force:
                    logger.info("skip %s (already parsed)", paper_id)
                    summary.skipped += 1
                    continue
                if is_arxiv_id(paper_id) and paper_id not in abstracts:
                    record_failure(
                        settings.failures_file,
                        "parse",
                        paper_id,
                        "arXiv abstract unavailable (API lookup failed); used GROBID's abstract",
                    )
                docling = docling_tables_or_none(extractor, pdf, paper_id, settings)
                paper = parse_pdf(
                    client, pdf, paper_id, sha, abstracts.get(paper_id), docling=docling
                )
                out.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
                r = paper.parse_report
                logger.info(
                    "parsed %s: %d sections, %d refs, %d citations, contributions=%s, "
                    "tables=%d (%s, usable %d), abstract=%s, fallback pages=%s",
                    paper_id,
                    r.n_sections,
                    r.n_references,
                    r.n_in_text_citations,
                    r.has_contributions,
                    len(paper.tables),
                    paper.table_source,
                    len(usable_tables(paper.tables)),
                    paper.abstract_source,
                    paper.fallback_pages or "none",
                )
                summary.ok += 1
            except Exception as exc:
                record_failure(
                    settings.failures_file, "parse", paper_id, f"{type(exc).__name__}: {exc}"
                )
                summary.failed += 1
    finally:
        client.close()

    logger.info(
        "parse done: ok=%d skipped=%d failed=%d", summary.ok, summary.skipped, summary.failed
    )
    typer.echo(f"parse: {summary.ok} ok, {summary.skipped} skipped, {summary.failed} failed")
