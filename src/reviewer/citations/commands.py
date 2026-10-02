"""`reviewer citations` implementation."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import typer

from reviewer.citations import raw as rawlib
from reviewer.citations.arxiv_meta import ArxivMetaClient, ArxivRecord
from reviewer.citations.checks import run_checks
from reviewer.citations.http import SourceError
from reviewer.citations.report import flag_counts, render_report, verdict_counts
from reviewer.citations.resolve import Resolution, Resolver, build_resolved, matching_arxiv
from reviewer.citations.sources import CrossrefClient, IndexClient, OpenAlexClient
from reviewer.citations.support import judge_sentence
from reviewer.claims.prompts import load_prompt
from reviewer.config import Settings, get_settings
from reviewer.failures import record_failure
from reviewer.llm.client import LLMClient
from reviewer.models import CitationReport, ParsedPaper, ReferenceCheck, TokenUsage

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def is_private(paper: ParsedPaper) -> bool:
    src = paper.source_path.replace("\\", "/")
    return paper.paper_id.startswith(("own-", "or-")) or "/papers/own/" in src or "/private/" in src


def report_dir(settings: Settings, private: bool) -> Path:
    return (settings.private_reports_dir if private else settings.reports_dir) / "citations"


def parsed_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.json") if not p.name.endswith(".pass1.json"))
    return [path]


class Checker:
    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.index = IndexClient(settings)
        self.openalex = OpenAlexClient(settings)
        self.crossref = CrossrefClient(settings)
        self.arxiv = ArxivMetaClient(settings)
        self.resolver = Resolver(self.index, self.openalex, self.crossref, self.arxiv)
        self.llm = llm or LLMClient(settings)
        self.prompt = load_prompt(settings.citation_prompt, PROMPTS_DIR)
        self.openalex.probe_budget()

    @staticmethod
    def _year_mismatch(res: Resolution, arxiv_recs: dict[str, ArxivRecord]) -> bool:
        cited = set(rawlib.cited_years(res.ref.raw_string))
        if not res.records or not cited:
            return False
        known = {r.version.year for r, _ in res.records if r.version.year}
        known |= {y for a in res.arxiv_ids if (y := rawlib.arxiv_year(a))}
        arxiv = matching_arxiv(res, arxiv_recs)
        if arxiv:
            known.add(arxiv.published.year)
        return not (cited & known)

    def _more_versions(self, res: Resolution, arxiv_recs: dict[str, ArxivRecord]) -> None:
        mismatch = self._year_mismatch(res, arxiv_recs)
        if mismatch or rawlib.cites_arxiv_only(res.ref.raw_string):
            self.resolver.find_versions(
                res, matching_arxiv(res, arxiv_recs), year_mismatch=mismatch
            )

    def _crossref_doi_versions(self, resolutions: list[Resolution]) -> None:
        """Every venue DOI also gets Crossref's record (free, cached) as a version: it names
        venues OpenAlex left unnamed and carries the issued year (e.g. NeurIPS proceedings
        issued the year after the conference)."""
        for res in resolutions:
            dois = {r.doi for r, _ in res.records if r.doi and not r.version.is_preprint}
            for doi in sorted(dois):
                try:
                    found = self.crossref.by_doi(doi)
                except SourceError as e:
                    res.errors.append(f"crossref doi:{doi}: {e}")
                    continue
                for rec in found:
                    v = rec.version
                    if v.is_preprint or rawlib.title_similarity(res.ref.raw_string, v.title) < 0.8:
                        continue
                    for other, _ in res.records:
                        if other.doi == doi and (other.version.venue or "").startswith("doi:"):
                            other.version.venue = v.venue or other.version.venue
                    res.records.append((rec, rawlib.title_similarity(res.ref.raw_string, v.title)))

    def check(self, paper: ParsedPaper, support: bool = True) -> CitationReport:
        """Resolve and flag every reference; with `support`, also judge each citing sentence
        against the cited abstract (LLM). `support=False` makes no LLM call (MCP
        check_citations)."""
        s = self.settings
        logger.info("%s: resolving %d references", paper.paper_id, len(paper.references))
        with ThreadPoolExecutor(s.citation_ref_workers) as ex:
            resolutions = list(ex.map(self.resolver.resolve, paper.references))
        no_abstract = [
            r for r in resolutions if r.records and not any(x.abstract for x, _ in r.records)
        ]
        with ThreadPoolExecutor(s.citation_ref_workers) as ex:
            list(ex.map(self.resolver.find_arxiv_version, no_abstract))
        arxiv_ids = [a for r in resolutions for a in r.arxiv_ids]
        arxiv_recs = self.arxiv.lookup(arxiv_ids) if arxiv_ids else {}
        with ThreadPoolExecutor(s.citation_ref_workers) as ex:
            list(ex.map(lambda r: self._more_versions(r, arxiv_recs), resolutions))

        self._crossref_doi_versions(resolutions)
        checks = [
            ReferenceCheck(
                ref_id=r.ref.ref_id,
                raw_string=r.ref.raw_string,
                cited_years=rawlib.cited_years(r.ref.raw_string),
                cited_as_arxiv_only=rawlib.cites_arxiv_only(r.ref.raw_string),
                resolved=build_resolved(r, arxiv_recs, s.citation_abstract_chars),
                attempts=r.attempts,
            )
            for r in resolutions
        ]
        run_checks(checks, paper.paper_id)
        errors = [f"{r.ref.ref_id} {e}" for r in resolutions for e in r.errors]

        by_id = {c.ref_id: c for c in checks}
        sentences = paper.in_text_citations if support else []
        logger.info("%s: judging %d citing sentences", paper.paper_id, len(sentences))
        with ThreadPoolExecutor(s.citation_llm_workers) as ex:
            results = list(
                ex.map(
                    lambda c: judge_sentence(
                        c, by_id, paper.title, self.llm, self.prompt, s.llm_model
                    ),
                    sentences,
                )
            )
        usage = TokenUsage()
        support = []
        for res in results:
            support += res.judgments
            usage += res.usage
            if res.error:
                errors.append(f"support judge: {res.error}")
        return CitationReport(
            paper_id=paper.paper_id,
            title=paper.title,
            private=is_private(paper),
            generated=datetime.now(UTC),
            index_available=self.index.available,
            model=s.llm_model,
            prompt_version=self.prompt.version,
            references=checks,
            support=support,
            token_usage=usage,
            errors=errors,
        )


def write_report(report: CitationReport, settings: Settings) -> Path:
    out_dir = report_dir(settings, report.private)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{report.paper_id}.md"
    for path, text in (
        (md, render_report(report)),
        (md.with_suffix(".json"), report.model_dump_json(indent=2) + "\n"),
    ):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return md


def run_citations(path: Path, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    files = parsed_files(path)
    if not files:
        raise typer.BadParameter(f"no parsed papers at {path}")
    checker = Checker(settings)
    logger.info(
        "reviewer-index at %s: %s",
        settings.index_url,
        "up" if checker.index.available else "not running, skipped",
    )
    for f in files:
        try:
            paper = ParsedPaper.model_validate_json(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            record_failure(settings.failures_file, "citations", f.stem, f"unreadable parse: {e}")
            continue
        try:
            report = checker.check(paper)
        except (OSError, ValueError, SourceError) as e:
            record_failure(
                settings.failures_file, "citations", paper.paper_id, f"{type(e).__name__}: {e}"
            )
            continue
        if report.errors:
            record_failure(
                settings.failures_file,
                "citations",
                paper.paper_id,
                f"{len(report.errors)} lookup/judge error(s); first: {report.errors[0]}",
            )
        out = write_report(report, settings)
        flags = flag_counts(report)
        verdicts = verdict_counts(report)
        typer.echo(
            f"{paper.paper_id}: {sum(1 for r in report.references if r.resolved)}/"
            f"{len(report.references)} resolved; flags "
            + ", ".join(f"{t.value}={n}" for t, n in sorted(flags.items()))
            + "; support "
            + ", ".join(f"{v.value}={n}" for v, n in sorted(verdicts.items()))
            + f"; ${report.token_usage.cost_usd:.2f} -> {out}"
        )
    oa = checker.openalex
    typer.echo(
        f"OpenAlex: {oa.n_searches} paid title searches this run; remaining budget "
        f"{'unknown' if oa.remaining_usd is None else f'${oa.remaining_usd:.3f}'}"
        + (" (floor reached; later searches skipped)" if oa.budget_exhausted else "")
    )


def run_summary(seed: int, n_sample: int, n_not_in: int, settings: Settings | None = None) -> None:
    """Write SUMMARY.md next to the public and (separately) the private citation reports."""
    from reviewer.citations.summary import load_reports, render_summary

    settings = settings or get_settings()
    for private in (False, True):
        directory = report_dir(settings, private)
        reports = load_reports(directory) if directory.is_dir() else []
        if not reports:
            continue
        out = directory / "SUMMARY.md"
        out.write_text(render_summary(reports, seed, n_sample, n_not_in), encoding="utf-8")
        typer.echo(f"{len(reports)} report(s) -> {out}")
