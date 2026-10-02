"""`reviewer fetch` implementation. Owned by the corpus workstream."""

import logging

from reviewer.config import Settings, get_settings
from reviewer.corpus.arxiv_client import ArxivClient, ArxivError
from reviewer.corpus.corpus import load_corpus
from reviewer.corpus.ids import strip_version
from reviewer.failures import record_failure
from reviewer.models import PaperTier

logger = logging.getLogger(__name__)


def fetch_ids(ids: list[str], settings: Settings, client: ArxivClient) -> tuple[int, int, int]:
    """Download each id; returns (fetched, skipped, failed). Failures are recorded, not raised."""
    fetched = skipped = failed = 0
    for arxiv_id in ids:
        dest = settings.arxiv_dir / f"{arxiv_id}.pdf"
        existed = dest.exists()
        try:
            client.download_pdf(arxiv_id, dest)
        except ArxivError as e:
            record_failure(settings.failures_file, "fetch", arxiv_id, str(e))
            failed += 1
            continue
        if existed:
            skipped += 1
        else:
            fetched += 1
    return fetched, skipped, failed


def run_fetch(ids: list[str] | None = None) -> None:
    """Download corpus PDFs from arXiv (all of data/corpus.yaml, or just `ids`)."""
    settings = get_settings()
    if ids:
        targets = [strip_version(i) for i in ids]
    else:
        targets = [e.paper_id for e in load_corpus(settings) if e.tier != PaperTier.own]
    with ArxivClient(settings) as client:
        fetched, skipped, failed = fetch_ids(targets, settings, client)
    logger.info(
        "fetch done: %d requested, %d fetched, %d already present, %d failed",
        len(targets),
        fetched,
        skipped,
        failed,
    )
