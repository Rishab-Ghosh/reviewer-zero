"""GROBID on the user's machine, if one answers (MCP_PLAN §3.2–3.4).

The PDF may go only to a GROBID at a loopback address: a GROBID_URL pointing anywhere else is
ignored, so the "never leaves your machine" promise holds. TEI is cached per PDF (sha256) under
REVIEWER_ZERO_CACHE/papers/<sha>/grobid, and a cached TEI is used without contacting GROBID.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from reviewer.cache import sha256_file
from reviewer.config import Settings
from reviewer.mcp.config import McpConfig
from reviewer.models import ParsedPaper
from reviewer.parse.grobid_client import GrobidClient, grobid_cache_key
from reviewer.parse.tei import parse_tei

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def is_local_url(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOOPBACK_HOSTS


def paper_settings(config: McpConfig, sha: str) -> Settings:
    base = config.pipeline_settings()
    return base.model_copy(update={"paper_cache_dir": config.cache / "papers" / sha})


def grobid_paper(pdf: Path, config: McpConfig) -> ParsedPaper | None:
    """ParsedPaper from a cached or local GROBID parse; None if neither is available."""
    sha = sha256_file(pdf)
    settings = paper_settings(config, sha)
    if not is_local_url(settings.grobid_url):
        logger.warning("GROBID_URL is not a loopback address; not sending the PDF there")
        return None
    client = GrobidClient(settings)
    try:
        tei = client.cache.get_text(grobid_cache_key(sha))
        if tei is None:
            if not client.is_alive():
                return None
            tei = client.process(pdf)
    except (httpx.HTTPError, RuntimeError, OSError) as e:
        logger.warning("GROBID parse failed: %s", type(e).__name__)
        return None
    finally:
        client.close()
    return parse_tei(tei, paper_id=f"pdf-{sha[:16]}", source_path=str(pdf), pdf_sha256=sha)
