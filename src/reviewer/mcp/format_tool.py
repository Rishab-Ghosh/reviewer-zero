"""check_format: the existing format checker on a local PDF (MCP_PLAN §3.3). No LLM, no network
except a local GROBID container, if one answers, for author names."""

import logging
from pathlib import Path
from typing import Literal

from reviewer.cache import sha256_file
from reviewer.format.checks import EDITION_YEAR
from reviewer.format.commands import check_pdf
from reviewer.format.layout import declared_venue
from reviewer.format.venues import VENUES
from reviewer.mcp.config import McpConfig
from reviewer.mcp.local_parse import grobid_paper
from reviewer.mcp.prior_work import QueryError
from reviewer.models import DeclaredVenue, FormatCheckResult, SubmissionStage, Venue

logger = logging.getLogger(__name__)

VenueId = Literal["iclr2027", "neurips2026", "icml2026", "cvpr2027", "acl", "other"]
StageId = Literal["submission", "camera_ready", "preprint"]
BUILT: dict[str, Venue] = {
    "iclr2027": Venue.iclr,
    "neurips2026": Venue.neurips,
    "icml2026": Venue.icml,
}
NOT_BUILT_NOTE = {
    "cvpr2027": "CVPR 2027 page and section rules aren't built yet. Every other check runs.",
    "acl": "Format rules for this venue aren't built yet. Every other check runs.",
    "other": "Format rules for this venue aren't built yet. Every other check runs.",
}


def local_pdf(pdf_path: str) -> Path:
    p = Path(pdf_path).expanduser()
    if not p.is_file():
        raise QueryError(f"No file at {pdf_path!r}. Pass the full path of the PDF.")
    with p.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise QueryError(f"{p.name} is not a PDF.")
    return p


def run_check_format(
    pdf_path: str, venue: VenueId, stage: StageId, config: McpConfig | None = None
) -> FormatCheckResult:
    config = config or McpConfig()
    pdf = local_pdf(pdf_path)
    if venue not in BUILT and venue not in NOT_BUILT_NOTE:
        raise QueryError(
            "venue must be one of iclr2027, neurips2026, icml2026, cvpr2027, acl, other."
        )
    stage_enum = SubmissionStage(stage)
    notes: list[str] = []
    paper = grobid_paper(pdf, config)
    names = [a.name for a in paper.authors] if paper else []
    if paper is None:
        notes.append(
            "GROBID is not running, so author names on page 1 were not checked; affiliation, "
            "email, self-citation, URL and metadata checks ran. For name checks: "
            "docker run --rm -p 8070:8070 grobid/grobid:0.9.1-crf"
        )
    if venue in BUILT:
        v = BUILT[venue]
        declared = DeclaredVenue(venue=v, stage=stage_enum, year=EDITION_YEAR[v])
        rules_mode: Literal["declared", "none"] = "declared"
        rules = VENUES[v].edition
    else:
        declared = DeclaredVenue(venue=None, stage=stage_enum)
        rules_mode = "none"
        rules = NOT_BUILT_NOTE[venue]
        notes.append(rules)
    report, doc = check_pdf(
        pdf,
        paper_id=sha256_file(pdf)[:16],
        title=(paper.title if paper else "") or "",
        private=True,
        author_names=names,
        declared=declared,
        rules_mode=rules_mode,
    )
    detected = declared_venue(doc.pages[0] if doc.pages else "")
    if detected.venue is not None and (venue not in BUILT or detected.venue != BUILT[venue]):
        notes.append(
            f"Page 1 declares {detected.text!r}, which does not match the venue you chose "
            f"({venue}). Check you are using the right style file."
        )
    vp = next((p for p in report.pages if venue in BUILT and p.venue == BUILT[venue]), None)
    limit = None
    if vp is not None:
        limit = (
            vp.submission_limit
            if stage_enum == SubmissionStage.submission
            else vp.camera_ready_limit
        )
    return FormatCheckResult(
        venue=venue,
        stage=stage_enum,
        rules=rules,
        text_engine=doc.engine,
        author_names_from="grobid" if paper else "none",
        detected_header=detected,
        n_pages=report.n_pages,
        main_text_pages=vp.main_text_pages if vp else None,
        page_limit=limit,
        findings=report.findings,
        notes=notes,
    )
