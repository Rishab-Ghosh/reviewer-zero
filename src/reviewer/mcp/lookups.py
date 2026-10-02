"""paper(key) and citations(key): thin index lookups with BibTeX (MCP_PLAN §3.5). No LLM."""

from typing import Literal

from reviewer.mcp.bibtex import bibtex, unique_keys
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.prior_work import QueryError
from reviewer.models import CitationsResult, IndexPaperDetail, PaperRecord

MAX_LIMIT = 200


def _link(d: IndexPaperDetail) -> str | None:
    if d.arxiv_id:
        return f"https://arxiv.org/abs/{d.arxiv_id}"
    if d.doi:
        return f"https://doi.org/{d.doi}"
    return None


def record(d: IndexPaperDetail, cite_key: str | None = None) -> PaperRecord:
    return PaperRecord(**d.model_dump(), link=_link(d), bibtex=bibtex(d, cite_key))


def _check_key(key: str) -> str:
    key = key.strip()
    if not key or len(key) > 64 or any(c.isspace() for c in key):
        raise QueryError(
            "Pass one paper key: a bare arXiv id such as 2106.09685, or a key returned by "
            "find_prior_work (openalex:W..., s2:...)."
        )
    return key


def run_paper(index: HostedIndex, key: str) -> PaperRecord:
    key = _check_key(key)
    d = index.paper(key)
    if d is None:
        raise QueryError(f"No paper with key {key!r} in the Reviewer Zero index.")
    return record(d)


def run_citations(
    index: HostedIndex,
    key: str,
    direction: Literal["cites", "cited_by"] = "cites",
    limit: int = 50,
) -> CitationsResult:
    key = _check_key(key)
    if not 1 <= limit <= MAX_LIMIT:
        raise QueryError(f"limit must be between 1 and {MAX_LIMIT}.")
    neighbours = index.citations(key, direction, limit + 1)
    if neighbours is None:
        raise QueryError(f"No paper with key {key!r} in the Reviewer Zero index.")
    truncated = len(neighbours) > limit
    neighbours = neighbours[:limit]
    meta = index.papers([p.key for p in neighbours])
    details = [
        meta.get(p.key)
        or IndexPaperDetail(key=p.key, kind=p.kind, title=p.title, year=p.year, abstract=p.abstract)
        for p in neighbours
    ]
    keys = unique_keys(details)
    return CitationsResult(
        key=key,
        direction=direction,
        papers=[record(d, k) for d, k in zip(details, keys, strict=True)],
        truncated=truncated,
    )
