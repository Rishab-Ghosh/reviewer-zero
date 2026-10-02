"""Synthetic builders shared by the eval tests (no network, no real outputs)."""

from datetime import UTC, datetime

from reviewer.models import (
    Claim,
    CorpusEntry,
    CriticNote,
    ExtractionMetadata,
    Failure,
    GoldClaim,
    GoldClaims,
    PaperClaims,
    PaperTier,
    ParsedPaper,
    TokenUsage,
)

TS = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def claim(statement: str, result: str = "28.4 BLEU", ctype: str = "new_method") -> Claim:
    return Claim(
        statement=statement,
        contribution_type=ctype,
        method="Transformer: attention-only encoder-decoder",
        problem="machine translation",
        setting="WMT14 En-De",
        result=result,
        novelty_assertion="first model built only on attention",
        evidence=["Table 2", "Section 6.1"],
        confidence=0.9,
    )


def paper_claims(paper_id: str, statements: list[str], result: str = "28.4 BLEU") -> PaperClaims:
    return PaperClaims(
        paper_id=paper_id,
        title=f"Title of {paper_id}",
        one_line_summary="A paper that does a thing.",
        claims=[claim(s, result) for s in statements],
        keywords=["a", "b", "c", "d", "e"],
        extraction_metadata=ExtractionMetadata(
            model="claude-sonnet-5",
            prompt_version="extract_v1+critic_v1",
            timestamp=TS,
            token_usage=TokenUsage(input_tokens=1000, output_tokens=200, cost_usd=0.004),
            critic_review=[
                CriticNote(draft_index=0, final_index=0, verdict="rewritten", reason="was vague")
            ],
        ),
    )


def parsed(
    paper_id: str, abstract: str = "We reach 28.4 BLEU.", contributions: str | None = None
) -> ParsedPaper:
    return ParsedPaper(
        paper_id=paper_id,
        source_path=f"data/papers/{paper_id}.pdf",
        title=f"Title of {paper_id}",
        abstract=abstract,
        contributions=contributions,
    )


def entry(paper_id: str, tier: str) -> CorpusEntry:
    return CorpusEntry(paper_id=paper_id, tier=PaperTier(tier), reason=f"reason {paper_id}")


def gold(
    paper_id: str,
    statements: list[str],
    source: str = "author",
    results: list[str | None] | None = None,
) -> GoldClaims:
    results = results or [None] * len(statements)
    return GoldClaims(
        paper_id=paper_id,
        source=source,
        author="tester",
        claims=[GoldClaim(statement=s, result=r) for s, r in zip(statements, results, strict=True)],
    )


def failure(paper_id: str, stage: str, error: str, ts: datetime = TS) -> Failure:
    return Failure(stage=stage, paper_id=paper_id, error=error, timestamp=ts)
