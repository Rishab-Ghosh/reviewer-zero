from datetime import UTC, datetime

from reviewer.eval.compare import ComparisonRow, render_comparison
from reviewer.models import Claim, ExtractionMetadata, PaperClaims, TokenUsage


def _pc(pid: str, statements: list[str], model: str) -> PaperClaims:
    claims = [
        Claim(
            statement=s,
            contribution_type="new_method",
            method="M",
            problem="P",
            setting="S | pipes",
            result="R",
            novelty_assertion="N",
            evidence=["Table 1"],
            confidence=0.8,
        )
        for s in statements
    ]
    return PaperClaims(
        paper_id=pid,
        title="T",
        one_line_summary="summary",
        claims=claims,
        keywords=list("abcde"),
        extraction_metadata=ExtractionMetadata(
            model=model,
            prompt_version="v",
            timestamp=datetime.now(UTC),
            token_usage=TokenUsage(input_tokens=10, output_tokens=5, cost_usd=0.01),
        ),
    )


def test_render_comparison_side_by_side_and_missing():
    rows = [
        ComparisonRow(
            paper_id="p1",
            title="Paper One",
            paper=None,
            default=_pc("p1", ["A improves performance.", "B got 3.5 BLEU."], "small"),
            strong=_pc("p1", ["A2 gets 1.2 points.", "B2", "C2"], "big"),
        ),
        ComparisonRow(paper_id="p2", title="Paper Two", paper=None, default=None, strong=None),
    ]
    md = render_comparison(rows, "small", "big", datetime(2026, 9, 24, tzinfo=UTC))
    assert "| # | small | big |" in md
    assert "**A improves performance.**" in md and "**C2**" in md
    assert "S \\| pipes" in md  # table-safe escaping
    assert "No claims from `small`" in md
    assert "$0.0100" in md
