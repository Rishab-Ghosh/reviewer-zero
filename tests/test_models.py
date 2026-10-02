from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reviewer.cache import DiskCache, content_key
from reviewer.models import Claim, ClaimsDraft, ExtractionMetadata, PaperClaims, TokenUsage


def _claim(i: int = 0) -> Claim:
    return Claim(
        statement=f"claim {i}",
        contribution_type="new_method",
        method="m",
        problem="p",
        setting="s",
        result="r",
        novelty_assertion="n",
        evidence=["Table 1"],
        confidence=0.9,
    )


def _meta() -> ExtractionMetadata:
    return ExtractionMetadata(
        model="x", prompt_version="v1", timestamp=datetime.now(UTC), token_usage=TokenUsage()
    )


@pytest.mark.parametrize("n", [0, 5])
def test_paper_claims_rejects_wrong_claim_count(n):
    with pytest.raises(ValidationError):
        PaperClaims(
            paper_id="a",
            title="t",
            one_line_summary="s",
            claims=[_claim(i) for i in range(n)],
            keywords=list("abcde"),
            extraction_metadata=_meta(),
        )


def test_paper_claims_accepts_three():
    pc = PaperClaims(
        paper_id="a",
        title="t",
        one_line_summary="s",
        claims=[_claim(i) for i in range(3)],
        keywords=list("abcdef"),
        extraction_metadata=_meta(),
    )
    assert len(pc.claims) == 3


def test_docstrings_reach_json_schema():
    schema = ClaimsDraft.model_json_schema()
    statement = schema["$defs"]["Claim"]["properties"]["statement"]
    assert "search for prior work" in " ".join(statement["description"].split())


def test_content_key_is_boundary_sensitive(tmp_path):
    assert content_key("ab", "c") != content_key("a", "bc")
    cache = DiskCache(tmp_path, "ns", ".json")
    cache.put_json("k", {"a": 1})
    assert cache.get_json("k") == {"a": 1}
    assert cache.get_json("missing") is None


def test_paper_claims_empty_only_when_verification_dropped_all():
    from reviewer.models import VerificationRecord

    meta = _meta().model_copy(
        update={"verification": [VerificationRecord(input_index=0, outcome="dropped", reason="x")]}
    )
    pc = PaperClaims(
        paper_id="a",
        title="t",
        one_line_summary="s",
        claims=[],
        keywords=list("abcde"),
        extraction_metadata=meta,
    )
    assert pc.claims == []
