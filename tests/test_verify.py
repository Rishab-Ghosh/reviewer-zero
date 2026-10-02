import pytest

from reviewer.claims.extractor import extract_claims
from reviewer.claims.prompts import load_prompt
from reviewer.claims.verify import normalize, quote_in_paper, verification_problems, verify_claims
from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    Claim,
    ClaimsDraft,
    ClaimVerdict,
    ClaimVerification,
    CriticNote,
    CriticOutput,
    ParsedPaper,
    Section,
    SupportItem,
    Table,
    TokenUsage,
)

PAPER = (
    "## 4 Results\nOur method reaches 88.3% accuracy on AMC23, while the baseline reaches "
    "84.1%.\n### Table 2: Ablation\nVariant | Acc | Len\nCertainty-guided | 88.3 | 3406\n"
    "p=0.5 | 87.9 | 3266"
)
NORM = normalize(PAPER)


def _claim(statement: str, result: str = "r") -> Claim:
    return Claim(
        statement=statement,
        contribution_type="new_method",
        method="m",
        problem="p",
        setting="AMC23",
        result=result,
        novelty_assertion="n",
        evidence=["Table 2"],
        confidence=0.8,
    )


def _item(element: str, quote: str, field: str = "statement") -> SupportItem:
    return SupportItem(field=field, element=element, quote=quote, source="Table 2")


def test_quote_matching():
    assert quote_in_paper("Our  method reaches 88.3% accuracy on AMC23", NORM)
    assert quote_in_paper("Certainty-guided | 88.3 | 3406", NORM)
    assert quote_in_paper("Our method reaches ... on AMC23", NORM)  # ellipsis, in order
    assert not quote_in_paper("on AMC23 ... Our method reaches", NORM)  # wrong order
    assert not quote_in_paper("Our method reaches 90% accuracy", NORM)
    assert not quote_in_paper("88.3", NORM)  # too short to be a quote
    assert quote_in_paper("Our method reaches “88.3%”".replace("“", "").replace("”", ""), NORM)


def test_problems_cover_every_number_with_a_verified_quote():
    claim = _claim("Reaches 88.3% vs 84.1% and length 3406")
    ok = ClaimVerification(
        items=[
            _item("88.3%", "Our method reaches 88.3% accuracy on AMC23"),
            _item("84.1%", "while the baseline reaches 84.1%"),
            _item("3406", "Certainty-guided | 88.3 | 3406"),
        ],
        verdict="keep",
        claim=claim,
        reason="all supported",
    )
    assert verification_problems(ok, NORM) == []
    missing = ok.model_copy(update={"items": ok.items[:2]})
    assert verification_problems(missing, NORM) == [
        "number '3406' in the statement has no verified quote containing it"
    ]
    wrong_quote = ok.model_copy(
        update={"items": [*ok.items[:2], _item("3406", "Certainty-guided | 88.3 | 3999")]}
    )
    problems = verification_problems(wrong_quote, NORM)
    assert any("not verbatim" in p for p in problems) and any("'3406'" in p for p in problems)
    # a quote that exists but does not contain the number does not cover it
    elsewhere = ok.model_copy(
        update={"items": [*ok.items[:2], _item("3406", "p=0.5 | 87.9 | 3266")]}
    )
    assert any("'3406'" in p for p in verification_problems(elsewhere, NORM))
    assert (
        verification_problems(
            ClaimVerification(items=[], verdict="drop", claim=None, reason="x"), NORM
        )
        == []
    )


class ScriptedVerifier:
    def __init__(self, answers: list[ClaimVerification]) -> None:
        self.answers = answers
        self.users: list[str] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.users.append(req.user)
        return TransportResponse(
            parsed=self.answers.pop(0),
            usage=TokenUsage(input_tokens=100, output_tokens=10),
            stop_reason="end_turn",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, anthropic_api_key=None, llm_max_retries=1)


def test_verify_retries_then_keeps_fixes_or_drops(settings):
    c = _claim("Reaches 88.3%")
    good = ClaimVerification(
        items=[_item("88.3%", "Our method reaches 88.3% accuracy on AMC23")],
        verdict="keep",
        claim=c,
        reason="ok",
    )
    bad = good.model_copy(update={"items": [_item("88.3%", "made-up quote with 88.3%")]})
    fixed_claim = _claim("Reaches 88.3% on AMC23")
    fixed = good.model_copy(update={"verdict": "fixed", "claim": fixed_claim})
    drop = ClaimVerification(items=[], verdict="drop", claim=None, reason="unsupported")
    fake = ScriptedVerifier([bad, good, fixed, drop, bad, bad])
    kept, records, _ = verify_claims(
        LLMClient(settings, transport=fake),
        [c, _claim("Reaches 90%"), _claim("x"), _claim("y 88.3")],
        paper_context=PAPER,
        prompt=load_prompt("critic_v5"),
        model="claude-sonnet-5",
        max_retries=1,
    )
    assert [r.outcome for r in records] == ["kept", "fixed", "dropped", "dropped"]
    assert kept == [c, fixed_claim]
    assert records[0].attempts == 2 and "<verification_errors>" in fake.users[1]
    assert records[3].problems and "could not be verified" in records[3].reason


def _paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p1",
        source_path="p1.pdf",
        pdf_sha256="0" * 64,
        title="Paper One",
        abstract="Abstract.",
        sections=[
            Section(
                heading="Results",
                number="4",
                paragraphs=["It reaches 88.3% accuracy on AMC23 overall."],
            )
        ],
        tables=[
            Table(label="Table 2", caption="Ablation", rows=[["Variant", "Acc"], ["Ours", "88.3"]])
        ],
    )


class PipelineFake:
    """Pass one, critic, then scripted verification answers."""

    def __init__(self, verifications: list[ClaimVerification]) -> None:
        draft = ClaimsDraft(
            one_line_summary="s",
            claims=[_claim("Reaches 88.3%"), _claim("Reaches 99%")],
            keywords=[f"k{i}" for i in range(5)],
        )
        notes = [
            CriticNote(
                draft_index=i, final_index=i, verdict=ClaimVerdict.kept, defects=[], reason="ok"
            )
            for i in range(2)
        ]
        self.queues = {
            "ClaimsDraft": [draft],
            "CriticOutput": [CriticOutput(review=notes, final=draft)],
            "ClaimVerification": list(verifications),
        }

    def __call__(self, req: TransportRequest) -> TransportResponse:
        return TransportResponse(
            parsed=self.queues[req.schema.__name__].pop(0),
            usage=TokenUsage(input_tokens=10, output_tokens=1),
            stop_reason="end_turn",
        )


def test_extraction_with_verification_records_and_can_empty(settings):
    s5 = settings.model_copy(update={"verify_prompt": "critic_v5", "number_grounding": False})
    keep = ClaimVerification(
        items=[_item("88.3%", "It reaches 88.3% accuracy on AMC23 overall.")],
        verdict="keep",
        claim=_claim("Reaches 88.3%"),
        reason="ok",
    )
    drop = ClaimVerification(items=[], verdict="drop", claim=None, reason="99% not in paper")
    claims, _ = extract_claims(
        _paper(), "claude-sonnet-5", LLMClient(s5, PipelineFake([keep, drop])), s5
    )
    meta = claims.extraction_metadata
    assert [c.statement for c in claims.claims] == ["Reaches 88.3%"]
    assert [r.outcome for r in meta.verification] == ["kept", "dropped"]
    assert meta.prompt_version.endswith("+critic_v5") and "verify" in meta.pass_usage
    fresh = s5.model_copy(update={"data_dir": s5.data_dir / "fresh_cache"})
    claims, _ = extract_claims(
        _paper(), "claude-sonnet-5", LLMClient(fresh, PipelineFake([drop, drop])), fresh
    )
    assert claims.claims == [] and all(
        r.outcome == "dropped" for r in claims.extraction_metadata.verification
    )


def test_api_failure_fails_the_paper_instead_of_dropping(settings):
    from reviewer.claims.extractor import ExtractionError
    from reviewer.llm.client import LLMAPIError

    s5 = settings.model_copy(update={"verify_prompt": "critic_v5", "number_grounding": False})

    class Offline(PipelineFake):
        def __call__(self, req: TransportRequest) -> TransportResponse:
            if req.schema is ClaimVerification:
                raise LLMAPIError("connection error")
            return super().__call__(req)

    with pytest.raises(ExtractionError, match="verification: claim 0: LLMAPIError"):
        extract_claims(_paper(), "claude-sonnet-5", LLMClient(s5, Offline([])), s5)
