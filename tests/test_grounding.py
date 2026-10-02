import pytest

from reviewer.claims.extractor import extract_claims
from reviewer.claims.grounding import (
    ground_claims,
    numbers_in,
    paper_number_forms,
    ungrounded_numbers,
)
from reviewer.config import Settings
from reviewer.llm.client import LLMClient
from reviewer.models import (
    Claim,
    ClaimsDraft,
    ClaimVerdict,
    CriticNote,
    CriticOutput,
    Defect,
    ParsedPaper,
    Section,
    Table,
)
from tests.test_extractor import ScriptedTransport


def _claim(statement: str, setting: str = "s", result: str = "r") -> Claim:
    return Claim(
        statement=statement,
        contribution_type="new_method",
        method="m 99.9",  # method is not checked
        problem="p",
        setting=setting,
        result=result,
        novelty_assertion="n",
        evidence=["Table 1"],
        confidence=0.8,
    )


def _missing(claim_text: str, paper_text: str) -> list[str]:
    return ungrounded_numbers(_claim(claim_text), paper_number_forms(paper_text), paper_text)


def test_numbers_skip_names_and_parse_thousands():
    assert [w for w, _, _ in numbers_in("Qwen3-8B and R1 got 1,234.5 on 12 tasks")] == [
        "8B",
        "1,234.5",
        "12",
    ]


@pytest.mark.parametrize(
    ("claim", "paper", "missing"),
    [
        ("accuracy 88.3%", "reaches 88.3 on AMC", []),
        ("accuracy 0.5", "accuracy of 50%", []),  # percent quoted as fraction
        ("accuracy 50%", "accuracy of 0.5", []),  # fraction quoted as percent
        ("1416 hours", "took 14.16 s", ["1416"]),  # no x100 for numbers above 1
        ("88.30 points", "88.3", []),  # trailing zero
        ("88.3 points", "88.27", []),  # rounded to the claim's precision
        ("75% narrower", "about 74.8 narrower", ["75%"]),  # whole numbers must match exactly
        ("16,000 tokens", "lengths of 16K tokens", []),
        ("1,416 GPU hours", "an estimated 1, 416 GPU-hours", []),  # PDF-spaced separator
        ("from (2,128) to (207,768)", "(n_layer, d_model) = (2, 128) ... (207, 768)", []),
        ("over 110,000 CoTs", "approximately 110,799 CoTs", ["110,000"]),
        ("a drop to -0.016", "DiD -0.016", []),
    ],
)
def test_grounding_rules(claim, paper, missing):
    assert _missing(claim, paper) == missing


def test_section_numbers_and_all_fields():
    paper = ParsedPaper(
        paper_id="p",
        source_path="p.pdf",
        pdf_sha256="0" * 64,
        title="T",
        abstract="A 12.5 result.",
        sections=[Section(heading="Results", number="5.2", paragraphs=["Won."])],
        tables=[Table(label="Table 1", caption="c", rows=[["m", "7.7"]])],
    )
    claims = [
        _claim("see Section 5.2", setting="12.5 and 7.7"),
        _claim("ok", result="3.14 gain"),
    ]
    (g,) = ground_claims(claims, paper)
    assert (g.claim_index, g.missing) == (1, ["3.14"])  # method field (99.9) is not checked


# --------------------------------------------------------------------------- extractor v4


def _paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p1",
        source_path="p1.pdf",
        pdf_sha256="0" * 64,
        title="Paper One",
        abstract="Abstract.",
        sections=[
            Section(heading="Introduction", number="1", paragraphs=["Intro."]),
            Section(heading="Related Work", number="2", paragraphs=["RELATED 31.4 stuff"]),
            Section(
                heading="Method", number="3", paragraphs=["Overview.", "METHOD DETAIL gives 42.5"]
            ),
        ],
    )


def _draft(statement: str) -> ClaimsDraft:
    return ClaimsDraft(
        one_line_summary="summary",
        claims=[_claim(statement)],
        keywords=[f"kw{i}" for i in range(5)],
    )


def _critic(statement: str) -> CriticOutput:
    note = CriticNote(
        draft_index=0,
        final_index=0,
        verdict=ClaimVerdict.rewritten,
        defects=[Defect.missing_numbers],
        reason="x",
    )
    return CriticOutput(review=[note], final=_draft(statement))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, anthropic_api_key=None, llm_max_retries=0)


def test_v4_critic_sees_full_paper_and_ungrounded_numbers_go_back(settings):
    fake = ScriptedTransport(
        [_draft("draft claim")],
        [_critic("method reaches 77.7 accuracy"), _critic("method reaches 42.5 accuracy")],
    )
    claims, _ = extract_claims(_paper(), "claude-sonnet-5", LLMClient(settings, fake), settings)
    extract_req, critic1, critic2 = fake.calls
    assert "METHOD DETAIL" not in extract_req.user  # pass one still sees excerpts only
    assert "METHOD DETAIL gives 42.5" in critic1.user and "RELATED 31.4" not in critic1.user
    assert "<number_grounding_errors>" in critic2.user and "77.7" in critic2.user
    assert claims.claims[0].statement == "method reaches 42.5 accuracy"
    g = claims.extraction_metadata.grounding
    assert g is not None and g.rounds == 1 and g.final == []
    assert [(x.claim_index, x.missing) for x in g.initial] == [(0, ["77.7"])]
    assert claims.extraction_metadata.prompt_version == "extract_v3+critic_v4"


def test_still_ungrounded_after_max_rounds_is_recorded_not_fatal(settings):
    fake = ScriptedTransport(
        [_draft("draft")],
        [_critic("reaches 77.7"), _critic("reaches 77.8"), _critic("reaches 77.9")],
    )
    claims, _ = extract_claims(_paper(), "claude-sonnet-5", LLMClient(settings, fake), settings)
    g = claims.extraction_metadata.grounding
    assert g.rounds == 2 and [x.missing for x in g.final] == [["77.9"]]
    assert claims.claims[0].statement == "reaches 77.9"


def test_grounding_can_be_turned_off(settings):
    off = settings.model_copy(update={"number_grounding": False})
    fake = ScriptedTransport([_draft("d")], [_critic("reaches 77.7")])
    claims, _ = extract_claims(_paper(), "claude-sonnet-5", LLMClient(off, fake), off)
    assert claims.extraction_metadata.grounding is None and len(fake.calls) == 2
