import json
import re

import pytest
import typer

from reviewer.config import Settings
from reviewer.eval.judge import (
    ClaimRef,
    JudgeError,
    agreement,
    best_excerpt,
    calibration_sample,
    fresh_sample,
    judge_context,
    judge_context_full,
    judge_paper,
    judge_user_message,
    judgment_problems,
    load_judge_prompt,
    parse_calibration,
    render_calibration,
)
from reviewer.eval.judge_commands import (
    judge_out_dir,
    run_agreement,
    run_calibration_file,
    run_judge,
    run_summary,
)
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    Claim,
    ClaimJudgment,
    ExtractionMetadata,
    GoldClaim,
    GoldClaims,
    JudgeOutput,
    PaperClaims,
    ParsedPaper,
    Section,
    Table,
    TokenUsage,
)


def _claim(statement: str, result: str = "r", evidence: list[str] | None = None) -> Claim:
    return Claim(
        statement=statement,
        contribution_type="new_method",
        method="m",
        problem="p",
        setting="s",
        result=result,
        novelty_assertion="n",
        evidence=evidence or [],
        confidence=0.9,
    )


def _paper(pid: str = "2501.00001") -> ParsedPaper:
    return ParsedPaper(
        paper_id=pid,
        source_path="x.pdf",
        pdf_sha256="0" * 64,
        title="Early Exit for Reasoning",
        abstract="We stop reasoning early when confident.",
        sections=[
            Section(heading="Introduction", paragraphs=["Reasoning models overthink."]),
            Section(heading="Related Work", paragraphs=["RELATED STUFF"]),
            Section(heading="Experimental Results", paragraphs=["Accuracy rises to 69.2%."]),
            Section(heading="Conclusion", paragraphs=["It works."]),
        ],
        tables=[Table(label="Table 2", caption="Main results", raw_text="DEER 69.2 vanilla 64.2")],
    )


def _claims(pid: str, n: int = 2) -> PaperClaims:
    meta = ExtractionMetadata(
        model="m",
        prompt_version="extract_v3+critic_v3",
        timestamp="2026-09-25T00:00:00Z",
        token_usage=TokenUsage(),
    )
    return PaperClaims(
        paper_id=pid,
        title="t",
        one_line_summary="s",
        claims=[
            _claim(f"claim {i} reaches 69.2% accuracy", "69.2%", ["Table 2"]) for i in range(n)
        ],
        keywords=["a", "b", "c", "d", "e"],
        extraction_metadata=meta,
    )


def test_context_has_results_and_tables_not_related_work():
    ctx = judge_context(_paper())
    assert "Experimental Results" in ctx and "69.2%" in ctx and "Table 2: Main results" in ctx
    assert "Introduction" in ctx and "Conclusion" in ctx and "RELATED STUFF" not in ctx


def test_v2_context_is_full_text_without_related_work_and_uncut_tables():
    paper = _paper()
    paper.sections.insert(2, Section(heading="Method", paragraphs=["METHOD DETAIL"]))
    paper.sections.append(Section(heading="References", paragraphs=["[1] REF"]))
    paper.tables[0].raw_text = "x " * 5000
    ctx = judge_context_full(paper)
    assert "METHOD DETAIL" in ctx and "RELATED STUFF" not in ctx and "[1] REF" not in ctx
    assert "truncated" not in ctx and "x " * 4999 in ctx
    assert "METHOD DETAIL" not in judge_context(paper)  # v1 unchanged


def test_gold_is_reference_only_when_present():
    gold = GoldClaims(
        paper_id="p", source="author", claims=[GoldClaim(statement="G1", result="3x")]
    )
    msg = judge_user_message(_paper(), _claims("p"), gold)
    assert "<reference_claims>\n- G1 (result: 3x)" in msg
    assert "<reference_claims>" not in judge_user_message(_paper(), _claims("p"), None)


def _j(i: int, s: int, c: int, sr: str = "", cr: str = "") -> ClaimJudgment:
    return ClaimJudgment(
        claim_index=i, specificity=s, specificity_reason=sr, correctness=c, correctness_reason=cr
    )


def test_judgment_problems():
    assert judgment_problems(JudgeOutput(judgments=[_j(0, 5, 4), _j(1, 3, 5, sr="vague")]), 2) == []
    problems = judgment_problems(JudgeOutput(judgments=[_j(0, 2, 6)]), 2)
    assert any("expected one judgment" in p for p in problems)
    assert any("needs a one-line reason" in p for p in problems)
    assert any("is not 1-5" in p for p in problems)


class FakeJudge:
    """Scores claim i as specificity 5-i, correctness 4; the first `bad` calls omit a reason."""

    def __init__(self, bad: int = 0) -> None:
        self.bad = bad
        self.calls = 0

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls += 1
        n = len(re.findall(r'<claim index="\d+">', req.user))
        js = [
            _j(i, 5 - i, 4, sr="" if self.calls <= self.bad else ("generic" if 5 - i <= 3 else ""))
            for i in range(n)
        ]
        return TransportResponse(
            parsed=JudgeOutput(judgments=js),
            usage=TokenUsage(input_tokens=1000, output_tokens=100),
            stop_reason="end_turn",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        anthropic_api_key=None,
        llm_max_retries=2,
    )


def test_judge_paper_retries_until_reasons_given(settings):
    fake = FakeJudge(bad=1)
    out = judge_paper(
        LLMClient(settings, transport=fake),
        _paper(),
        _claims("p", n=3),
        None,
        tier="lab",
        prompt=load_judge_prompt("judge_v1"),
        model="claude-opus-5-5",
        max_retries=2,
    )
    assert fake.calls == 2 and [j.specificity for j in out.judgments] == [5, 4, 3]
    assert out.judgments[2].specificity_reason == "generic" and out.used_gold is False
    with pytest.raises(JudgeError):
        judge_paper(
            LLMClient(settings, transport=FakeJudge(bad=99)),
            _paper(),
            _claims("q", n=3),
            None,
            tier="lab",
            prompt=load_judge_prompt("judge_v1"),
            model="m",
            max_retries=1,
        )


def test_calibration_sample_takes_all_own_and_is_seeded():
    refs = [ClaimRef("own-x-12345678", i, "own") for i in range(4)]
    refs += [ClaimRef(f"25{i:02d}.00001", k, "lab") for i in range(10) for k in range(3)]
    refs += [ClaimRef(f"26{i:02d}.00001", k, "ordinary") for i in range(8) for k in range(3)]
    refs += [ClaimRef("1706.03762", 0, "famous")]
    a = calibration_sample(refs)
    assert len(a) == 10 and sum(r.tier == "own" for r in a) == 4
    assert all(r.tier != "famous" for r in a)
    assert a == calibration_sample(list(reversed(refs)))


def test_best_excerpt_prefers_cited_table_with_numbers():
    where, text = best_excerpt(_paper(), _claim("DEER reaches 69.2% accuracy", "69.2", ["Table 2"]))
    assert where == "Table 2" and "DEER 69.2" in text


def test_calibration_round_trip_and_agreement():
    items = [
        (ClaimRef("2501.00001", 0, "lab"), _claim("c0"), ("Table 2", "row")),
        (ClaimRef("own-x-12345678", 1, "own"), _claim("c1"), ("Intro", "para")),
    ]
    text = render_calibration(items)
    assert "## C2 · own · own-x-12345678#2" in text and "specificity (1-5): \n" in text
    blank = parse_calibration(text)
    assert blank["2501.00001#1"] == {"specificity": None, "correctness": None}
    filled = text.replace("- specificity (1-5): ", "- specificity (1-5): 4", 1).replace(
        "- correctness (1-5): ", "- correctness (1-5): 2", 1
    )
    human = parse_calibration(filled)
    assert human["2501.00001#1"] == {"specificity": 4, "correctness": 2}
    judge = {
        "2501.00001#1": {"specificity": 5, "correctness": 5},
        "own-x-12345678#2": {"specificity": 1, "correctness": 1},
    }
    ag = agreement(human, judge)
    assert ag["specificity"]["n"] == 1 and ag["specificity"]["within_1"] == 1.0
    assert ag["correctness"]["differ_2plus"] == [("2501.00001#1", 2, 5)]


def _setup_corpus(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True)
    settings.corpus_file.write_text(
        "papers:\n"
        "  - {paper_id: '2501.00001', tier: lab}\n"
        "  - {paper_id: '2601.00002', tier: ordinary}\n"
    )
    for pid in ("2501.00001", "2601.00002"):
        for d, obj in ((settings.claims_dir, _claims(pid, 3)), (settings.parsed_dir, _paper(pid))):
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{pid}.json").write_text(obj.model_dump_json())


def test_calibration_file_agreement_and_summary_end_to_end(settings):
    _setup_corpus(settings)
    run_calibration_file(settings=settings)
    text = settings.calibration_file.read_text()
    assert text.count("## C") == 6 and "judge" in text and "specificity (1-5): \n" in text
    with pytest.raises(typer.Exit):  # nothing graded yet
        run_agreement(settings=settings, client=LLMClient(settings, transport=FakeJudge()))
    first = parse_calibration(text)
    cid = next(iter(first))
    graded = text.replace("- specificity (1-5): ", "- specificity (1-5): 1", 1)
    settings.calibration_file.write_text(graded)
    with pytest.raises(typer.Exit):  # grades entered: never overwritten
        run_calibration_file(settings=settings)
    result = run_agreement(settings=settings, client=LLMClient(settings, transport=FakeJudge()))
    assert result["specificity"]["n"] == 1
    report = (
        settings.private_reports_dir / "judge_agreement_calibration_10_judge_v2.md"
    ).read_text()
    assert "| specificity | 1 |" in report
    assert cid.split("#")[0] in {p.stem for p in settings.judge_dir.glob("*.json")}

    run_judge(settings=settings, client=LLMClient(settings, transport=FakeJudge()))
    run_summary(settings=settings)
    summary = (settings.private_reports_dir / "judge_summary_judge_v2.md").read_text()
    assert "| lab | 1 | 3 |" in summary and "| ordinary | 1 | 3 |" in summary
    assert "## 10 lowest-scoring claims" in summary and "Specificity: generic" in summary
    data = json.loads((settings.judge_dir / "2501.00001.json").read_text())
    assert data["tier"] == "lab" and data["prompt_version"] == "judge_v2"


def test_fresh_sample_excludes_graded_and_is_seeded(settings):
    refs = [ClaimRef(f"25{i:02d}.00001", k, "lab") for i in range(10) for k in range(3)]
    done = {r.id for r in refs[:10]}
    a = fresh_sample(refs, done, 5)
    assert len(a) == 5 and not {r.id for r in a} & done
    assert a == fresh_sample(list(reversed(refs)), done, 5)
    _setup_corpus(settings)
    with pytest.raises(typer.Exit):  # needs the 10-claim file to exclude
        run_calibration_file(settings=settings, fresh=2)
    run_calibration_file(settings=settings)
    run_calibration_file(settings=settings, fresh=2)  # only 6 claims exist; 6 used -> 0 left
    text = (settings.private_reports_dir / "calibration_fresh_2.md").read_text()
    assert "clean check (0 fresh claims)" in text


def test_judge_other_claim_set_writes_separately(settings):
    _setup_corpus(settings)
    other = settings.data_dir / "claims_v4"
    other.mkdir()
    (other / "2501.00001.json").write_text(_claims("2501.00001", 1).model_dump_json())
    run_judge(
        settings=settings, client=LLMClient(settings, transport=FakeJudge()), claims_dir=other
    )
    assert judge_out_dir(settings, other).name == "judge_v2__claims_v4"
    assert [p.stem for p in judge_out_dir(settings, other).glob("*.json")] == ["2501.00001"]
    assert not settings.judge_dir.exists()
    run_summary(settings=settings, claims_dir=other)
    assert (settings.private_reports_dir / "judge_summary_judge_v2__claims_v4.md").exists()


def test_compare_and_spotcheck(settings):
    from reviewer.eval.judge_commands import run_compare, run_spotcheck
    from reviewer.models import SupportItem, VerificationRecord

    _setup_corpus(settings)
    v4, v5 = settings.data_dir / "claims_v4", settings.data_dir / "claims_v5"
    for d in (v4, v5):
        d.mkdir()
    for pid in ("2501.00001", "2601.00002"):
        before = _claims(pid, 3)
        (v4 / f"{pid}.json").write_text(before.model_dump_json())
        recs = [
            VerificationRecord(
                input_index=0,
                outcome="fixed",
                reason="number corrected",
                items=[
                    SupportItem(field="result", element="69.2%", quote="q row", source="Table 2")
                ],
            ),
            VerificationRecord(
                input_index=1, outcome="dropped", reason="unsupported", problems=["p"]
            ),
            VerificationRecord(input_index=2, outcome="kept", reason="ok"),
        ]
        meta = before.extraction_metadata.model_copy(
            update={"prompt_version": "extract_v3+critic_v3+critic_v5", "verification": recs}
        )
        after = before.model_copy(
            update={
                "claims": [
                    before.claims[0].model_copy(update={"statement": "FIXED"}),
                    before.claims[2],
                ],
                "extraction_metadata": meta,
            }
        )
        (v5 / f"{pid}.json").write_text(after.model_dump_json())
    for d in (v4, v5):
        run_judge(
            settings=settings, client=LLMClient(settings, transport=FakeJudge()), claims_dir=d
        )
    stats = run_compare(v4, v5, settings=settings)
    assert stats["a"]["all"]["claims"] == 6 and stats["b"]["all"]["claims"] == 4
    out = run_spotcheck(v4, v5, n=3, settings=settings)
    text = out.read_text()
    assert text.count("## S") == 3 and "change is right (y/n): " in text
    assert "specificity" not in text.lower() or "judge scores are not" in text.lower()
    if "**fixed**" in text:
        assert "**After:** FIXED" in text and '"q row"' in text
    with pytest.raises(typer.BadParameter):  # never overwrite a spot-check file
        run_spotcheck(v4, v5, n=3, settings=settings)
