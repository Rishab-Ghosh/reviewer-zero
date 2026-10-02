import pytest

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.meta.blinding import blind_text
from reviewer.meta.calibration import calibrate, extract_features
from reviewer.meta.evidence import (
    citations_evidence,
    claims_evidence,
    format_evidence,
    methods_evidence,
    novelty_evidence,
    writing_evidence,
)
from reviewer.meta.input_builder import build_meta_input, checker_status
from reviewer.meta.judge import (
    MetaJudgeError,
    judge_paper,
    judgment_problems,
    rating_scale,
    reconcile_judgment,
    render_input,
)
from reviewer.meta.prompts import load_meta_prompt
from reviewer.meta.report import render_review
from reviewer.models import (
    CandidateCounts,
    CheckerName,
    CheckerRunStatus,
    Claim,
    ClaimNovelty,
    ExtractionMetadata,
    JudgedCandidate,
    MetaJudgment,
    MetaReview,
    NoveltyLabel,
    NoveltyQuery,
    PaperClaims,
    PaperNovelty,
    ParsedPaper,
    ReviewCategory,
    Section,
    Strength,
    SubScoreItem,
    TokenUsage,
    Weakness,
    WeaknessSeverity,
)


def _paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p1",
        source_path="p1.pdf",
        pdf_sha256="0" * 64,
        title="A Great Paper",
        abstract="We do things.",
        sections=[
            Section(heading="Introduction", number="1", paragraphs=["We introduce a method."]),
            Section(
                heading="Acknowledgments",
                number="5",
                paragraphs=[
                    "We thank the reviewers for their feedback. Funded by grant no. 12345."
                ],
            ),
            Section(heading="Related Work", number="2", paragraphs=["Prior work did X."]),
        ],
    )


# --------------------------------------------------------------------------- blinding


def test_blind_text_strips_venue_headers_urls_funding_and_acknowledgements():
    paper = _paper()
    paper.sections.insert(
        1,
        Section(
            heading="Method",
            number="2",
            paragraphs=[
                "Published as a conference paper at ICLR 2024",
                "Code is at https://github.com/someone/repo for reproducibility.",
                "Our method works well.",
            ],
        ),
    )
    text = blind_text(paper)
    assert "Published as a conference paper" not in text
    assert "github.com" not in text and "[url removed]" in text
    assert "thank the reviewers" not in text and "grant no" not in text
    assert "Acknowledgments" not in text
    assert "Related Work" not in text and "Prior work did X" not in text  # existing skip kept
    assert "A Great Paper" in text  # title kept
    assert "We introduce a method" in text  # ordinary content kept


# --------------------------------------------------------------------------- evidence


def _claim(tag: str, confidence: float = 0.8) -> Claim:
    return Claim(
        statement=f"{tag} statement",
        contribution_type="new_method",
        method="m",
        problem="p",
        setting="s",
        result="r",
        novelty_assertion="n",
        evidence=["Table 1"],
        confidence=confidence,
    )


def _paper_claims() -> PaperClaims:
    meta = ExtractionMetadata(
        model="m",
        prompt_version="v",
        timestamp="2026-09-27T00:00:00Z",
        token_usage=TokenUsage(),
    )
    return PaperClaims(
        paper_id="p1",
        title="t",
        one_line_summary="s",
        claims=[_claim("a", 0.9), _claim("b", 0.5)],
        keywords=["a", "b", "c", "d", "e"],
        extraction_metadata=meta,
    )


def test_claims_evidence_ids_and_text():
    items = claims_evidence(_paper_claims())
    assert [i.id for i in items] == ["claim:1", "claim:2"]
    assert items[0].checker == CheckerName.claims
    assert "a statement" in items[0].text


def test_claims_evidence_empty_when_missing():
    assert claims_evidence(None) == []


def _judged_candidate(key: str, label: NoveltyLabel, reason: str = "r") -> JudgedCandidate:
    from reviewer.models import Candidate, CandidateSource, IndexPaper

    p = IndexPaper(key=key, title=f"Paper {key}", abstract="abs", year=2020, similarity=0.8)
    cand = Candidate(paper=p, similarity=0.8, sources=[CandidateSource.search_statement])
    return JudgedCandidate(candidate=cand, label=label, reason=reason)


def test_novelty_evidence_top10_and_duplicate_flag():
    from reviewer.models import SeverityResult

    ranked = [_judged_candidate(f"200{i}.0000{i}", NoveltyLabel.close) for i in range(3)]
    claim_novelty = ClaimNovelty(
        claim_index=0,
        query=NoveltyQuery(statement="STMT"),
        verdict="incremental",
        top=ranked[:5],
        ranked=ranked,
        counts=CandidateCounts(pool=60),
        token_usage=TokenUsage(),
        severity=SeverityResult(
            score=3,
            drivers=[ranked[0].candidate.paper.key],
            reason="already done",
            prompt_version="p",
            model="m",
        ),
    )
    paper_novelty = PaperNovelty(
        paper_id="p1",
        title="t",
        tier="lab",
        cutoff=None,
        index_version="v",
        model="m",
        prompt_version="p",
        timestamp="2026-09-27T00:00:00Z",
        claims=[claim_novelty],
        token_usage=TokenUsage(),
    )
    items = novelty_evidence(paper_novelty)
    ids = [i.id for i in items]
    assert ids == ["novelty:1:top1", "novelty:1:top2", "novelty:1:top3", "novelty:1:duplicate"]
    assert "already done" in items[-1].text
    assert all("verdict" not in i.text.lower() for i in items)  # never a numeric/verdict word
    assert all("severity" not in i.text.lower() for i in items)  # no raw severity score leaks


def test_novelty_evidence_no_flag_below_severity_3():
    from reviewer.models import SeverityResult

    ranked = [_judged_candidate("2001.00001", NoveltyLabel.close)]
    cn = ClaimNovelty(
        claim_index=0,
        query=NoveltyQuery(statement="STMT"),
        verdict="incremental",
        top=ranked,
        ranked=ranked,
        counts=CandidateCounts(pool=60),
        token_usage=TokenUsage(),
        severity=SeverityResult(
            score=1, drivers=[], reason="minor overlap", prompt_version="p", model="m"
        ),
    )
    pn = PaperNovelty(
        paper_id="p1",
        title="t",
        tier="lab",
        cutoff=None,
        index_version="v",
        model="m",
        prompt_version="p",
        timestamp="2026-09-27T00:00:00Z",
        claims=[cn],
        token_usage=TokenUsage(),
    )
    ids = [i.id for i in novelty_evidence(pn)]
    assert ids == ["novelty:1:top1"] and "duplicate" not in " ".join(ids)


def test_citations_evidence_skips_cant_tell_and_ids_flags_by_ref():
    from reviewer.models import (
        CitationFlag,
        CitationReport,
        CitationSupport,
        FlagType,
        ReferenceCheck,
        SupportVerdict,
    )

    report = CitationReport(
        paper_id="p1",
        title="t",
        private=False,
        generated="2026-09-27T00:00:00Z",
        index_available=True,
        model="m",
        prompt_version="v",
        references=[
            ReferenceCheck(
                ref_id="b1",
                raw_string="Smith 2020",
                flags=[CitationFlag(type=FlagType.unresolved, ref_id="b1", detail="not found")],
            )
        ],
        support=[
            CitationSupport(
                section_heading="Intro",
                sentence="X shows Y [1].",
                ref_id="b1",
                verdict=SupportVerdict.contradicted,
                judged_by_llm=True,
            ),
            CitationSupport(
                section_heading="Intro",
                sentence="Z [2].",
                ref_id="b2",
                verdict=SupportVerdict.cant_tell_from_abstract,
                judged_by_llm=True,
            ),
        ],
    )
    items = citations_evidence(report)
    ids = [i.id for i in items]
    assert "cite:b1" in ids and "cite:s1" in ids
    assert len(items) == 2  # the cant_tell support judgment is skipped


def test_methods_evidence_skips_na_and_unverified_concerns():
    from reviewer.models import (
        CheckedConcern,
        CheckedFinding,
        MethodsVerdict,
        PaperMethodsReview,
        QuoteCheck,
    )

    review = PaperMethodsReview(
        paper_id="p1",
        tier="lab",
        model="m",
        prompt_version="v",
        effort="medium",
        findings=[
            CheckedFinding(
                item="seeds_variance",
                verdict=MethodsVerdict.no,
                model_verdict=MethodsVerdict.no,
                rationale="no seeds reported",
                concern="results may not be robust",
                quotes=[],
            ),
            CheckedFinding(
                item="leakage",
                verdict=MethodsVerdict.na,
                model_verdict=MethodsVerdict.na,
                rationale="not applicable",
                concern="",
                quotes=[],
            ),
        ],
        other_concerns=[
            CheckedConcern(
                concern="verified one", quotes=[QuoteCheck(quote="q", verified=True)], verified=True
            ),
            CheckedConcern(concern="unverified one", quotes=[], verified=False),
        ],
        token_usage=TokenUsage(),
    )
    ids = [i.id for i in methods_evidence(review)]
    assert ids == ["methods:seeds_variance", "methods:other:0"]


def test_format_evidence_disambiguates_repeated_checks():
    from reviewer.models import (
        DeclaredVenue,
        FormatFinding,
        FormatReport,
        FormatSeverity,
        SubmissionStage,
    )

    report = FormatReport(
        paper_id="p1",
        title="t",
        private=False,
        generated="2026-09-27T00:00:00Z",
        n_pages=9,
        declared=DeclaredVenue(venue=None, stage=SubmissionStage.preprint),
        findings=[
            FormatFinding(
                check="unresolved_reference", severity=FormatSeverity.warning, detail="ref A"
            ),
            FormatFinding(
                check="unresolved_reference", severity=FormatSeverity.warning, detail="ref B"
            ),
            FormatFinding(check="page_limit", severity=FormatSeverity.violation, detail="10 pages"),
        ],
    )
    ids = [i.id for i in format_evidence(report)]
    assert ids == [
        "format:unresolved_reference:0",
        "format:unresolved_reference:1",
        "format:page_limit",
    ]


def test_writing_evidence_covers_abstract_contribution_related_work_and_findings():
    from reviewer.models import (
        AbstractClaimTrace,
        AbstractClaimVerdict,
        ContributionCheck,
        ContributionVerdict,
        RelatedWorkVerdict,
        WritingFinding,
        WritingReport,
    )

    report = WritingReport(
        paper_id="p1",
        title="t",
        private=False,
        model="m",
        prompt_version="v",
        generated="2026-09-27T00:00:00Z",
        abstract_trace=[
            AbstractClaimTrace(
                abstract_span="we improve X by 10%",
                result_span="",
                gap="not shown",
                verdict=AbstractClaimVerdict.overstated,
            )
        ],
        contribution=ContributionCheck(
            intro_span="we propose a method", problem="", verdict=ContributionVerdict.clear
        ),
        related_work=RelatedWorkVerdict.mostly_lists,
        findings=[
            WritingFinding(
                check="undefined_term",
                quote="the FooNet",
                evidence=[],
                location="Section 2",
                detail="never defined",
            )
        ],
        dropped=[],
        token_usage=TokenUsage(),
    )
    ids = [i.id for i in writing_evidence(report)]
    assert ids == [
        "writing:abstract_claim:0",
        "writing:contribution",
        "writing:related_work",
        "writing:undefined_term:0",
    ]


def test_checker_status_ok_failed_not_run():
    assert checker_status(CheckerName.claims, object(), None).status == CheckerRunStatus.ok
    assert checker_status(CheckerName.claims, None, "boom").status == CheckerRunStatus.failed
    assert checker_status(CheckerName.claims, None, None).status == CheckerRunStatus.not_run


# --------------------------------------------------------------------------- input builder


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        anthropic_api_key=None,
    )


def test_build_meta_input_reports_not_run_for_missing_checkers(settings):
    mi = build_meta_input(settings, _paper(), venue="ICLR", venue_year=2025)
    assert mi.paper_id == "p1" and mi.venue == "ICLR" and mi.venue_year == 2025
    assert mi.evidence == []
    statuses = {s.checker.value: s.status for s in mi.checker_status}
    assert set(statuses) == {"claims", "novelty", "citations", "methods", "format", "writing"}
    assert all(v == CheckerRunStatus.not_run for v in statuses.values())
    assert "A Great Paper" in mi.blinded_text


def test_build_meta_input_reports_failed_when_told(settings):
    mi = build_meta_input(settings, _paper(), venue="ICLR", failures={"citations": "index down"})
    statuses = {s.checker.value: s for s in mi.checker_status}
    assert statuses["citations"].status == CheckerRunStatus.failed
    assert statuses["citations"].detail == "index down"


def test_build_meta_input_loads_real_claims_output(settings):
    settings.claims_dir.mkdir(parents=True)
    (settings.claims_dir / "p1.json").write_text(_paper_claims().model_dump_json())
    mi = build_meta_input(settings, _paper(), venue="ICLR")
    assert any(e.id == "claim:1" for e in mi.evidence)
    statuses = {s.checker.value: s.status for s in mi.checker_status}
    assert statuses["claims"] == CheckerRunStatus.ok


# --------------------------------------------------------------------------- judge


def _mi(evidence_ids: list[str] = ()):
    from reviewer.models import EvidenceItem, MetaInput

    return MetaInput(
        paper_id="p1",
        title="t",
        venue="ICLR",
        venue_year=2025,
        blinded_text="full text here",
        evidence=[EvidenceItem(id=i, checker="claims", text=f"evidence {i}") for i in evidence_ids],
        checker_status=[],
    )


def _sub(score: int = 3) -> SubScoreItem:
    return SubScoreItem(score=score, reason="fine")


def _judgment(
    rating: int = 6, evidence_ids: list[str] = ("claim:1",), n_weak: int = 1
) -> MetaJudgment:
    return MetaJudgment(
        summary="A paper about things.",
        strengths=[
            Strength(
                text="good", category=ReviewCategory.soundness, evidence_ids=list(evidence_ids)
            )
        ],
        weaknesses=[
            Weakness(
                text=f"bad {i}",
                severity=WeaknessSeverity.major,
                category=ReviewCategory.soundness,
                evidence_ids=list(evidence_ids),
            )
            for i in range(n_weak)
        ],
        questions=["why?"],
        would_change_my_mind="a bigger ablation",
        soundness=_sub(),
        presentation=_sub(),
        contribution=_sub(),
        raw_rating=rating,
        confidence=3,
    )


def test_rating_scale_known_and_unknown_venue():
    assert rating_scale("ICLR") == [1, 3, 5, 6, 8, 10]
    assert 10 in rating_scale("SomeOtherVenue")


def test_judgment_problems_checks_scale_and_weaknesses():
    scale = rating_scale("ICLR")
    ok = _judgment(rating=6)
    assert judgment_problems(ok, scale) == []
    bad_rating = _judgment(rating=7)
    assert any("scale" in p for p in judgment_problems(bad_rating, scale))
    no_weak = ok.model_copy(update={"weaknesses": []})
    assert any("weakness" in p for p in judgment_problems(no_weak, scale))


def test_reconcile_judgment_drops_unresolved_ids():
    j = _judgment(evidence_ids=["claim:1", "claim:99"])
    cleaned, dropped = reconcile_judgment(j, known_ids={"claim:1"})
    assert dropped == 2  # the strength and the weakness both cite the unresolved id
    assert cleaned.strengths == [] and cleaned.weaknesses == []


def test_reconcile_judgment_keeps_fully_resolved_items():
    j = _judgment(evidence_ids=["claim:1"])
    cleaned, dropped = reconcile_judgment(j, known_ids={"claim:1"})
    assert dropped == 0 and len(cleaned.strengths) == 1 and len(cleaned.weaknesses) == 1


def test_render_input_includes_scale_evidence_and_status():
    mi = _mi(["claim:1"])
    text = render_input(mi)
    assert "1, 3, 5, 6, 8, 10" in text
    assert 'evidence id="claim:1"' in text
    assert "full text here" in text


class FakeJudgeLLM:
    def __init__(self, judgment: MetaJudgment) -> None:
        self.judgment = judgment
        self.calls = 0

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls += 1
        return TransportResponse(
            parsed=self.judgment,
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            stop_reason="end_turn",
        )


def test_judge_paper_retries_on_bad_rating_then_reconciles(settings):
    good = _judgment(rating=6, evidence_ids=["claim:1"])

    class TwoAttempts:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, req: TransportRequest) -> TransportResponse:
            self.calls += 1
            j = _judgment(rating=7) if self.calls == 1 else good  # 7 is not on ICLR's scale
            return TransportResponse(
                parsed=j, usage=TokenUsage(input_tokens=10, output_tokens=1), stop_reason="end_turn"
            )

    fake = TwoAttempts()
    client = LLMClient(settings, transport=fake)
    mi = _mi(["claim:1"])
    out, dropped, _usage = judge_paper(
        client, mi, prompt=load_meta_prompt("meta_v1"), model="claude-opus-5-5", max_retries=2
    )
    assert fake.calls == 2 and out.raw_rating == 6 and dropped == 0


def test_judge_paper_raises_after_exhausting_retries(settings):
    always_bad = _judgment(rating=7)
    fake = FakeJudgeLLM(always_bad)
    client = LLMClient(settings, transport=fake)
    with pytest.raises(MetaJudgeError):
        judge_paper(
            client,
            _mi(["claim:1"]),
            prompt=load_meta_prompt("meta_v1"),
            model="claude-opus-5-5",
            max_retries=1,
        )


# --------------------------------------------------------------------------- calibration


def test_extract_features_median_across_runs():
    js = [_judgment(rating=5), _judgment(rating=8)]
    feats = extract_features(js)
    assert feats["raw_rating"] == 6.5
    assert feats["rating_spread"] == 3.0


def test_calibrate_is_unfitted_and_uses_median_rating():
    cal = calibrate([_judgment(rating=6), _judgment(rating=8)])
    assert cal.fitted is False
    assert cal.rating == 7.0
    assert cal.rating_percentile is None and cal.p_accept is None


def test_calibrate_empty_judgments():
    cal = calibrate([])
    assert cal.fitted is False and cal.rating == 0.0


# --------------------------------------------------------------------------- report rendering


def test_render_review_has_no_numeric_novelty_verdict_and_shows_checker_status():
    from reviewer.models import CheckerStatus

    review = MetaReview(
        paper_id="p1",
        title="A Great Paper",
        venue="ICLR",
        venue_year=2025,
        model="claude-opus-5-5",
        prompt_version="meta_v1",
        generated="2026-09-27T00:00:00Z",
        judgments=[_judgment(rating=6, evidence_ids=["novelty:1:top1"])],
        dropped_evidence_ids=2,
        calibration=calibrate([_judgment(rating=6)]),
        checker_status=[
            CheckerStatus(checker=CheckerName.novelty, status=CheckerRunStatus.ok),
            CheckerStatus(checker=CheckerName.methods, status=CheckerRunStatus.not_run),
        ],
        token_usage=TokenUsage(input_tokens=1000, output_tokens=100, cost_usd=0.5),
    )
    text = render_review(review)
    assert "# Meta-review: p1" in text
    assert "Display rating: 6.5" in text  # raw 6 -> 1.984 + 0.750 * 6
    assert "2 strength/weakness item(s) were dropped" in text
    assert "*not run*" in text
    assert "## Strengths" in text and "## Weaknesses" in text and "## Rating" in text
    assert "novelty_verdict" not in text.lower()


def test_render_review_sorts_weaknesses_fatal_first():
    j = _judgment(rating=6, evidence_ids=["claim:1"], n_weak=0).model_copy(
        update={
            "weaknesses": [
                Weakness(
                    text="minor thing",
                    severity=WeaknessSeverity.minor,
                    category=ReviewCategory.presentation,
                    evidence_ids=["claim:1"],
                ),
                Weakness(
                    text="fatal thing",
                    severity=WeaknessSeverity.fatal,
                    category=ReviewCategory.soundness,
                    evidence_ids=["claim:1"],
                ),
                Weakness(
                    text="major thing",
                    severity=WeaknessSeverity.major,
                    category=ReviewCategory.novelty,
                    evidence_ids=["claim:1"],
                ),
            ]
        }
    )
    from reviewer.models import CheckerStatus

    review = MetaReview(
        paper_id="p1",
        title="t",
        venue="ICLR",
        venue_year=2025,
        model="m",
        prompt_version="v",
        generated="2026-09-27T00:00:00Z",
        judgments=[j],
        calibration=calibrate([j]),
        checker_status=[CheckerStatus(checker=CheckerName.claims, status=CheckerRunStatus.ok)],
        token_usage=TokenUsage(),
    )
    text = render_review(review)
    assert text.index("fatal thing") < text.index("major thing") < text.index("minor thing")


def test_render_review_omits_baselines_section_when_none():
    from reviewer.models import CheckerStatus

    review = MetaReview(
        paper_id="p1",
        title="t",
        venue="ICLR",
        venue_year=2025,
        model="m",
        prompt_version="v",
        generated="2026-09-27T00:00:00Z",
        judgments=[_judgment(rating=6, evidence_ids=["claim:1"])],
        calibration=calibrate([_judgment(rating=6)]),
        checker_status=[CheckerStatus(checker=CheckerName.claims, status=CheckerRunStatus.ok)],
        token_usage=TokenUsage(),
    )
    text = render_review(review)
    assert "Baselines and contamination probe" not in text


def test_render_review_shows_baselines_and_probe():
    from reviewer.models import BaselineRating, CheckerStatus, ContaminationProbe

    review = MetaReview(
        paper_id="p1",
        title="t",
        venue="ICLR",
        venue_year=2025,
        model="m",
        prompt_version="v",
        generated="2026-09-27T00:00:00Z",
        judgments=[_judgment(rating=6, evidence_ids=["claim:1"])],
        calibration=calibrate([_judgment(rating=6)]),
        checker_status=[CheckerStatus(checker=CheckerName.claims, status=CheckerRunStatus.ok)],
        token_usage=TokenUsage(),
        baseline_abstract=BaselineRating(reasoning="skimmed", raw_rating=5, confidence=3),
        baseline_full=BaselineRating(reasoning="read closely", raw_rating=3, confidence=4),
        contamination_probe=ContaminationProbe(
            accepted_guess=True, reasoning="I recall this exact paper", confidence=5
        ),
    )
    text = render_review(review)
    assert "Abstract-only baseline (a):** raw rating **5**" in text
    assert "Full-text, no-checkers baseline (b):** raw rating **3**" in text
    assert "Contamination probe:** model guesses accepted" in text


# --------------------------------------------------------------------------- baselines


def test_judge_baseline_rating_retries_then_succeeds(settings):
    from reviewer.models import BaselineRating

    good = BaselineRating(reasoning="ok", raw_rating=6, confidence=4)

    class TwoAttempts:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, req: TransportRequest) -> TransportResponse:
            self.calls += 1
            out = BaselineRating(reasoning="bad", raw_rating=7, confidence=4)  # not on ICLR scale
            return TransportResponse(
                parsed=out if self.calls == 1 else good,
                usage=TokenUsage(input_tokens=10, output_tokens=1),
                stop_reason="end_turn",
            )

    fake = TwoAttempts()
    client = LLMClient(settings, transport=fake)
    from reviewer.meta.baselines import judge_baseline_rating

    out, usage = judge_baseline_rating(
        client,
        "some paper text",
        prompt=load_meta_prompt("baseline_abstract_v1"),
        model="claude-opus-5-5",
        venue="ICLR",
        max_retries=2,
    )
    assert fake.calls == 2 and out.raw_rating == 6
    assert usage.input_tokens == 20


def test_judge_baseline_rating_raises_after_exhausting_retries(settings):
    from reviewer.meta.baselines import BaselineJudgeError, judge_baseline_rating
    from reviewer.models import BaselineRating

    always_bad = BaselineRating(reasoning="bad", raw_rating=7, confidence=4)
    fake = FakeJudgeLLM(always_bad)
    client = LLMClient(settings, transport=fake)
    with pytest.raises(BaselineJudgeError):
        judge_baseline_rating(
            client,
            "text",
            prompt=load_meta_prompt("baseline_full_v1"),
            model="claude-opus-5-5",
            venue="ICLR",
            max_retries=1,
        )


def test_run_baseline_abstract_uses_title_and_abstract_only(settings):
    from reviewer.meta.baselines import run_baseline_abstract
    from reviewer.models import BaselineRating

    seen = {}

    class Capturing:
        def __call__(self, req: TransportRequest) -> TransportResponse:
            seen["user"] = req.user
            return TransportResponse(
                parsed=BaselineRating(reasoning="ok", raw_rating=6, confidence=3),
                usage=TokenUsage(input_tokens=10, output_tokens=1),
                stop_reason="end_turn",
            )

    client = LLMClient(settings, transport=Capturing())
    paper = ParsedPaper(
        paper_id="p1",
        source_path="x.pdf",
        title="A Great Paper",
        abstract="We show something.",
        sections=[Section(heading="Body", paragraphs=["full body text should not appear"])],
    )
    out, _usage = run_baseline_abstract(
        client,
        paper,
        prompt=load_meta_prompt("baseline_abstract_v1"),
        model="claude-opus-5-5",
        venue="ICLR",
        venue_year=2025,
        max_retries=1,
    )
    assert out.raw_rating == 6
    assert "We show something." in seen["user"]
    assert "full body text should not appear" not in seen["user"]


def test_run_contamination_probe_success(settings):
    from reviewer.meta.baselines import run_contamination_probe
    from reviewer.models import ContaminationProbe

    fake = FakeJudgeLLM(
        ContaminationProbe(accepted_guess=False, reasoning="no memory of this", confidence=1)
    )
    client = LLMClient(settings, transport=fake)
    out, _usage = run_contamination_probe(
        client,
        "Some Paper Title",
        prompt=load_meta_prompt("contamination_v1"),
        model="claude-opus-5-5",
        venue="ICLR",
        venue_year=2025,
        max_retries=1,
    )
    assert out.accepted_guess is False and out.confidence == 1


def test_run_contamination_probe_rejects_bad_confidence(settings):
    from reviewer.meta.baselines import BaselineJudgeError, run_contamination_probe
    from reviewer.models import ContaminationProbe

    bad = ContaminationProbe(accepted_guess=True, reasoning="x", confidence=9)
    fake = FakeJudgeLLM(bad)
    client = LLMClient(settings, transport=fake)
    with pytest.raises(BaselineJudgeError):
        run_contamination_probe(
            client,
            "Some Paper Title",
            prompt=load_meta_prompt("contamination_v1"),
            model="claude-opus-5-5",
            venue="ICLR",
            venue_year=2025,
            max_retries=0,
        )


def test_format_evidence_can_drop_decision_revealing_checks():
    from reviewer.meta.evidence import STATUS_REVEALING_FORMAT_CHECKS
    from reviewer.models import FormatCheck, FormatFinding, FormatReport, FormatSeverity

    fmt = FormatReport.model_construct(
        findings=[
            FormatFinding(
                check=FormatCheck.author_block, severity=FormatSeverity.violation, detail="x"
            ),
            FormatFinding(
                check=FormatCheck.page_limit, severity=FormatSeverity.warning, detail="y"
            ),
            FormatFinding(
                check=FormatCheck.unreferenced_table, severity=FormatSeverity.info, detail="z"
            ),
        ]
    )
    assert len(format_evidence(fmt)) == 3
    kept = format_evidence(fmt, STATUS_REVEALING_FORMAT_CHECKS)
    assert [e.id for e in kept] == ["format:unreferenced_table"]


def test_strip_status_removes_headers_merged_into_titles_and_paragraphs():
    from reviewer.meta.blinding import strip_status

    assert (
        strip_status("Under review as a conference paper at ICLR 2024 GRADIENT DESCENT FOR X")
        == "GRADIENT DESCENT FOR X"
    )
    out = strip_status(
        "a contrastive objective Published as a conference paper at ICLR 2024 Dataset"
    )
    assert "published" not in out.lower() and "ICLR" not in out
    assert out == "a contrastive objective Dataset"


def test_blind_text_strips_mid_paragraph_headers_and_title_headers():
    from reviewer.models import Section

    paper = ParsedPaper(
        paper_id="or-x",
        source_path="data/openreview/pdf/x.pdf",
        title="Under review as a conference paper at ICLR 2025 SOME TITLE",
        abstract="We study things.",
        sections=[
            Section(
                heading="Method",
                paragraphs=[
                    "We train it. Published as a conference paper at ICLR 2025 Then we test."
                ],
            )
        ],
    )
    text = blind_text(paper)
    assert "SOME TITLE" in text and "Then we test." in text
    assert "under review" not in text.lower() and "published as" not in text.lower()


def test_meta_v2_anchors_the_rating_to_the_iclr_scale():
    text = load_meta_prompt("meta_v2").text
    assert "Rating scale (ICLR: 1, 3, 5, 6, 8, 10)" in text
    assert "Use the full range when the evidence supports it" in text
    assert load_meta_prompt("meta_v2").version == "meta_v2"


# --------------------------------------------------------------------------- zero claims


def _zero_claims() -> PaperClaims:
    """Every claim dropped by verification: the one way a claims file may hold zero claims."""
    from reviewer.models import VerificationRecord

    data = _paper_claims().model_dump()
    data["claims"] = []
    data["extraction_metadata"]["verification"] = [
        VerificationRecord(
            input_index=0, outcome="dropped", reason="no verified quote"
        ).model_dump()
    ]
    return PaperClaims.model_validate(data)


def test_zero_claims_is_a_valid_result_and_the_input_says_so(settings):
    settings.claims_dir.mkdir(parents=True)
    (settings.claims_dir / "p1.json").write_text(_zero_claims().model_dump_json())
    mi = build_meta_input(settings, _paper(), venue="ICLR")
    by = {s.checker.value: s for s in mi.checker_status}
    assert by["claims"].status == CheckerRunStatus.ok and by["claims"].detail == "found 0 claims"
    assert by["novelty"].status == CheckerRunStatus.not_run
    assert "found none" in by["novelty"].detail
    assert [e.id for e in mi.evidence] == ["claims:none"]


def test_novelty_not_run_says_why_when_claims_failed(settings):
    mi = build_meta_input(settings, _paper(), venue="ICLR", failures={"claims": "critic invalid"})
    by = {s.checker.value: s for s in mi.checker_status}
    assert by["claims"].status == CheckerRunStatus.failed
    assert "did not produce any" in by["novelty"].detail


def test_ensure_checkers_skips_novelty_on_zero_claims_and_runs_the_rest(settings, monkeypatch):
    import reviewer.meta.commands as mc

    settings.claims_dir.mkdir(parents=True)
    settings.parsed_dir.mkdir(parents=True)
    (settings.claims_dir / "p1.json").write_text(_zero_claims().model_dump_json())
    (settings.parsed_dir / "p1.json").write_text(_paper().model_dump_json())
    ran: list[str] = []

    def novelty(**_kw):
        raise AssertionError("novelty must not run when there are no claims")

    monkeypatch.setattr(mc, "novelty_run_check", novelty)
    monkeypatch.setattr(mc, "run_citations", lambda *a, **k: ran.append("citations"))
    monkeypatch.setattr(mc, "run_format", lambda *a, **k: ran.append("format"))
    monkeypatch.setattr(mc, "run_writing", lambda *a, **k: ran.append("writing"))
    monkeypatch.setattr(mc, "methods_run_check", lambda **k: ran.append("methods"))
    failures = mc.ensure_checkers(_paper(), settings, mc.SpendGuard(10.0), lite=True)
    assert failures == {}
    assert ran == ["citations", "format", "writing", "methods"]


def test_render_review_states_zero_claims_and_why_novelty_did_not_run():
    from reviewer.models import CheckerStatus

    review = MetaReview(
        paper_id="p1",
        title="t",
        venue="ICLR",
        model="m",
        prompt_version="v",
        generated="2026-09-28T00:00:00Z",
        judgments=[_judgment(rating=3, evidence_ids=["claims:none"])],
        calibration=calibrate([_judgment(rating=3)]),
        checker_status=[
            CheckerStatus(
                checker=CheckerName.claims, status=CheckerRunStatus.ok, detail="found 0 claims"
            ),
            CheckerStatus(
                checker=CheckerName.novelty,
                status=CheckerRunStatus.not_run,
                detail="no claims to check: the claims checker found none",
            ),
        ],
        token_usage=TokenUsage(),
    )
    text = render_review(review)
    assert "- claims: ran (found 0 claims)" in text
    assert "- novelty: *not run* (no claims to check: the claims checker found none)" in text


def test_display_rating_is_monotone_and_clamped():
    from reviewer.meta.calibration import display_rating

    vals = [display_rating(r) for r in (1, 3, 5, 6, 8, 10)]
    assert vals == sorted(vals)
    assert round(display_rating(6), 2) == 6.48
    assert display_rating(-50) == 1.0 and display_rating(50) == 10.0


def test_critic_fallback_claims_are_flagged_in_status_and_evidence(settings):
    data = _paper_claims().model_dump()
    data["extraction_metadata"]["critic_fallback"] = "critic pass still invalid after 2 retries"
    settings.claims_dir.mkdir(parents=True)
    (settings.claims_dir / "p1.json").write_text(PaperClaims.model_validate(data).model_dump_json())
    mi = build_meta_input(settings, _paper(), venue="ICLR")
    claims_status = next(s for s in mi.checker_status if s.checker == CheckerName.claims)
    assert claims_status.status == CheckerRunStatus.ok
    assert "not reviewed" in claims_status.detail
    assert all("critic pass failed" in e.text for e in mi.evidence if e.id.startswith("claim:"))
