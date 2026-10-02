"""Writing checker: quote verification, term ordering, captions, related work, the LLM call
with a fake transport, reports. No network."""

import pytest

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    AbstractClaimTrace,
    AbstractClaimVerdict,
    CaptionCheck,
    ContributionCheck,
    ContributionVerdict,
    Figure,
    ParsedPaper,
    RelatedWorkCheck,
    RelatedWorkVerdict,
    Section,
    Table,
    TermUse,
    TermVerdict,
    TokenUsage,
    WritingCheck,
    WritingOutput,
    WritingReport,
)
from reviewer.writing.checker import check_paper, load_writing_prompt, writing_user_message
from reviewer.writing.commands import is_private, report_dir, run_summary, run_writing, write_report
from reviewer.writing.report import render_report, stratified_sample, summary_markdown
from reviewer.writing.verify import find_span, norm, paper_text, verify_output

ABSTRACT = (
    "We propose SparseMix, a routing method for mixture models. SparseMix achieves state of "
    "the art accuracy on every benchmark we test. It also halves training cost."
)


def _paper(pid: str = "p1", source: str = "data/papers/arxiv/p1.pdf") -> ParsedPaper:
    return ParsedPaper(
        paper_id=pid,
        source_path=source,
        title="SparseMix",
        abstract=ABSTRACT,
        sections=[
            Section(
                heading="Introduction",
                number="1",
                paragraphs=[
                    "Mixture models are costly. We train the router with the TRG objective on "
                    "all tokens. Our contribution is a router that cuts training cost by 40% "
                    "at equal accuracy on GLUE.",
                ],
            ),
            Section(
                heading="Related Work",
                number="2",
                paragraphs=[
                    "Smith et al. proposed dense routing for mixtures of experts. Jones et al. "
                    "studied load balancing in large sparse models. Unlike these, we learn the "
                    "router jointly with the experts.",
                ],
            ),
            Section(
                heading="Method",
                number="3",
                paragraphs=[
                    "The Token Routing Gain (TRG) objective scores each expert per token. "
                    "Training cost falls by 40% relative to the dense baseline."
                ],
            ),
            Section(heading="References", paragraphs=["[1] Smith. Dense routing. 2020."]),
        ],
        tables=[
            Table(
                label="Table 1",
                caption="Results on GLUE. Bold is best.",
                rows=[["Method", "GLUE"], ["SparseMix", "85.1"], ["Dense", "86.0"]],
            )
        ],
        figures=[Figure(label="Figure 1", caption="Loss curves for SM-B and SM-L runs.")],
    )


def _output(**overrides) -> WritingOutput:
    base = dict(
        abstract_claims=[
            AbstractClaimTrace(
                abstract_span="SparseMix achieves state of the art accuracy on every benchmark",
                result_span="SparseMix | 85.1",
                gap="Table 1 shows SparseMix below the dense baseline on GLUE.",
                verdict=AbstractClaimVerdict.overstated,
            ),
            AbstractClaimTrace(
                abstract_span="It also halves training cost.",
                result_span="Training cost falls by 40% relative to the dense baseline.",
                gap="40%, not half.",
                verdict=AbstractClaimVerdict.overstated,
            ),
        ],
        contribution=ContributionCheck(
            intro_span="Our contribution is a router that cuts training cost by 40%",
            problem="",
            verdict=ContributionVerdict.clear,
        ),
        undefined_terms=[
            TermUse(
                term="TRG",
                first_use_span="We train the router with the TRG objective on all tokens.",
                definition_span="The Token Routing Gain (TRG) objective scores each expert",
                verdict=TermVerdict.defined_later,
            )
        ],
        captions=[
            CaptionCheck(
                label="Figure 1",
                caption_span="Loss curves for SM-B and SM-L runs.",
                missing="SM-B and SM-L are never expanded in the caption.",
            )
        ],
        related_work=RelatedWorkCheck(
            listing_spans=["Smith et al. proposed dense routing for mixtures of experts."],
            engaging_span="Unlike these, we learn the router jointly with the experts.",
            verdict=RelatedWorkVerdict.engages,
        ),
    )
    base.update(overrides)
    return WritingOutput(**base)


# --------------------------------------------------------------------------- spans


def test_norm_straightens_quotes_dashes_and_ligatures():
    assert norm("“eﬃcient” –  model\n") == '"efficient" - model'


def test_find_span_respects_region_and_case_and_ellipsis():
    pt = paper_text(_paper())
    abstract = {0}
    assert find_span(pt, "sparsemix achieves STATE of the art", abstract) is not None
    assert find_span(pt, "We train the router with the TRG", abstract) is None
    assert find_span(pt, "Mixture models are costly … the TRG objective on all tokens") is not None
    # pieces under three words are not enough to anchor an elision
    assert find_span(pt, "Mixture … tokens") is None
    assert find_span(pt, "a sentence that is not in the paper at all") is None


def test_references_are_not_searchable():
    pt = paper_text(_paper())
    assert find_span(pt, "Smith. Dense routing. 2020.") is None


def test_span_cannot_straddle_two_blocks():
    pt = paper_text(_paper())
    assert find_span(pt, "at equal accuracy on GLUE. Smith et al. proposed") is None


# --------------------------------------------------------------------------- checks


def test_verified_output_keeps_findings_with_quotes():
    v = verify_output(_paper(), _output())
    kinds = [f.check for f in v.findings]
    assert kinds.count(WritingCheck.abstract_overclaim) == 2
    assert WritingCheck.undefined_term in kinds
    assert WritingCheck.caption_not_standalone in kinds
    assert WritingCheck.related_work_list not in kinds
    assert WritingCheck.unclear_contribution not in kinds
    assert v.contribution is not None and v.related_work == RelatedWorkVerdict.engages
    over = v.findings[0]
    assert over.location == "abstract" and "Table 1" in over.evidence[0]
    assert not v.dropped


def test_unverified_abstract_span_is_dropped():
    bad = AbstractClaimTrace(
        abstract_span="SparseMix beats every baseline by a wide margin",
        result_span="",
        gap="",
        verdict=AbstractClaimVerdict.not_found,
    )
    v = verify_output(_paper(), _output(abstract_claims=[bad]))
    assert not v.findings or all(f.check != WritingCheck.abstract_overclaim for f in v.findings)
    assert v.dropped[0].reason.startswith("abstract_span not found")


def test_result_span_from_the_abstract_is_rejected():
    item = AbstractClaimTrace(
        abstract_span="It also halves training cost.",
        result_span="We propose SparseMix, a routing method for mixture models.",
        gap="x",
        verdict=AbstractClaimVerdict.overstated,
    )
    v = verify_output(_paper(), _output(abstract_claims=[item]))
    assert v.dropped and "outside the abstract" in v.dropped[0].reason


@pytest.mark.parametrize(
    ("result_span", "gap", "verdict", "reason"),
    [
        ("", "", AbstractClaimVerdict.supported, "supported without"),
        ("Training cost falls by 40% relative", "", AbstractClaimVerdict.overstated, "gap"),
        ("Training cost falls by 40% relative", "", AbstractClaimVerdict.not_found, "not_found"),
    ],
)
def test_abstract_verdict_must_match_its_fields(result_span, gap, verdict, reason):
    item = AbstractClaimTrace(
        abstract_span="It also halves training cost.",
        result_span=result_span,
        gap=gap,
        verdict=verdict,
    )
    v = verify_output(_paper(), _output(abstract_claims=[item]))
    assert v.dropped and reason in v.dropped[0].reason
    assert not v.trace


def test_supported_and_unverifiable_claims_are_traced_not_flagged():
    items = [
        AbstractClaimTrace(
            abstract_span="We propose SparseMix, a routing method for mixture models.",
            result_span="Our contribution is a router that cuts training cost",
            gap="",
            verdict=AbstractClaimVerdict.supported,
        ),
        AbstractClaimTrace(
            abstract_span="It also halves training cost.",
            result_span="",
            gap="",
            verdict=AbstractClaimVerdict.unverifiable,
        ),
    ]
    v = verify_output(_paper(), _output(abstract_claims=items))
    assert len(v.trace) == 2
    assert all(f.check != WritingCheck.abstract_overclaim for f in v.findings)


def test_vague_contribution_needs_intro_quote_and_problem():
    c = ContributionCheck(
        intro_span="Our contribution is a router that cuts training cost by 40%",
        problem="It does not say what is new about the router.",
        verdict=ContributionVerdict.vague,
    )
    v = verify_output(_paper(), _output(contribution=c))
    f = next(f for f in v.findings if f.check == WritingCheck.unclear_contribution)
    assert f.location == "1 Introduction"
    outside = c.model_copy(update={"intro_span": "Unlike these, we learn the router jointly"})
    v = verify_output(_paper(), _output(contribution=outside))
    assert "introduction" in v.dropped[0].reason


def test_term_defined_later_uses_code_order():
    v = verify_output(_paper(), _output())
    f = next(f for f in v.findings if f.check == WritingCheck.undefined_term)
    assert f.detail.startswith("defined_later")
    assert "TRG objective on all tokens" in f.quote
    assert "Token Routing Gain (TRG)" in f.evidence[0]
    assert "3 Method" in f.evidence[0]


def test_term_first_use_is_the_earliest_occurrence_not_the_models_span():
    t = TermUse(
        term="TRG",
        first_use_span="The Token Routing Gain (TRG) objective scores each expert per token.",
        definition_span="",
        verdict=TermVerdict.never_defined,
    )
    # the earliest use is in the intro; the model's span is the definition itself
    v = verify_output(_paper(), _output(undefined_terms=[t]))
    f = next(f for f in v.findings if f.check == WritingCheck.undefined_term)
    assert "TRG objective on all tokens" in f.quote and f.location == "1 Introduction"
    # code found the parenthesised expansion later, so it is defined_later, not never_defined
    assert f.detail.startswith("defined_later")


def test_term_defined_at_first_use_is_dropped():
    p = _paper()
    p.sections[0].paragraphs = ["We use the Token Routing Gain (TRG) objective on all tokens."]
    t = TermUse(
        term="TRG",
        first_use_span="We use the Token Routing Gain (TRG) objective",
        definition_span="",
        verdict=TermVerdict.never_defined,
    )
    v = verify_output(p, _output(undefined_terms=[t]))
    assert all(f.check != WritingCheck.undefined_term for f in v.findings)
    assert any("at or before" in d.reason for d in v.dropped)


def test_term_must_be_in_its_span_and_long_enough():
    t = TermUse(
        term="GLUE",
        first_use_span="We train the router with the TRG objective on all tokens.",
        definition_span="",
        verdict=TermVerdict.never_defined,
    )
    short = t.model_copy(update={"term": "A"})
    v = verify_output(_paper(), _output(undefined_terms=[t, short]))
    reasons = [d.reason for d in v.dropped]
    assert "first_use_span does not contain the term" in reasons
    assert "term too short to locate" in reasons


def test_term_does_not_match_inside_longer_words():
    p = _paper()
    p.sections[2].paragraphs = ["The TRGX variant differs. TRG is never spelled out here."]
    t = TermUse(
        term="TRG",
        first_use_span="We train the router with the TRG objective on all tokens.",
        definition_span="",
        verdict=TermVerdict.never_defined,
    )
    v = verify_output(p, _output(undefined_terms=[t]))
    f = next(f for f in v.findings if f.check == WritingCheck.undefined_term)
    assert f.detail.startswith("never_defined")


def test_caption_must_come_from_the_named_label():
    wrong = CaptionCheck(
        label="Table 1",
        caption_span="Loss curves for SM-B and SM-L runs.",
        missing="x",
    )
    v = verify_output(_paper(), _output(captions=[wrong]))
    assert "not Table 1" in v.dropped[0].reason
    table = CaptionCheck(label="Table 1", caption_span="Results on GLUE.", missing="metric?")
    v = verify_output(_paper(), _output(captions=[table]))
    f = next(f for f in v.findings if f.check == WritingCheck.caption_not_standalone)
    assert f.location == "Table 1"
    prefixed = CaptionCheck(
        label="Figure 1", caption_span="Figure 1: Loss curves for SM-B", missing="x"
    )
    v = verify_output(_paper(), _output(captions=[prefixed]))
    assert any(f.check == WritingCheck.caption_not_standalone for f in v.findings)
    body = CaptionCheck(label="Figure 1", caption_span="Mixture models are costly.", missing="x")
    v = verify_output(_paper(), _output(captions=[body]))
    assert "any caption" in v.dropped[0].reason


def test_related_work_listing_needs_verified_spans():
    rw = RelatedWorkCheck(
        listing_spans=[
            "Smith et al. proposed dense routing for mixtures of experts.",
            "Jones et al. studied load balancing in large sparse models.",
            "An invented sentence that the paper never contains anywhere.",
        ],
        engaging_span="",
        verdict=RelatedWorkVerdict.mostly_lists,
    )
    v = verify_output(_paper(), _output(related_work=rw))
    f = next(f for f in v.findings if f.check == WritingCheck.related_work_list)
    assert f.location == "2 Related Work" and len(f.evidence) == 1
    none = rw.model_copy(update={"listing_spans": ["An invented sentence the paper lacks."]})
    v = verify_output(_paper(), _output(related_work=none))
    assert all(f.check != WritingCheck.related_work_list for f in v.findings)
    assert v.dropped


# --------------------------------------------------------------------------- LLM call


class FakeWriter:
    def __init__(self, malformed: int = 0) -> None:
        self.malformed = malformed
        self.calls = 0
        self.requests: list[TransportRequest] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls += 1
        self.requests.append(req)
        usage = TokenUsage(input_tokens=20_000, output_tokens=2_000)
        if self.calls <= self.malformed:
            return TransportResponse(None, usage, "end_turn", "bad json")
        return TransportResponse(_output(), usage, "end_turn")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        anthropic_api_key=None,
        llm_max_retries=2,
    )


def test_context_keeps_related_work_and_figures_but_not_references():
    msg = writing_user_message(_paper())
    assert "Related Work" in msg and "Loss curves for SM-B" in msg
    assert "Dense routing. 2020." not in msg and "85.1" in msg


def test_check_paper_retries_malformed_output_and_counts_spend(settings):
    fake = FakeWriter(malformed=1)
    r = check_paper(
        LLMClient(settings, transport=fake),
        _paper(),
        private=False,
        prompt=load_writing_prompt("writing_v1"),
        model="claude-opus-5-5",
        max_retries=2,
    )
    assert fake.calls == 2
    assert r.token_usage.cost_usd == pytest.approx(2 * (20_000 * 4 + 2_000 * 20) / 1e6)
    assert len(r.findings) == 4 and r.prompt_version == "writing_v1"


def _write_parsed(tmp_path, papers: list[ParsedPaper]):
    d = tmp_path / "parsed"
    d.mkdir()
    for p in papers:
        (d / f"{p.paper_id}.json").write_text(p.model_dump_json())
    return d


def test_is_private_treats_openreview_dev_papers_as_private():
    assert is_private(_paper("or-AbC123", "data/openreview/pdf/AbC123.pdf"))
    assert not is_private(_paper("2401.00001", "data/papers/arxiv/2401.00001.pdf"))


def test_run_writing_routes_private_papers_and_caches(settings, tmp_path):
    d = _write_parsed(
        tmp_path,
        [_paper("2401.00001"), _paper("own-x-deadbeef", "data/papers/own/x.pdf")],
    )
    fake = FakeWriter()
    run_writing(d, settings=settings, client=LLMClient(settings, transport=fake))
    assert (report_dir(settings, False) / "2401.00001.md").exists()
    assert (report_dir(settings, True) / "own-x-deadbeef.json").exists()
    assert not (report_dir(settings, False) / "own-x-deadbeef.json").exists()
    run_writing(d, settings=settings, client=LLMClient(settings, transport=fake))
    assert fake.calls == 1  # identical text: the second paper is an LLM cache hit
    assert "own-x-deadbeef" not in fake.requests[0].user  # paper ids are never sent


def test_run_writing_stops_at_budget(settings, tmp_path):
    d = _write_parsed(tmp_path, [_paper("2401.00001"), _paper("2401.00002")])
    s = settings.model_copy(update={"writing_budget_usd": 0.01})
    fake = FakeWriter()
    run_writing(d, settings=s, client=LLMClient(s, transport=fake))
    assert fake.calls == 1
    assert not (report_dir(s, False) / "2401.00002.json").exists()


# --------------------------------------------------------------------------- reports


def _report(pid: str, settings: Settings) -> WritingReport:
    return check_paper(
        LLMClient(settings, transport=FakeWriter()),
        _paper(pid),
        private=False,
        prompt=load_writing_prompt("writing_v1"),
        model="claude-opus-5-5",
        max_retries=0,
    )


def test_render_report_lists_trace_and_findings(settings):
    md = render_report(_report("p1", settings))
    assert "## Abstract claims traced to results" in md
    assert "overstated" in md and 'quote: "Loss curves for SM-B' in md
    assert "## Related work lists prior work without engaging (0)" in md


def test_stratified_sample_spreads_over_checks(settings):
    reports = [_report(f"p{i}", settings) for i in range(3)]
    picked = stratified_sample(reports, 4, seed=1)
    assert len({f.check for _, f in picked}) == 3
    assert stratified_sample(reports, 4, seed=1) == picked
    md = summary_markdown(reports, seed=1, n_sample=4)
    assert "| **total** |" in md and md.count("grade:") == 4


def test_run_summary_writes_public_and_private(settings):
    write_report(_report("2401.00001", settings), settings)
    priv = _report("own-x-deadbeef", settings).model_copy(update={"private": True})
    write_report(priv, settings)
    run_summary(seed=7, n_sample=3, settings=settings)
    pub = (report_dir(settings, False) / "SUMMARY.md").read_text()
    assert "2401.00001" in pub and "own-x-deadbeef" not in pub
    assert (report_dir(settings, True) / "SUMMARY.md").exists()


def test_intro_continues_through_an_unnumbered_split():
    p = _paper()
    p.sections.insert(
        1,
        Section(
            heading="Submitted to a conference. Do not distribute.",
            paragraphs=["In short, our contribution is a faster router for sparse mixtures."],
        ),
    )
    c = ContributionCheck(
        intro_span="our contribution is a faster router for sparse mixtures",
        problem="",
        verdict=ContributionVerdict.clear,
    )
    v = verify_output(p, _output(contribution=c))
    assert v.contribution is not None and not v.dropped
