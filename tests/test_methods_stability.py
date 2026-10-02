import pytest
import typer

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.methods.checker import items_for
from reviewer.methods.quotes import QuoteIndex
from reviewer.methods.stability import (
    Flips,
    code_agreement,
    load_runs,
    render_stability,
    run_stability,
    strict_flips,
    verdict_flips,
)
from reviewer.methods.structured import derive, implied_verdict
from reviewer.models import (
    MethodsVerdict,
    StructuredFinding,
    StructuredMethodsOutput,
    TokenUsage,
    Violation,
)
from tests.test_methods import BAD, GOOD, _paper
from tests.test_methods import FakeLLM as FakeCurrent

V = MethodsVerdict


def _sf(item="seeds_variance", applies=True, aq="", violations=(), compliance=()):
    return StructuredFinding(
        item=item,
        applies=applies,
        applies_quote=aq,
        violations=list(violations),
        compliance_quotes=list(compliance),
    )


def _v(code="seeds_variance__variance_missing", quote=GOOD):
    return Violation(code=code, quote=quote, explanation="single run")


def test_derive_rules():
    ix = QuoteIndex.for_paper(_paper())
    assert derive(_sf(violations=[_v()]), ix).verdict == V.no
    d = derive(_sf(violations=[_v()]), ix)
    assert d.violation_codes == ["seeds_variance__variance_missing"] and d.concern == "single run"
    assert derive(_sf(compliance=[GOOD]), ix).verdict == V.yes
    # an unverified violation does not count; a verified compliance quote then decides
    assert derive(_sf(violations=[_v(quote=BAD)], compliance=[GOOD]), ix).verdict == V.yes
    bad = derive(_sf(violations=[_v(quote=BAD)]), ix)
    assert bad.verdict == V.unclear and bad.model_verdict == V.no and bad.downgraded
    assert derive(_sf(), ix).verdict == V.unclear and not derive(_sf(), ix).downgraded
    assert derive(_sf(applies=False, aq=GOOD), ix).verdict == V.na
    assert derive(_sf(applies=False, aq=BAD), ix).verdict == V.unclear
    # a code from another item is ignored
    other = _sf(violations=[_v(code="leakage__tuned_on_test")])
    assert implied_verdict(other) == V.unclear and derive(other, ix).verdict == V.unclear


def test_flip_metrics():
    from reviewer.models import CheckedFinding, PaperMethodsReview

    def rev(verdicts, codes=()):
        fs = [
            CheckedFinding(
                item=i,
                verdict=v,
                model_verdict=v,
                rationale="",
                concern="",
                quotes=[],
                violation_codes=list(codes) if v == V.no else [],
            )
            for i, v in zip(items_for("methods_v2"), verdicts, strict=False)
        ]
        return PaperMethodsReview(
            paper_id="p",
            tier="t",
            model="m",
            prompt_version="v",
            effort="e",
            findings=fs,
            other_concerns=[],
            token_usage=TokenUsage(),
        )

    a = rev([V.no, V.yes, V.unclear])
    b = rev([V.no, V.unclear, V.unclear], codes=["x"])
    pairs = [(a, b)]
    assert verdict_flips(pairs) == Flips(3, 1)
    assert strict_flips(pairs) == Flips(3, 0)  # yes -> unclear is not a `no` flip
    assert code_agreement(pairs) == Flips(1, 1)  # item 0 is `no` in both, codes differ
    lo, hi = Flips(100, 5).wilson()
    assert 0.02 < lo < 0.05 < hi < 0.12


class FakeStructured:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls += 1
        fs = [_sf(item=i.value, compliance=[GOOD]) for i in items_for("methods_v2s")]
        return TransportResponse(
            StructuredMethodsOutput(findings=fs, other_concerns=[]),
            TokenUsage(input_tokens=1000, output_tokens=100),
            "end_turn",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        anthropic_api_key=None,
        llm_max_retries=1,
    )
    s.openreview_parsed_dir.mkdir(parents=True)
    for sid in ("S1", "S2"):
        (s.openreview_parsed_dir / f"or-{sid}.json").write_text(
            _paper(f"or-{sid}").model_copy(update={"title": sid}).model_dump_json()
        )
    return s


def test_run_stability_replicates_are_fresh_calls(settings):
    fake = FakeStructured()
    run_stability("structured", ["S1", "S2"], 2, 5.0, settings, LLMClient(settings, transport=fake))
    assert fake.calls == 4  # rep1 is not served from rep0's cache entry
    runs = load_runs(settings)
    assert set(runs["structured"]) == {0, 1} and len(runs["structured"][1]) == 2
    assert all(f.verdict == V.yes for r in runs["structured"][0].values() for f in r.findings)
    text = render_stability(runs)
    assert "| structured | 2 | 0/20 = **0%**" in text
    # rerun: everything on disk, no calls
    run_stability("structured", ["S1", "S2"], 2, 5.0, settings, LLMClient(settings, transport=fake))
    assert fake.calls == 4


def test_run_stability_current_condition_uses_v2(settings):
    seen: list[str] = []
    fake = FakeCurrent()

    def spy(req):
        seen.append(req.system[:40])
        # FakeCurrent answers with v1 items; give it v2 items instead
        from reviewer.models import MethodsOutput
        from tests.test_methods import _finding

        if req.schema is MethodsOutput:
            fs = [_finding(i) for i in items_for("methods_v2")]
            return TransportResponse(
                MethodsOutput(findings=fs, other_concerns=[]),
                TokenUsage(input_tokens=10, output_tokens=10),
                "end_turn",
            )
        return fake(req)

    run_stability("current", ["S1"], 2, 5.0, settings, LLMClient(settings, transport=spy))
    runs = load_runs(settings)
    assert len(runs["current"][0]) == len(runs["current"][1]) == 1
    assert len(runs["current"][0]["or-S1"].findings) == 10
    with pytest.raises(typer.BadParameter):
        run_stability("temperature0", ["S1"], settings=settings)
