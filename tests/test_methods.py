import pytest
import yaml

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.methods.checker import (
    ITEMS,
    MethodsError,
    assemble,
    check_paper,
    load_methods_prompt,
    output_problems,
)
from reviewer.methods.commands import run_check, run_eval
from reviewer.methods.evaluate import (
    match_problems,
    outcomes,
    precision_sample,
    raised_concerns,
    recall,
)
from reviewer.methods.quotes import QuoteIndex, fragments, squash
from reviewer.models import (
    ChecklistFinding,
    ConcernCheck,
    ConcernCheckOutput,
    CritiqueEntry,
    CritiqueMatch,
    CritiqueMatchLabel,
    CritiqueMatchOutput,
    Critiques,
    CritiqueScope,
    MethodsOutput,
    MethodsVerdict,
    OtherConcern,
    ParsedPaper,
    QuoteRepair,
    QuoteRepairOutput,
    Section,
    Table,
    TokenUsage,
)

RESULTS = (
    "We train each model once with a single seed and report the final accuracy on the test "
    "split. Our method reaches 69.2% accuracy, while the base- line reaches 64.2% under "
    "the “same” budget."
)
GOOD = "We train each model once with a single seed"
BAD = "We train every model five times with different seeds"


def _paper(pid: str = "2501.00001") -> ParsedPaper:
    return ParsedPaper(
        paper_id=pid,
        source_path="x.pdf",
        title="Early Exit for Reasoning",
        abstract="We stop reasoning early when confident, saving 40% of tokens.",
        sections=[
            Section(heading="Introduction", paragraphs=["Reasoning models overthink a lot."]),
            Section(heading="Related Work", paragraphs=["Prior work on early exit exists."]),
            Section(heading="Experiments", paragraphs=[RESULTS]),
        ],
        tables=[
            Table(
                label="Table 2", caption="Main results", rows=[["Method", "Acc"], ["DEER", "69.2"]]
            )
        ],
    )


# --------------------------------------------------------------------------- quotes


def test_squash_forgives_parse_noise_but_keeps_decimals():
    assert squash("base- line “same” ﬁne") == squash('baseline "same" fine')
    assert squash("69.2%") != squash("692%")


def test_verify_quotes():
    ix = QuoteIndex.for_paper(_paper())
    assert ix.verify(GOOD)
    assert ix.verify('the baseline reaches 64.2% under the "same" budget')  # hyphen, quotes
    assert ix.verify("We train each model ... report the final accuracy")
    assert not ix.verify("report the final accuracy ... We train each model")  # order
    assert not ix.verify(BAD)
    assert not ix.verify("a single seed")  # too short
    assert not ix.verify("We train each model once ... accuracy")  # 1-word fragment
    assert ix.verify("Table 2: Main results Method | Acc DEER")  # table as rendered
    assert fragments("a b … c d [...] e") == ["a b", "c d", "e"]


def test_verify_rejects_fragments_far_apart():
    far = "alpha beta gamma delta " + "filler words here " * 100 + "epsilon zeta eta theta"
    ix = QuoteIndex(far)
    assert ix.verify("alpha beta gamma delta filler words")
    assert not ix.verify("alpha beta gamma delta ... epsilon zeta eta theta")


# --------------------------------------------------------------------------- checker


def _finding(item, verdict="no", quotes=(GOOD,), concern="c") -> ChecklistFinding:
    return ChecklistFinding(
        item=item, verdict=verdict, rationale="r", quotes=list(quotes), concern=concern
    )


def _output(bad_items=(), other_quotes=(GOOD,), drop=None) -> MethodsOutput:
    fs = [
        _finding(i, quotes=(BAD,) if i.value in bad_items else (GOOD,))
        for i in ITEMS
        if i.value != drop
    ]
    return MethodsOutput(
        findings=fs, other_concerns=[OtherConcern(concern="one model", quotes=list(other_quotes))]
    )


def test_output_problems():
    assert output_problems(_output()) == []
    assert "missing checklist items: leakage" in output_problems(_output(drop="leakage"))
    dup = _output()
    dup.findings.append(_finding(ITEMS[0]))
    assert any("more than once" in p for p in output_problems(dup))


def test_assemble_downgrades_unverified():
    out = _output(bad_items=("leakage",), other_quotes=(BAD,))
    out.findings[0] = _finding(ITEMS[0], verdict="unclear", quotes=())
    findings, concerns = assemble(out, QuoteIndex.for_paper(_paper()), {})
    by = {f.item.value: f for f in findings}
    assert by["leakage"].verdict == MethodsVerdict.unclear and by["leakage"].downgraded
    assert by["leakage"].model_verdict == MethodsVerdict.no
    assert by[ITEMS[0].value].verdict == MethodsVerdict.unclear
    assert not by[ITEMS[0].value].downgraded  # already unclear
    assert by["seeds_variance"].verdict == MethodsVerdict.no
    assert not concerns[0].verified
    # a repaired quote rescues the finding
    findings, _ = assemble(out, QuoteIndex.for_paper(_paper()), {"leakage": [GOOD]})
    leak = next(f for f in findings if f.item.value == "leakage")
    assert leak.verdict == MethodsVerdict.no and leak.repaired and not leak.downgraded


class FakeLLM:
    """Checker: first `bad` calls miss an item; leakage and other:0 quotes fail. Repair: fixes
    them when `fix`. Matcher and verifier answer from the request."""

    def __init__(self, bad: int = 0, fix: bool = True) -> None:
        self.bad, self.fix = bad, fix
        self.calls: dict[str, int] = {}

    def __call__(self, req: TransportRequest) -> TransportResponse:
        name = req.schema.__name__
        self.calls[name] = self.calls.get(name, 0) + 1
        if name == "MethodsOutput":
            drop = "leakage" if self.calls[name] <= self.bad else None
            parsed = _output(bad_items=("leakage",), other_quotes=(BAD,), drop=drop)
        elif name == "QuoteRepairOutput":
            q = [GOOD] if self.fix else []
            parsed = QuoteRepairOutput(
                repairs=[QuoteRepair(key="leakage", quotes=q), QuoteRepair(key="other:0", quotes=q)]
            )
        elif name == "CritiqueMatchOutput":
            parsed = CritiqueMatchOutput(
                matches=[
                    CritiqueMatch(
                        critique_index=0,
                        scope="methodology",
                        label="match",
                        finding_ids=["other:0"],
                        reason="same",
                    ),
                    CritiqueMatch(
                        critique_index=1, scope="design", label="none", finding_ids=[], reason="x"
                    ),
                ]
            )
        else:
            ids = [line.split('"')[1] for line in req.user.splitlines() if "<concern id=" in line]
            parsed = ConcernCheckOutput(
                checks=[ConcernCheck(concern_id=i, validity="valid", reason="ok") for i in ids]
            )
        return TransportResponse(
            parsed=parsed,
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
        methods_prompt="methods_v1",  # the fakes answer with v1's nine items
    )


def _check(settings, fake, repair=True, paper=None):
    return check_paper(
        LLMClient(settings, transport=fake),
        paper or _paper(),
        tier="lab",
        prompt=load_methods_prompt("methods_v1"),
        repair_prompt=load_methods_prompt("repair_v1") if repair else None,
        model="claude-opus-5-5",
        effort="medium",
        max_retries=2,
    )


def test_check_paper_retries_and_repairs(settings):
    fake = FakeLLM(bad=1)
    review = _check(settings, fake)
    assert fake.calls == {"MethodsOutput": 2, "QuoteRepairOutput": 1}
    assert not any(f.downgraded for f in review.findings)
    assert review.other_concerns[0].verified and review.other_concerns[0].repaired
    assert review.prompt_version == "methods_v1+repair_v1"


def test_check_paper_downgrades_when_repair_fails(settings):
    review = _check(settings, FakeLLM(fix=False))
    leak = next(f for f in review.findings if f.item.value == "leakage")
    assert leak.downgraded and leak.verdict == MethodsVerdict.unclear
    assert not review.other_concerns[0].verified
    no_repair = _check(settings, FakeLLM(), repair=False)
    assert sum(f.downgraded for f in no_repair.findings) == 1
    with pytest.raises(MethodsError):
        _check(settings, FakeLLM(bad=99), paper=_paper().model_copy(update={"title": "Other"}))


def test_check_prompt_never_contains_critiques(settings):
    seen: list[str] = []
    fake = FakeLLM()

    def spy(req):
        seen.append(req.user)
        return fake(req)

    _check(settings, spy)
    assert all("<critique" not in u and "Related Work" not in u for u in seen)


# --------------------------------------------------------------------------- evaluation


def _critiques(pid="2501.00001") -> Critiques:
    return Critiques(
        paper_id=pid,
        source_document="x",
        entries=[
            CritiqueEntry(limitation="Only one model", source_block="list"),
            CritiqueEntry(limitation="Slow", explanation="expensive", source_block="list"),
        ],
    )


def test_raised_and_recall(settings):
    review = _check(settings, FakeLLM(fix=False))
    raised = {r.key: r for r in raised_concerns(review)}
    assert raised["leakage"].status == "downgraded" and not raised["leakage"].strict
    assert raised["other:0"].status == "other_unverified"
    assert raised["seeds_variance"].strict
    matches = CritiqueMatchOutput.model_validate(
        FakeLLM()(
            TransportRequest("m", "s", "u", CritiqueMatchOutput, 1, "medium")
        ).parsed.model_dump()
    ).matches
    items = outcomes(_critiques(), review, matches)
    assert items[0].lenient == CritiqueMatchLabel.match
    assert items[0].strict == CritiqueMatchLabel.none  # matched finding was unverified
    assert recall(items, "lenient").full == 0.5
    assert recall(items, "lenient", CritiqueScope.methodology).full == 1.0
    assert recall(items, "strict").n == 2 and recall(items, "strict").match == 0


def test_match_problems():
    out = CritiqueMatchOutput(
        matches=[
            CritiqueMatch(
                critique_index=0, scope="design", label="match", finding_ids=[], reason="x"
            ),
            CritiqueMatch(
                critique_index=0, scope="design", label="none", finding_ids=["zz"], reason="x"
            ),
        ]
    )
    problems = match_problems(out, 2, {"leakage"})
    assert any("one entry per critique" in p for p in problems)
    assert any("unknown finding ids" in p for p in problems)
    assert any("needs finding_ids" in p for p in problems)


def test_precision_sample_is_seeded_and_strict(settings):
    reviews = [_check(settings, FakeLLM(fix=False))]
    a, b = precision_sample(reviews, 5), precision_sample(reviews, 5)
    assert a == b and len(a) == 5 and all(r.strict for r in a)


def test_run_check_and_eval_end_to_end(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True)
    pid = "2501.00001"
    (settings.data_dir / "corpus.yaml").write_text(
        yaml.safe_dump({"papers": [{"paper_id": pid, "tier": "lab", "title": "T"}]})
    )
    settings.parsed_dir.mkdir()
    (settings.parsed_dir / f"{pid}.json").write_text(_paper(pid).model_dump_json())
    settings.critiques_dir.mkdir(parents=True)
    (settings.critiques_dir / f"{pid}.yaml").write_text(
        yaml.safe_dump(_critiques(pid).model_dump(mode="json"))
    )
    fake = FakeLLM()
    client = LLMClient(settings, transport=fake)
    reviews = run_check(settings=settings, client=client)
    assert len(reviews) == 1 and (settings.methods_dir / f"{pid}.json").exists()
    assert (settings.private_reports_dir / "methods_report.md").exists()
    items, checks = run_eval(settings=settings, client=client, precision_n=3)
    assert len(items) == 2 and len(checks) == 3
    report = (settings.private_reports_dir / "methods_eval.md").read_text()
    assert "## Recall" in report and "valid 3" in report
    assert (settings.private_reports_dir / "methods_precision_sample.md").exists()
    # rerun is served from disk, no new calls
    before = dict(fake.calls)
    run_check(settings=settings, client=client)
    assert fake.calls == before
