from reviewer.methods.checker import items_for
from reviewer.methods.consensus import bucket_counts, combine, concerns_match
from reviewer.methods.evaluate import raised_concerns
from reviewer.methods.report import render_review
from reviewer.models import (
    CheckedConcern,
    CheckedFinding,
    MethodsVerdict,
    PaperMethodsReview,
    QuoteCheck,
    TokenUsage,
)

V = MethodsVerdict
ITEMS = items_for("methods_v2")[:4]


def _f(item, v, quote="q one two three four five"):
    return CheckedFinding(
        item=item,
        verdict=v,
        model_verdict=v,
        rationale=f"r {v.value}",
        concern=f"c {item.value}" if v == V.no else "",
        quotes=[QuoteCheck(quote=quote, verified=True)],
    )


def _c(text, quote="a verified passage of six words here", ok=True):
    return CheckedConcern(concern=text, quotes=[QuoteCheck(quote=quote, verified=ok)], verified=ok)


def _r(verdicts, concerns=()):
    return PaperMethodsReview(
        paper_id="or-S",
        tier="openreview",
        model="m",
        prompt_version="methods_v2+repair_v1",
        effort="medium",
        findings=[_f(i, v) for i, v in zip(ITEMS, verdicts, strict=True)],
        other_concerns=list(concerns),
        token_usage=TokenUsage(cost_usd=0.1),
    )


def test_items_confirmed_possible_and_disagreement():
    r = combine(_r([V.no, V.no, V.yes, V.yes]), _r([V.no, V.unclear, V.yes, V.unclear]))
    by = {f.item: f for f in r.findings}
    assert by[ITEMS[0]].agreement == "confirmed" and by[ITEMS[0]].verdict == V.no
    assert by[ITEMS[1]].agreement == "possible" and by[ITEMS[1]].verdict == V.no
    assert by[ITEMS[2]].agreement == "" and by[ITEMS[2]].verdict == V.yes
    assert by[ITEMS[3]].verdict == V.unclear and "runs disagree" in by[ITEMS[3]].rationale
    assert r.runs == 2 and abs(r.token_usage.cost_usd - 0.2) < 1e-9
    # a `no` only in the second run is still possible, with that run's text
    r2 = combine(_r([V.yes] * 4), _r([V.no, V.yes, V.yes, V.yes]))
    f = r2.findings[0]
    assert f.agreement == "possible" and f.concern == f"c {ITEMS[0].value}"


def test_other_concerns_matching_is_conservative():
    shared = "the baseline was trained for fewer epochs than the method"
    assert concerns_match(_c("x", quote=shared), _c("totally different words", quote=shared))
    assert concerns_match(
        _c("Only two datasets evaluated despite general claims", quote="p1 aaaa bbbb cccc"),
        _c("Evaluation uses only two datasets, general claims unsupported", quote="p2 dddd eeee"),
    )
    assert not concerns_match(
        _c("Judge model errors are not measured", quote="p1 aaaa bbbb"),
        _c("Threshold chosen on the test set", quote="p2 cccc dddd"),
    )
    a = _r([V.yes] * 4, [_c("Judge model errors are not measured", quote="judge passage one two")])
    b = _r(
        [V.yes] * 4,
        [
            _c("GPT-4 judge errors never measured", quote="judge passage one two"),
            _c("Unrelated: threshold chosen on test", quote="other passage three four"),
            _c("unverified one", ok=False),
        ],
    )
    r = combine(a, b)
    # the unverified concern is dropped
    assert [c.agreement for c in r.other_concerns] == ["confirmed", "possible"]
    assert bucket_counts(r) == {
        "items_confirmed": 0,
        "items_possible": 0,
        "other_confirmed": 1,
        "other_possible": 1,
    }


def test_possible_findings_are_not_strict_and_render_separately():
    r = combine(
        _r([V.no, V.no, V.yes, V.yes], [_c("Judge errors unmeasured", quote="jp one two three")]),
        _r([V.no, V.yes, V.yes, V.yes]),
    )
    raised = {x.key: x for x in raised_concerns(r)}
    assert raised[ITEMS[0].value].strict and not raised[ITEMS[1].value].strict
    assert raised[ITEMS[1].value].status == "possible"
    assert raised["other:0"].status == "other_possible" and not raised["other:0"].strict
    text = "\n".join(render_review(r, "T"))
    confirmed = text.index("### Confirmed findings")
    possible = text.index("### Possible findings (raised in only one of two runs)")
    assert confirmed < text.index(f"**{ITEMS[0].value}: no**") < possible
    assert text.index(f"**{ITEMS[1].value}: no**") > possible
    assert text.index("Judge errors unmeasured") > possible
