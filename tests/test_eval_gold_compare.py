import pytest

from reviewer.eval.checks import build_idf, check_near_duplicates, max_similarities
from reviewer.eval.gold_compare import (
    check_numbers,
    claims_haystack,
    closest_claim,
    compare_gold,
    contains_number,
    number_cores,
)
from reviewer.eval.schemas import GoldAlignment, GoldComparison, NumberCheck

from .test_eval_helpers import gold, paper_claims

# --------------------------------------------------------------------------- number cores


@pytest.mark.parametrize(
    ("token", "cores"),
    [
        ("48.6%", ["48.6"]),
        ("−0.016", ["0.016"]),
        ("-0.016", ["0.016"]),
        ("+3.2", ["3.2"]),
        ("±0.4", ["0.4"]),
        ("3x", ["3"]),
        ("2.5×", ["2.5"]),
        ("175B", ["175"]),
        ("1,000", ["1,000"]),
        ("3–5%", ["3", "5"]),
        ("10-shot", ["10"]),
    ],
)
def test_number_cores(token, cores):
    assert number_cores(token) == cores


@pytest.mark.parametrize(
    ("core", "haystack", "found"),
    [
        ("0.5", "a gain of 0.503 points", False),
        ("0.5", "reaches 10.5 BLEU", False),
        ("0.5", "reaches 0.5 BLEU", True),
        ("0.5", "(0.5%)", True),
        ("0.5", "gap of 0.5.", True),
        ("0.5", "gaps of 0.5, 0.7", True),
        ("0.016", "drops by −0.016", True),
        ("0.016", "drops by -0.016", True),
        ("48.6", "48.6% accuracy", True),
        ("48.6", "48.6 accuracy", True),
        ("48.6", "148.6 accuracy", False),
        ("48", "48.6 accuracy", False),
        ("1,000", "1000 users", False),
    ],
)
def test_contains_number_boundaries(core, haystack, found):
    assert contains_number(core, haystack) is found


def test_unicode_minus_and_percent_equivalence():
    assert [c.found for c in check_numbers("−0.016 and 48.6%", "got -0.016 and 48.6")] == [
        True,
        True,
    ]
    assert [c.found for c in check_numbers("-0.016 and 48.6", "got −0.016 and 48.6%")] == [
        True,
        True,
    ]


def test_check_numbers_uses_number_heuristic():
    checks = check_numbers("GPT-4 in Table 2 scores 71.3% over 2019 baselines", "71.3")
    assert [(c.number, c.cores, c.found) for c in checks] == [("71.3%", ["71.3"], True)]
    assert check_numbers(None, "x") == [] and check_numbers("no numbers", "x") == []


def test_haystack_covers_all_fields_and_summary():
    pc = paper_claims("p", ["S one."])
    pc.one_line_summary = "Summary with 9.99 in it."
    c = pc.claims[0]
    c.method, c.problem, c.setting = "m 1.01", "p 1.02", "s 1.03"
    c.novelty_assertion, c.evidence = "n 1.04", ["Table 1.05"]
    hay = claims_haystack(pc)
    for n in ("9.99", "1.01", "1.02", "1.03", "1.04", "1.05", "28.4"):
        assert contains_number(n, hay)


# --------------------------------------------------------------------------- alignment + recall


def test_closest_claim():
    cands = ["A geometry benchmark of 500 problems.", "Sparse attention cuts memory."]
    idf = build_idf(cands)
    idx, score = closest_claim("Sparse attention memory savings", cands, idf)
    assert idx == 1 and 0 < score <= 1
    assert closest_claim("anything", [], idf) == (None, 0.0)


def test_compare_gold_alignment_and_recall():
    pc = paper_claims(
        "p",
        ["Sparse attention cuts memory by 40%.", "A geometry benchmark of 500 problems."],
        result="memory down 40%, accuracy −0.016",
    )
    g = gold(
        "p",
        ["New benchmark: 500 geometry problems.", "Attention sparsity saves memory."],
        results=["500 problems, 0.5 accuracy", "-0.016 accuracy and 40% memory, 3x faster"],
    )
    cmp = compare_gold(g, pc, build_idf([c.statement for c in pc.claims]))
    assert [a.closest_index for a in cmp.alignments] == [1, 0]
    assert [[n.found for n in a.numbers] for a in cmp.alignments] == [
        [True, False],
        [True, True, False],
    ]
    assert (cmp.n_found, cmp.n_numbers) == (3, 5)
    assert cmp.recall == pytest.approx(0.6)


def test_recall_none_without_gold_numbers_and_without_claims():
    g = gold("p", ["No numbers here."], results=["qualitatively better"])
    assert compare_gold(g, paper_claims("p", ["x"]), {}).recall is None
    g2 = gold("p", ["Claim."], results=["12.5%"])
    cmp = compare_gold(g2, None, {})
    assert cmp.alignments[0].closest_index is None
    assert cmp.recall == 0.0


def test_recall_math():
    def a(*found: bool) -> GoldAlignment:
        return GoldAlignment(
            gold_index=0,
            closest_index=0,
            score=0.5,
            numbers=[NumberCheck(number="1.0", cores=["1.0"], found=f) for f in found],
        )

    cmp = GoldComparison(alignments=[a(True, False, True), a(), a(False)])
    assert (cmp.n_found, cmp.n_numbers) == (2, 4) and cmp.recall == 0.5


# --------------------------------------------------------------------------- similarity


def test_max_similarities():
    s = ["Sparse attention cuts memory.", "Sparse attention cuts memory a lot.", "Geometry."]
    sims = max_similarities(s, build_idf(s))
    assert [j for j, _ in sims] == [1, 0, 0]
    assert sims[0][1] > 0.6 and sims[2][1] == 0.0
    assert max_similarities(["only"], {}) == [(None, 0.0)]


def test_threshold_configurable_default_0_6():
    idf = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}
    two = ["alpha beta", "alpha gamma"]  # cosine 0.5
    three = ["alpha beta gamma", "alpha beta delta"]  # cosine 2/3
    assert check_near_duplicates(two, idf) == []
    assert len(check_near_duplicates(three, idf)) == 2  # 0.67 >= 0.6, < old 0.9
    assert len(check_near_duplicates(two, idf, threshold=0.4)) == 2
    assert check_near_duplicates(three, idf, threshold=0.9) == []


def test_setting_numbers_counted_separately():
    from reviewer.eval.checks import build_idf
    from reviewer.eval.gold_compare import compare_gold
    from reviewer.models import GoldClaim, GoldClaims

    from .test_eval_helpers import paper_claims

    gold = GoldClaims(
        paper_id="p",
        source="author",
        claims=[
            GoldClaim(
                statement="s", result="gain of 48.6%", setting="24 equities, 1,000 replicates"
            )
        ],
    )
    comp = compare_gold(
        gold, paper_claims("p", ["X gains 48.6% on 24 equities.", "Y"]), build_idf([])
    )
    assert (comp.n_found, comp.n_numbers) == (1, 1)
    assert (comp.n_found_all, comp.n_numbers_all) == (2, 3)
