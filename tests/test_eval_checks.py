import pytest

from reviewer.eval.checks import (
    VAGUE_PHRASES,
    build_idf,
    find_numbers,
    has_number,
    jaccard,
    run_checks,
    tfidf_cosine,
)
from reviewer.models import FlagKind

from .test_eval_helpers import paper_claims, parsed


def _kinds(flags, kind):
    return [f for f in flags if f.kind == kind]


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    "text",
    [
        "reaches 28.4 BLEU",
        "improves accuracy by 5%",
        "is 3x faster",
        "a 2.5× speedup",
        "trained on 8 GPUs",
        "context of 2048 tokens",
        "on 12 benchmarks",
        "with 175B parameters",
        "in the 10-shot setting",
        "uses 1 GPU",
        "learning rate 1e-4",
        "over 1,000 users",
    ],
)
def test_has_number_positive(text):
    assert has_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "introduced in 2017",
        "published in 1999 and 2023.",
        "fine-tunes GPT-4 and T5",
        "evaluated on Llama-3.1-8B",
        "see Table 2 and Figure 3",
        "a 3D scene model",
        "the 2nd stage",
        "in phase 1 of the method",
        "no numbers here at all",
    ],
)
def test_has_number_negative(text):
    assert not has_number(text), find_numbers(text)


# --------------------------------------------------------------------------- vague phrases


def test_vague_phrase_list_is_a_module_tuple():
    assert isinstance(VAGUE_PHRASES, tuple) and len(VAGUE_PHRASES) >= 25


@pytest.mark.parametrize(
    ("statement", "phrase"),
    [
        ("Our Novel Framework improves performance on many benchmarks.", "novel framework"),
        ("X significantly beats Y.", "significantly"),
        ("It shows promising results with 3 seeds.", "promising results"),
        ("The method is robust to noise.", "robust"),
        ("It outperforms existing methods on 12 datasets.", "outperforms existing methods"),
        ("This paves the way for agents.", "paves the way"),
        ("Integrates seamlessly into pipelines.", "seamlessly"),
    ],
)
def test_vague_phrase_positive(statement, phrase):
    flags = _kinds(
        run_checks(paper_claims("p", [statement, "Other claim 28.4 BLEU."]), None),
        FlagKind.vague_phrase,
    )
    assert any(f.claim_index == 0 and phrase in f.detail for f in flags)


def test_vague_phrase_word_boundaries():
    # "robustness" / "effectively" / "significantly" inside words must not match partials
    s = "Adversarial robustness of Transformer-XL on WikiText-103 perplexity 18.3."
    assert not _kinds(
        run_checks(paper_claims("p", [s, "Other 2 claims."]), None), FlagKind.vague_phrase
    )


def test_sota_without_number_flagged():
    flags = run_checks(paper_claims("p", ["It achieves state-of-the-art results.", "Other."]), None)
    assert any("state-of-the-art" in f.detail for f in _kinds(flags, FlagKind.vague_phrase))


def test_sota_with_number_not_flagged():
    flags = run_checks(
        paper_claims("p", ["It achieves state-of-the-art 28.4 BLEU on WMT14 En-De.", "Other."]),
        None,
    )
    assert not _kinds(flags, FlagKind.vague_phrase)


def test_sota_acronym_without_number():
    flags = run_checks(paper_claims("p", ["New SOTA on GLUE.", "Other."]), None)
    assert any("sota" in f.detail for f in _kinds(flags, FlagKind.vague_phrase))


# --------------------------------------------------------------------------- missing numbers


def test_missing_numbers_flagged_when_abstract_has_numbers():
    pc = paper_claims("p", ["A claim with no digits.", "Another one."], result="it works")
    flags = _kinds(
        run_checks(pc, parsed("p", abstract="We get 28.4 BLEU.")), FlagKind.missing_numbers
    )
    assert [f.claim_index for f in flags] == [0, 1]


def test_missing_numbers_not_flagged_when_result_has_number():
    pc = paper_claims("p", ["A claim with no digits.", "Another one."], result="28.4 BLEU")
    assert not _kinds(run_checks(pc, parsed("p")), FlagKind.missing_numbers)


def test_missing_numbers_not_flagged_when_abstract_has_only_years_and_model_names():
    pc = paper_claims("p", ["A claim.", "Another."], result="it works")
    abstract = "Since 2017, GPT-4 and Llama-3 changed things (see Table 1)."
    assert not _kinds(run_checks(pc, parsed("p", abstract=abstract)), FlagKind.missing_numbers)


def test_missing_numbers_needs_paper():
    pc = paper_claims("p", ["A claim.", "Another."], result="it works")
    assert not _kinds(run_checks(pc, None), FlagKind.missing_numbers)


# --------------------------------------------------------------------------- too long


def test_too_long():
    long = " ".join(["word"] * 61)
    ok = " ".join(["word"] * 60)
    flags = _kinds(run_checks(paper_claims("p", [long, ok]), None), FlagKind.too_long)
    assert [f.claim_index for f in flags] == [0]
    assert "61 words" in flags[0].detail


# --------------------------------------------------------------------------- similarity


def test_tfidf_cosine_identical():
    s = "Sparse attention reduces memory on long documents"
    assert tfidf_cosine(s, s, build_idf([s, "other text"])) == pytest.approx(1.0)


def test_tfidf_cosine_near_and_unrelated():
    a = "LoRA-X reduces fine-tuning memory by 40% on Llama-2 7B for instruction tuning"
    b = "LoRA-X reduces fine-tuning memory by 40% on Llama-2 7B for instruction following"
    c = "A new benchmark of 500 geometry problems for vision-language models"
    idf = build_idf([a, b, c])
    near = tfidf_cosine(a, b, idf)
    far = tfidf_cosine(a, c, idf)
    assert 0.7 < near < 1.0
    assert far < 0.1
    assert jaccard(a, b) > jaccard(a, c)


def test_near_duplicate_flags_both_with_method_numbers():
    s = "Sparse attention reduces memory on long documents by 40%"
    flags = _kinds(
        run_checks(paper_claims("p", [s, s + ".", "Unrelated 3 claims."]), None),
        FlagKind.near_duplicate,
    )
    assert sorted(f.claim_index for f in flags) == [0, 1]
    assert "TF-IDF cosine" in flags[0].detail and "Jaccard" in flags[0].detail


def test_no_near_duplicate_for_distinct_claims():
    pc = paper_claims(
        "p", ["Sparse attention cuts memory 40%.", "A new 500-problem geometry benchmark."]
    )
    assert not _kinds(run_checks(pc, None), FlagKind.near_duplicate)


def test_empty_statement_cosine_zero():
    assert tfidf_cosine("", "the of", {}) == 0.0


def test_powers_count_as_numbers():
    from reviewer.eval.checks import has_number

    assert has_number("generalizes to sequence lengths up to 2^20 after training on 2^8")
    assert not has_number("uses GPT-4 as a judge")


def test_scaled_hyphenated_counts_are_numbers():
    from reviewer.eval.checks import find_numbers

    assert find_numbers("prompting a 540B-parameter language model") == ["540B-parameter"]
    assert find_numbers("a GPT-3 style model") == []
