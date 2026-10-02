from pathlib import Path

import pytest
import yaml

from reviewer.config import PROJECT_ROOT
from reviewer.meta.leak_filter import filter_meta, is_score_leak, split_sentences
from reviewer.models import (
    MetaJudgment,
    MetaReviewOut,
    ReviewCategory,
    Strength,
    SubScoreItem,
    Weakness,
    WeaknessSeverity,
)

PUBLIC = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "leak_filter_public.yaml").read_text()
)
PRIVATE = PROJECT_ROOT / "data" / "private" / "leak_filter_dev.yaml"


@pytest.mark.parametrize("sentence", PUBLIC["metric"])
def test_metric_sentences_survive_colliding_scores(sentence):
    assert is_score_leak(sentence, scores=PUBLIC["metric_scores"]) is None


@pytest.mark.parametrize("case", PUBLIC["technical"], ids=lambda c: c["text"][:40])
def test_technical_sentences_survive_with_and_without_source(case):
    assert is_score_leak(case["text"]) is None
    assert is_score_leak(case["text"], paper_text=case["source"]) is None


@pytest.mark.parametrize("case", PUBLIC["adversarial"], ids=lambda c: c["text"][:40])
def test_adversarial_cases(case):
    hit = is_score_leak(case["text"], scores=[6, 5, 3])
    assert (hit is not None) == case["leak"], hit


def test_quoted_span_from_the_paper_is_exempt():
    s = "The authors write that the method is likely to be accepted by practitioners."
    assert is_score_leak(s) == "L2-likely"
    assert is_score_leak(s, paper_text="... is likely to be accepted by practitioners ...") is None


def test_sentence_split_keeps_decimals_and_abbreviations():
    text = "FID is 12.4 vs. 13.1, e.g. on CIFAR. Smith et al. disagree. (Table 2) shows it."
    assert split_sentences(text) == [
        "FID is 12.4 vs. 13.1, e.g. on CIFAR.",
        "Smith et al. disagree.",
        "(Table 2) shows it.",
    ]


def _sub() -> SubScoreItem:
    return SubScoreItem(score=3, reason="r")


def _judgment(**kw) -> MetaJudgment:
    base = dict(
        summary="The method is a renamed LoRA. The paper should be accepted after revision.",
        strengths=[
            Strength(
                text="Clear writing.",
                category=ReviewCategory.presentation,
                evidence_ids=["claim:1"],
            )
        ],
        weaknesses=[
            Weakness(
                text="Single seed. My overall score is 5.",
                severity=WeaknessSeverity.major,
                category=ReviewCategory.soundness,
                evidence_ids=["claim:1"],
            ),
            Weakness(
                text="No variance is reported.",
                severity=WeaknessSeverity.minor,
                category=ReviewCategory.soundness,
                evidence_ids=["claim:1"],
            ),
            Weakness(
                text="Overlaps LoRA.",
                severity=WeaknessSeverity.major,
                category=ReviewCategory.novelty,
                evidence_ids=["claim:1"],
            ),
            Weakness(
                text="Captions lack metrics.",
                severity=WeaknessSeverity.minor,
                category=ReviewCategory.presentation,
                evidence_ids=["claim:1"],
            ),
        ],
        questions=["Why no seeds?", "I would rate this a 6 unless more seeds are added."],
        would_change_my_mind="A seed-variance table would move this from a reject to an accept.",
        soundness=_sub(),
        presentation=_sub(),
        contribution=_sub(),
        raw_rating=5,
        confidence=3,
    )
    base.update(kw)
    return MetaJudgment(**base)


def test_task_3_4_examples_through_filter_meta():
    r = filter_meta(_judgment(), paper_text="", evidence_texts=[], scores=[5, 3, 3, 3, 3])
    out = r.out
    assert out.would_change_my_mind is None
    assert out.summary == "The method is a renamed LoRA."
    assert [s.text for s in out.sink] == [
        "Overlaps LoRA.",
        "No variance is reported.",
        "Captions lack metrics.",
    ]
    assert out.questions == ["Why no seeds?"]
    assert "'Most likely to sink this paper'" in out.left_out_note
    assert "'What would change my mind'" in out.left_out_note
    assert {d.pattern for d in r.dropped} >= {"L1-score-phrase", "L2-move-to", "L1-i-would-rate"}
    keep = _judgment(
        would_change_my_mind=(
            "The reported FID score of 12.4 uses only 5k samples. Only 3/5 seeds converge."
        )
    )
    assert filter_meta(keep, "", [], [3, 5]).out.would_change_my_mind.startswith("The reported FID")


def test_allowlist_has_no_numeric_field():
    def numeric(annotation) -> bool:
        return any(t in str(annotation) for t in ("int", "float"))

    assert not [n for n, f in MetaReviewOut.model_fields.items() if numeric(f.annotation)]


@pytest.mark.skipif(not PRIVATE.exists(), reason="private dev fixtures are local only")
def test_private_dev_fixtures():
    cases = yaml.safe_load(PRIVATE.read_text())["cases"]
    wrong = [
        c
        for c in cases
        if (is_score_leak(c["text"], scores=c.get("scores", [])) is not None) != c["leak"]
    ]
    assert not wrong, [(c["source"], c["field"], c["leak"]) for c in wrong]
