import math

from reviewer.meta.dev_report import (
    auc_block,
    build_report,
    contamination_split,
    correlation_block,
    floor_flag,
    is_accepted,
)
from reviewer.models import DevRow


def _row(
    paper_id: str,
    status: str,
    mean_rating: float,
    raw: float,
    baseline_a: int | None = None,
    baseline_b: int | None = None,
    contam_conf: int | None = None,
    contam_guess: bool | None = None,
) -> DevRow:
    return DevRow(
        paper_id=paper_id,
        venue="ICLR",
        year=2024,
        final_status=status,
        mean_reviewer_rating=mean_rating,
        meta_raw_rating=raw,
        baseline_abstract_rating=baseline_a,
        baseline_full_rating=baseline_b,
        contamination_confidence=contam_conf,
        contamination_accepted_guess=contam_guess,
    )


def test_is_accepted():
    assert is_accepted(_row("a", "accepted", 6, 6))
    assert not is_accepted(_row("a", "rejected", 3, 3))
    assert not is_accepted(_row("a", "withdrawn", 3, 3))


def test_correlation_block_tracks_perfectly_correlated_signal():
    rows = [_row(f"p{i}", "accepted", i, i, baseline_a=i, baseline_b=i) for i in range(1, 6)]
    corr = correlation_block(rows)
    assert math.isclose(corr["meta_raw"], 1.0)
    assert math.isclose(corr["baseline_abstract"], 1.0)
    assert math.isclose(corr["baseline_full"], 1.0)


def test_correlation_block_skips_missing_baselines():
    rows = [_row("a", "accepted", 5, 5), _row("b", "rejected", 3, 3)]
    corr = correlation_block(rows)
    assert math.isnan(corr["baseline_abstract"])


def test_auc_block_perfect_separation():
    rows = [
        _row("a", "accepted", 8, 9, baseline_a=8, baseline_b=8),
        _row("b", "accepted", 7, 8, baseline_a=7, baseline_b=7),
        _row("c", "rejected", 3, 2, baseline_a=3, baseline_b=3),
        _row("d", "withdrawn", 2, 1, baseline_a=2, baseline_b=2),
    ]
    auc = auc_block(rows)
    assert auc["meta_raw"] == 1.0
    assert auc["baseline_abstract"] == 1.0


def test_auc_block_treats_withdrawn_as_negative():
    # accepted paper scores lower than a withdrawn paper -> AUC should be hurt (not 1.0)
    rows = [
        _row("a", "accepted", 6, 3),
        _row("b", "withdrawn", 4, 8),
    ]
    assert auc_block(rows)["meta_raw"] == 0.0


def test_floor_flag_counts_accepted_papers_at_the_floor():
    rows = [
        _row("a", "accepted", 8, 1.0),
        _row("b", "accepted", 7, 3.0),
        _row("c", "rejected", 3, 1.0),
    ]
    flag = floor_flag(rows)
    assert flag["n_accepted"] == 2
    assert flag["n_accepted_at_floor"] == 1
    assert flag["accepted_at_floor_paper_ids"] == ["a"]


def test_contamination_split_known_vs_unknown():
    rows = [
        _row("a", "accepted", 8, 8, contam_conf=5, contam_guess=True),
        _row("b", "rejected", 3, 3, contam_conf=1, contam_guess=False),
        _row("c", "rejected", 3, 3, contam_conf=1, contam_guess=True),
    ]
    split = contamination_split(rows)
    assert split["known"]["n"] == 1
    assert split["unknown"]["n"] == 2
    assert split["n_guessed"] == 3
    assert math.isclose(split["probe_guess_accuracy"], 2 / 3)


def test_contamination_split_nan_accuracy_when_nobody_guessed():
    rows = [_row("a", "accepted", 8, 8)]
    split = contamination_split(rows)
    assert math.isnan(split["probe_guess_accuracy"])


def test_build_report_shape():
    rows = [
        _row("a", "accepted", 8, 7, baseline_a=6, baseline_b=6, contam_conf=1),
        _row("b", "rejected", 3, 2, baseline_a=4, baseline_b=3, contam_conf=1),
        _row("c", "withdrawn", 4, 1, baseline_a=3, baseline_b=2, contam_conf=1),
    ]
    report = build_report(rows)
    assert report["n_papers"] == 3
    assert report["n_by_group"] == {"accepted": 1, "rejected": 1, "withdrawn": 1}
    assert "correlation" in report and "auc" in report
    assert "mean_reviewer_rating" in report["rating_distribution_by_group"]
    assert "accepted" in report["meta_raw_rating_histogram_by_group"]
    assert report["accepted_floor_flag"]["n_accepted"] == 1


def test_human_agreement_perfect_when_reviewers_agree():
    from reviewer.meta.dev_report import human_agreement

    rows = [
        _row(f"p{i}", "accepted" if i >= 3 else "rejected", i, i).model_copy(
            update={"reviewer_ratings": [i, i, i]}
        )
        for i in range(1, 6)
    ]
    out = human_agreement(rows, draws=5)
    assert out["n"] == 5
    assert math.isclose(out["spearman_one_vs_rest"], 1.0)
    assert out["auc_one"] == 1.0


def test_human_agreement_skips_single_review_papers():
    from reviewer.meta.dev_report import human_agreement

    rows = [_row("a", "accepted", 5, 5).model_copy(update={"reviewer_ratings": [5]})]
    assert human_agreement(rows)["n"] == 0
