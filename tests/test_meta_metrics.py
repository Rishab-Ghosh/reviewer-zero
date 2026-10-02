import math

from reviewer.meta.metrics import by_group_distribution, rankdata, score_histogram, spearman


def test_rankdata_no_ties():
    assert rankdata([3, 1, 2]) == [3.0, 1.0, 2.0]


def test_rankdata_ties_get_average_rank():
    assert rankdata([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_positive():
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)


def test_spearman_perfect_negative():
    assert math.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_spearman_nan_when_no_variance():
    assert math.isnan(spearman([1, 1, 1], [2, 3, 4]))


def test_spearman_nan_for_single_point():
    assert math.isnan(spearman([1], [2]))


def test_spearman_rejects_length_mismatch():
    import pytest

    with pytest.raises(ValueError, match="length mismatch"):
        spearman([1, 2], [1, 2, 3])


def test_by_group_distribution_splits_and_summarizes():
    out = by_group_distribution([1, 2, 3, 10], ["a", "a", "a", "b"])
    assert out["a"]["n"] == 3
    assert out["a"]["mean"] == 2.0
    assert out["b"]["n"] == 1
    assert out["b"]["mean"] == 10.0


def test_score_histogram_counts_per_group():
    out = score_histogram([1, 1, 2, 1], ["accepted", "accepted", "accepted", "rejected"])
    assert out["accepted"] == {1: 2, 2: 1}
    assert out["rejected"] == {1: 1}
