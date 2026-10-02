import json

import pytest

from reviewer.meta.devset import eligible, load_decisions, mean_rating, stratified_sample
from reviewer.models import IclrDecisionRow


def _row(sub_id: str, venue: str, status: str, ratings: list[int]) -> IclrDecisionRow:
    return IclrDecisionRow(
        submission_id=sub_id,
        venue=venue,
        year=2024,
        title=f"Paper {sub_id}",
        final_status=status,
        n_reviews=len(ratings),
        ratings=ratings,
    )


def test_load_decisions_tolerates_null_decision_and_extra_fields(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "submission_id": "a1",
                "venue": "ICLR.cc/2024/Conference",
                "year": 2024,
                "title": "T",
                "final_status": "withdrawn",
                "decision": None,
                "number": 7,
                "tier": None,
                "ratings": [5, 6],
            }
        )
        + "\n"
    )
    rows = load_decisions(path)
    assert len(rows) == 1
    assert rows[0].decision is None
    assert rows[0].ratings == [5, 6]


def test_mean_rating_none_when_no_ratings():
    r = _row("a", "V", "withdrawn", [])
    assert mean_rating(r) is None


def test_mean_rating_averages():
    r = _row("a", "V", "accepted", [4, 6])
    assert mean_rating(r) == 5.0


def test_eligible_filters_venue_status_and_empty_ratings():
    rows = [
        _row("a", "ICLR.cc/2024/Conference", "accepted", [6]),
        _row("b", "ICLR.cc/2024/Conference", "desk_rejected", [6]),
        _row("c", "ICLR.cc/2023/Conference", "accepted", [6]),
        _row("d", "ICLR.cc/2024/Conference", "withdrawn", []),
    ]
    out = eligible(rows, ("ICLR.cc/2024/Conference",))
    assert [r.submission_id for r in out] == ["a"]


def test_stratified_sample_is_deterministic_and_respects_per_cell_plus_buffer():
    rows = [_row(f"a{i}", "ICLR.cc/2024/Conference", "accepted", [5]) for i in range(20)] + [
        _row(f"r{i}", "ICLR.cc/2024/Conference", "rejected", [3]) for i in range(20)
    ]
    cells = [("ICLR.cc/2024/Conference", "accepted"), ("ICLR.cc/2024/Conference", "rejected")]
    s1 = stratified_sample(rows, cells, per_cell=3, seed_tag="tag", buffer=2)
    s2 = stratified_sample(rows, cells, per_cell=3, seed_tag="tag", buffer=2)
    assert [r.submission_id for r in s1[cells[0]]] == [r.submission_id for r in s2[cells[0]]]
    assert len(s1[cells[0]]) == 5  # per_cell + buffer
    assert len(s1[cells[1]]) == 5


def test_stratified_sample_different_seed_gives_different_order():
    rows = [_row(f"a{i}", "V", "accepted", [5]) for i in range(20)]
    cells = [("V", "accepted")]
    s1 = stratified_sample(rows, cells, per_cell=5, seed_tag="tag1", buffer=0)
    s2 = stratified_sample(rows, cells, per_cell=5, seed_tag="tag2", buffer=0)
    ids1 = [r.submission_id for r in s1[cells[0]]]
    ids2 = [r.submission_id for r in s2[cells[0]]]
    assert ids1 != ids2


def test_stratified_sample_warns_when_cell_short(caplog):
    rows = [_row("a1", "V", "accepted", [5])]
    cells = [("V", "accepted")]
    with caplog.at_level("WARNING"):
        out = stratified_sample(rows, cells, per_cell=5, seed_tag="tag", buffer=5)
    assert len(out[cells[0]]) == 1
    assert "only 1 eligible" in caplog.text


@pytest.mark.parametrize("status", ["accepted", "rejected", "withdrawn"])
def test_eligible_keeps_all_three_target_statuses(status):
    rows = [_row("a", "V", status, [5])]
    assert len(eligible(rows, ("V",))) == 1


def test_interleave_cells_round_robins_and_keeps_cell_order():
    from reviewer.meta.dev_commands import interleave_cells

    rows = [_row(f"a{i}", "V", "accepted", [5]) for i in range(3)] + [
        _row(f"r{i}", "V", "rejected", [3]) for i in range(2)
    ]
    out = [r.submission_id for r in interleave_cells(rows)]
    assert out == ["a0", "r0", "a1", "r1", "a2"]


def test_record_failure_accepts_meta_stage(tmp_path):
    from reviewer.failures import record_failure

    path = tmp_path / "failures.jsonl"
    record_failure(path, "meta", "or-x", "LLMRefusalError: refused")
    assert '"meta"' in path.read_text()
