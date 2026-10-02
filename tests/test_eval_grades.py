import pytest
import yaml

from reviewer.eval.grades import (
    build_grades_template,
    load_grades,
    load_grades_file,
    render_summary,
    summarize,
    write_grades,
)
from reviewer.eval.loader import build_records
from reviewer.models import ClaimGrade

from .test_eval_helpers import entry, gold, paper_claims, parsed


def _records(statements_a=("A one 1.5.", "A two 2.5."), statements_f=("F one 3x.", "F two 4x.")):
    corpus = [entry("fam", "famous"), entry("ord", "ordinary")]
    claims = {
        "ord": paper_claims("ord", list(statements_a)),
        "fam": paper_claims("fam", list(statements_f)),
    }
    return build_records(
        corpus,
        {"ord": parsed("ord"), "fam": parsed("fam")},
        claims,
        {"fam": gold("fam", ["G"])},
        [],
    )


def test_template_one_per_claim_in_report_order():
    grades = build_grades_template(_records())
    assert [(g.paper_id, g.claim_index) for g in grades] == [
        ("ord", 0),
        ("ord", 1),
        ("fam", 0),
        ("fam", 1),
    ]
    assert all(g.specificity is None and g.matches_gold is None for g in grades)


def test_round_trip_and_header(tmp_path):
    path = tmp_path / "grades.yaml"
    grades = build_grades_template(_records())
    write_grades(path, grades, ["fam"])
    text = path.read_text()
    assert text.startswith("# reports/private/grades.yaml")
    assert "matches_gold" in text and "fam" in text.split("grades:")[0]
    assert load_grades(path) == grades


def _fill(path, **updates):
    data = yaml.safe_load(path.read_text())
    data["grades"][0].update(updates)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_merge_preserves_filled_grades(tmp_path):
    path = tmp_path / "grades.yaml"
    write_grades(path, build_grades_template(_records()))
    _fill(path, specificity=4, correctness=5, notes="good")
    write_grades(path, build_grades_template(_records()))
    g0 = load_grades(path)[0]
    assert (g0.specificity, g0.correctness, g0.notes) == (4, 5, "good")


def test_merge_follows_reordered_claim(tmp_path):
    path = tmp_path / "grades.yaml"
    write_grades(path, build_grades_template(_records()))
    _fill(path, specificity=2)
    write_grades(path, build_grades_template(_records(statements_a=("A two 2.5.", "A one 1.5."))))
    grades = load_grades(path)
    assert grades[1].statement == "A one 1.5." and grades[1].specificity == 2
    assert grades[0].specificity is None


def test_changed_statement_moves_old_grade_to_stale(tmp_path):
    path = tmp_path / "grades.yaml"
    write_grades(path, build_grades_template(_records()))
    _fill(path, specificity=3, notes="old")
    merged = write_grades(
        path, build_grades_template(_records(statements_a=("A new 9.", "A two 2.5.")))
    )
    assert merged.grades[0].statement == "A new 9." and merged.grades[0].specificity is None
    f = load_grades_file(path)
    assert [(s.statement, s.specificity, s.notes) for s in f.stale] == [("A one 1.5.", 3, "old")]
    # stale entries survive later runs too
    write_grades(path, build_grades_template(_records(statements_a=("A new 9.", "A two 2.5."))))
    assert len(load_grades_file(path).stale) == 1


def test_invalid_human_value_raises_and_file_untouched(tmp_path):
    path = tmp_path / "grades.yaml"
    write_grades(path, build_grades_template(_records()))
    _fill(path, specificity=9)
    before = path.read_text()
    with pytest.raises(ValueError):
        write_grades(path, build_grades_template(_records()))
    assert path.read_text() == before


# --------------------------------------------------------------------------- summarize


def _g(pid, tier, ctype, spec=None, corr=None, mg=None, i=0):
    return ClaimGrade(
        paper_id=pid,
        tier=tier,
        claim_index=i,
        contribution_type=ctype,
        statement=f"{pid} {i}",
        specificity=spec,
        correctness=corr,
        matches_gold=mg,
    )


def test_summarize_by_type_and_tier():
    grades = [
        _g("f", "famous", "new_method", 5, 4, mg=0, i=0),
        _g("f", "famous", "empirical_finding", 3, None, mg=-1, i=1),
        _g("o", "ordinary", "new_method", 1, 2),
        _g("w", "own", "new_dataset"),
    ]
    t = summarize(grades, gold_paper_ids=["f"])
    by_type = {r.group: r for r in t.by_type}
    assert by_type["new_method"].mean_specificity == 3.0
    assert by_type["new_method"].mean_correctness == 3.0
    assert by_type["empirical_finding"].mean_correctness is None
    assert by_type["new_dataset"].n_graded == 0
    by_tier = {r.group: r for r in t.by_tier}
    assert [r.group for r in t.by_tier] == ["own", "ordinary", "famous"]
    assert by_tier["famous"].mean_specificity == 4.0
    assert by_tier["own"].mean_specificity is None
    assert (t.n_graded, t.n_total) == (3, 4)
    assert (t.gold_matched, t.gold_graded, t.gold_match_rate) == (1, 2, 0.5)
    text = render_summary(t)
    assert "new_method" in text and "famous" in text and "50%" in text


def test_summarize_all_blank():
    t = summarize([_g("o", "ordinary", "new_method"), _g("o", "ordinary", "new_method", i=1)])
    assert t.n_graded == 0 and t.overall.mean_specificity is None
    assert t.gold_match_rate is None
    assert "Gold match rate: -" in render_summary(t)


def test_summarize_empty():
    t = summarize([])
    assert t.n_total == 0 and t.by_type == [] and t.by_tier == []
    render_summary(t)
