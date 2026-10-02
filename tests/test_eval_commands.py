"""End-to-end over a temporary data dir: loaders, run_eval, run_summarize."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from reviewer import config
from reviewer.cli import app
from reviewer.eval.loader import load_claims, load_corpus, load_gold

from .test_eval_helpers import failure, gold, paper_claims, parsed


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for sub in ("parsed", "claims", "gold"):
        (data / sub).mkdir(parents=True)
    (data / "corpus.yaml").write_text(
        yaml.safe_dump(
            {
                "papers": [
                    {
                        "paper_id": "1706.03762",
                        "tier": "famous",
                        "reason": "classic",
                        "title": "Attn",
                    },
                    {
                        "paper_id": "2501.00001",
                        "tier": "ordinary",
                        "reason": "workshop",
                        "title": "Ord",
                    },
                    {
                        "paper_id": "2501.00002",
                        "tier": "ordinary",
                        "reason": "avg",
                        "title": "Broken",
                    },
                ]
            }
        )
    )
    for pid in ("1706.03762", "2501.00001", "own-mine-abcd1234"):
        (data / "parsed" / f"{pid}.json").write_text(parsed(pid).model_dump_json())
        (data / "claims" / f"{pid}.json").write_text(
            paper_claims(pid, ["Claim one 28.4 BLEU.", "Claim two 3x faster."]).model_dump_json()
        )
    (data / "claims" / "1706.03762.pass1.json").write_text("{not even json")
    (data / "gold" / "1706.03762.yaml").write_text(
        "paper_id: 1706.03762\nsource: paper_contributions\nclaims:\n  - statement: Gold claim.\n"
    )
    (data / "gold" / "9999.99999.yaml").write_text(
        yaml.safe_dump(gold("9999.99999", ["x"]).model_dump(mode="json"))
    )
    (data / "gold" / "_example.yaml.sample").write_text("junk: [")
    (data / "failures.jsonl").write_text(
        failure("2501.00002", "parse", "GROBID 500").model_dump_json() + "\nnot json\n"
    )
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    config.get_settings.cache_clear()
    yield data
    config.get_settings.cache_clear()


def test_loaders(data_dir, caplog):
    assert [e.paper_id for e in load_corpus(data_dir / "corpus.yaml")] == [
        "1706.03762",
        "2501.00001",
        "2501.00002",
    ]
    claims, errors = load_claims(data_dir / "claims")
    assert set(claims) == {"1706.03762", "2501.00001", "own-mine-abcd1234"} and not errors
    g = load_gold(data_dir / "gold", {"1706.03762", "2501.00001"})
    assert set(g) == {"1706.03762"}
    assert g["1706.03762"].paper_id == "1706.03762"  # unquoted id read as float, fixed up
    assert "9999.99999.yaml does not match" in caplog.text


def test_run_eval_and_summarize(data_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 0, result.output
    md = (tmp_path / "reports" / "private" / "week1_claims.md").read_text()
    order = [
        md.index(f"`{pid}`")
        for pid in ("own-mine-abcd1234", "2501.00001", "2501.00002", "1706.03762")
    ]
    assert order == sorted(order)
    assert "parse failed: GROBID 500" in md
    assert md.count("### Gold vs extracted") == 1

    public = (tmp_path / "reports" / "week1_claims.md").read_text()
    assert "own-mine" not in public
    assert "Public version." in public
    assert public.count("### Gold vs extracted — source: paper_contributions") == 1
    assert not (tmp_path / "reports" / "grades.yaml").exists()

    grades_path = tmp_path / "reports" / "private" / "grades.yaml"
    result = runner.invoke(app, ["eval", "summarize"])
    assert result.exit_code == 0
    assert "no grades yet: 0/6 claims graded across 3 papers" in result.output

    data = yaml.safe_load(grades_path.read_text())
    assert len(data["grades"]) == 6
    famous = next(g for g in data["grades"] if g["paper_id"] == "1706.03762")
    famous.update(specificity=4, correctness=5, matches_gold=0)
    grades_path.write_text(yaml.safe_dump(data, sort_keys=False))

    assert runner.invoke(app, ["eval"]).exit_code == 0  # re-run keeps the grade
    kept = yaml.safe_load(grades_path.read_text())["grades"]
    assert any(g["specificity"] == 4 for g in kept)

    result = runner.invoke(app, ["eval", "summarize"])
    assert result.exit_code == 0
    assert "Graded 1/6 claims" in result.output
    assert "Gold match rate: 100%" in result.output


def test_summarize_without_file(data_dir):
    result = CliRunner().invoke(app, ["eval", "summarize"])
    assert result.exit_code == 0
    assert "no grades yet" in result.output


def test_unreadable_parsed_file_reported(data_dir, tmp_path):
    (data_dir / "parsed" / "2501.00002.json").write_text(json.dumps({"paper_id": "x"}))
    assert CliRunner().invoke(app, ["eval"]).exit_code == 0
    md = (tmp_path / "reports" / "private" / "week1_claims.md").read_text()
    assert "parsed/2501.00002.json could not be loaded" in md


def _write_gold(directory, g):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{g.paper_id}.yaml").write_text(yaml.safe_dump(g.model_dump(mode="json")))


def test_load_all_gold_private_flags_and_precedence(data_dir, caplog):
    from reviewer.eval.loader import load_all_gold

    private_dir = data_dir / "private" / "gold"
    _write_gold(private_dir, gold("1706.03762", ["Private wins."], source="paper_contributions"))
    _write_gold(private_dir, gold("own-mine-abcd1234", ["Mine."]))
    _write_gold(data_dir / "gold", gold("2501.00001", ["Lab text."], source="lab_summary"))
    (private_dir / "_x.yaml.sample").write_text("junk: [")
    known = {"1706.03762", "2501.00001", "own-mine-abcd1234"}
    g, private_ids = load_all_gold(data_dir / "gold", private_dir, known)
    assert set(g) == known
    assert g["1706.03762"].claims[0].statement == "Private wins."
    assert private_ids == known  # private dir, private dir, lab_summary source
    assert "exists in both" in caplog.text
    assert "9999.99999.yaml does not match" in caplog.text


def test_run_eval_private_gold_only_in_private_report(data_dir, tmp_path):
    _write_gold(
        data_dir / "private" / "gold",
        gold("2501.00001", ["Secret lab gold 61.2%."], source="lab_summary", results=["61.2%"]),
    )
    assert CliRunner().invoke(app, ["eval"]).exit_code == 0
    private = (tmp_path / "reports" / "private" / "week1_claims.md").read_text()
    public = (tmp_path / "reports" / "week1_claims.md").read_text()
    assert "Secret lab gold" in private and "lab_summary (private) | 0/1 (0%)" in private
    assert "Secret lab gold" not in public and "61.2" not in public
    assert "own-mine" not in public
    grades = yaml.safe_load((tmp_path / "reports" / "private" / "grades.yaml").read_text())
    assert {g["paper_id"] for g in grades["grades"]} >= {"own-mine-abcd1234", "2501.00001"}


def test_legacy_grades_file_ignored(data_dir, tmp_path, caplog):
    legacy = tmp_path / "reports" / "grades.yaml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("grades: not-a-list-at-all\n")
    assert CliRunner().invoke(app, ["eval"]).exit_code == 0
    assert "ignoring old" in caplog.text
    assert legacy.read_text() == "grades: not-a-list-at-all\n"


def test_parse_quality_cli(data_dir):
    result = CliRunner().invoke(app, ["eval", "parse-quality"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("paper_id")
    assert any(line.startswith("own-mine-abcd1234") and " own " in line for line in lines)
    assert any(line.startswith("1706.03762") and "famous" in line for line in lines)
