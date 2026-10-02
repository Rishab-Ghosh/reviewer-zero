from reviewer.privacy import content_violations, private_path_violations, private_snippets


def test_private_paths_blocked():
    paths = [
        "data/private/gold/x.yaml",
        "data/papers/own/MyPaper.pdf",
        "data/parsed/own-mypaper-12345678.json",
        "data/claims/own-mypaper-12345678.pass1.json",
        "reports/private/week1_claims.md",
        "data/failures.jsonl",
        "data/parsed/1706.03762.json",
        "reports/week1_claims.md",
    ]
    assert private_path_violations(paths) == paths[:6]


def test_content_checks_own_ids_and_private_text(tmp_path):
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "p.yaml").write_text(
        "claims:\n  - statement: This private sentence is long enough to be checked by the guard.\n"
    )
    snippets = private_snippets(tmp_path)
    assert snippets
    leak = "Intro. This private sentence is long enough to be checked by the guard. More."
    assert content_violations("reports/week1_claims.md", leak, snippets)
    assert content_violations("STATUS.md", "see own-mypaper-1a2b3c4d", [])
    assert not content_violations("tests/test_x.py", "own-mypaper-1a2b3c4d", [])
    assert not content_violations("reports/week1_claims.md", "clean text", snippets)


def test_quotations_of_public_papers_are_not_private(tmp_path):
    from reviewer.privacy import drop_public, public_corpus_text

    (tmp_path / "1234.56789.json").write_text(
        '{"abstract": "Our approach cuts tokens by 48.6% on MATH500 with QwQ."}'
    )
    (tmp_path / "own-x-12345678.json").write_text(
        '{"abstract": "Secret own text that is long enough to count."}'
    )
    public = public_corpus_text(tmp_path)
    snippets = [
        "Our approach cuts tokens by 48.6% on MATH500 with QwQ.",
        "Secret own text that is long enough to count.",
    ]
    assert drop_public(snippets, public) == ["Secret own text that is long enough to count."]


def test_short_public_name_lists_are_not_snippets(tmp_path):
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "p.yaml").write_text(
        "claims:\n  - setting: 'Benchmarks: MATH-500, AIME 2025, HumanEval, and GPQA.'\n"
    )
    assert private_snippets(tmp_path) == []


def test_any_claims_version_dir_own_outputs_are_private():
    from reviewer.privacy import private_path_violations

    assert private_path_violations(["data/claims_v4/own-mypaper-12345678.json"]) == [
        "data/claims_v4/own-mypaper-12345678.json"
    ]
    assert private_path_violations(["data/claims_v4/2501.00001.json"]) == []
