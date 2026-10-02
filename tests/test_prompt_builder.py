import hashlib
import subprocess
from pathlib import Path

import pytest

from reviewer.claims.prompt_builder import (
    DEFAULT_FALLBACK_CHAR_BUDGET,
    DEFAULT_TABLE_CHAR_BUDGET,
    NO_CELLS_NOTE,
    TRUNCATED_NOTE,
    build_context,
    render,
    table_text,
)
from reviewer.claims.prompts import PROMPTS_DIR, load_prompt
from reviewer.config import Settings
from reviewer.models import Defect, Figure, ParsedPaper, Section, Table

SENTINEL = "SENTINEL_SHOULD_NOT_APPEAR"


def _paper(**overrides) -> ParsedPaper:
    sections = [
        Section(heading="Introduction", number="1", paragraphs=["Intro p1.", "Intro p2."]),
        Section(heading="Motivation", number="1.1", level=2, paragraphs=["Intro sub p."]),
        Section(heading="Related Work", number="2", paragraphs=[f"Related {SENTINEL}."]),
        Section(heading="Model Architectures", number="2.1", level=2, paragraphs=["RW model."]),
        Section(heading="Method", number="3", paragraphs=["Method first.", f"Method {SENTINEL}."]),
        Section(heading="Training Setup", number="3.1", level=2, paragraphs=["", "Setup first."]),
        Section(
            heading="Proof Details",
            number="3.2",
            level=2,
            paragraphs=["Proof first.", f"Proof {SENTINEL}."],
        ),
        Section(heading="Experiments", number="4", paragraphs=["Exp first.", f"Exp {SENTINEL}."]),
        Section(heading="Ablation Study", number="4.1", level=2, paragraphs=["Ablation first."]),
        Section(heading="Conclusion", number="5", paragraphs=["Concl p1.", "Concl p2."]),
        Section(heading="Future Directions", number="5.1", level=2, paragraphs=["Future p."]),
        Section(heading="Acknowledgements", paragraphs=[f"Thanks {SENTINEL}."]),
    ]
    fields = {
        "paper_id": "test-1",
        "source_path": "x.pdf",
        "title": "A Test Paper",
        "abstract": "We test things.",
        "sections": sections,
        "figures": [Figure(label="Figure 1", caption="Overview of the model.")],
        "tables": [
            Table(
                label="Table 2",
                caption="Main results.",
                rows=[["Model", "BLEU", ""], ["Ours", " 28.4 ", ""]],
                raw_text=f"raw {SENTINEL}",
            )
        ],
        "contributions": "Our contributions are: (1) a method; (2) a benchmark.",
    }
    fields.update(overrides)
    return ParsedPaper(**fields)


def test_intro_includes_subsections():
    ctx = build_context(_paper())
    assert [s.heading for s in ctx.introduction] == ["1 Introduction", "1.1 Motivation"]
    assert ctx.introduction[0].text == "Intro p1.\n\nIntro p2."


def test_intro_falls_back_to_first_section():
    paper = _paper(sections=[Section(heading="Overview", paragraphs=["o"]), Section(heading="B")])
    ctx = build_context(paper)
    assert [s.heading for s in ctx.introduction] == ["Overview"]


def test_conclusion_with_subsections():
    ctx = build_context(_paper())
    assert [s.heading for s in ctx.conclusion] == ["5 Conclusion", "5.1 Future Directions"]
    assert "Concl p2." in ctx.conclusion[0].text


def test_method_openers_first_paragraph_only_and_related_work_excluded():
    ctx = build_context(_paper())
    openers = {s.heading: s.text for s in ctx.method_experiment_openers}
    assert openers == {
        "3 Method": "Method first.",
        "3.1 Training Setup": "Setup first.",
        "3.2 Proof Details": "Proof first.",  # subsection of a method section counts
        "4 Experiments": "Exp first.",
        "4.1 Ablation Study": "Ablation first.",
    }
    assert "2.1 Model Architectures" not in openers


def test_contributions_headings_and_captions():
    ctx = build_context(_paper())
    assert ctx.contributions is not None and "benchmark" in ctx.contributions
    assert ctx.section_headings[0] == "1 Introduction"
    assert ctx.section_headings[1] == "  1.1 Motivation"
    assert "Acknowledgements" in ctx.section_headings
    assert ctx.figure_captions == ["Figure 1: Overview of the model."]
    assert [(t.header, t.body) for t in ctx.tables] == [
        ("Table 2: Main results.", "Model | BLEU\nOurs | 28.4")
    ]


def test_render_is_tagged_and_excludes_unselected_text():
    text = render(build_context(_paper()))
    for tag in (
        "title",
        "abstract",
        "contributions_statement",
        "introduction",
        "conclusion",
        "section_headings",
        "figure_captions",
        "tables",
        "method_and_experiment_openers",
    ):
        assert f"<{tag}>" in text and f"</{tag}>" in text
    assert SENTINEL not in text  # raw_text is not sent when rows exist
    assert "<table_captions>" not in text
    assert "### Table 2: Main results.\nModel | BLEU\nOurs | 28.4" in text
    assert "Our contributions are" in text


def test_render_without_contributions():
    text = render(build_context(_paper(contributions=None)))
    assert "no contributions statement found" in text


def test_load_prompt_versions():
    assert load_prompt("extract_v1").version == "extract_v1"
    critic = load_prompt("critic_v1.md")
    assert critic.version == "critic_v1" and "pass_one_draft" in critic.text


def test_configured_prompts_exist_and_mention_tables():
    settings = Settings(_env_file=None)
    extract, critic = load_prompt(settings.extract_prompt), load_prompt(settings.critic_prompt)
    assert (extract.version, critic.version) == ("extract_v3", "critic_v4")
    for prompt in (extract, critic):
        assert "<tables>" in prompt.text and "exactly as it appears" in prompt.text
        assert "2 to 4" not in prompt.text and "1 to 4" in prompt.text
    assert "fewer strong claims beat padding" in extract.text
    assert "pass_one_draft" in critic.text and "final_index" in critic.text


# sha256 of the v1 and v2 prompt files as committed; they must never change (reproducible
# diff).
V1_SHA256 = {
    "extract_v1.md": "280d1a6f275f06679b070d982d7dce359cc477f24b7772de0c3443cd8081bf48",
    "critic_v1.md": "b5cef16970264fb3df6743173506b489170dda6d86557583e247c1731a9412eb",
    "extract_v2.md": "1a235d7577358c556b7311107525d9e816dbed2b6a8134645cb76c6dacd667ee",
    "critic_v2.md": "b20a3adbe8e8f47389acfc9244c9c35b07a392d8ac57fc669d8db948ff2a8b5e",
}


@pytest.mark.parametrize("name", sorted(V1_SHA256))
def test_v1_prompts_unchanged(name):
    assert hashlib.sha256((PROMPTS_DIR / name).read_bytes()).hexdigest() == V1_SHA256[name]


def test_v1_prompts_match_git_head():
    repo = Path(__file__).resolve().parents[1]
    for name in V1_SHA256:
        rel = f"src/reviewer/claims/prompts/{name}"
        try:
            head = subprocess.run(
                ["git", "show", f"HEAD:{rel}"], cwd=repo, capture_output=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git history not available")
        assert (repo / rel).read_bytes() == head


# --------------------------------------------------------------------------- tables


def _table(**kw) -> Table:
    return Table(label=kw.pop("label", "Table 1"), caption=kw.pop("caption", "Results."), **kw)


def test_table_rows_collapse_whitespace_and_strip_trailing_empty_cells():
    t = table_text(
        _table(rows=[["Method", "Acc\n(%)", "", " "], ["", "", ""], ["Ours  v2", "91.3", "", "x"]]),
        1000,
    )
    assert t.header == "Table 1: Results."
    assert t.body == "Method | Acc (%)\nOurs v2 | 91.3 |  | x"


def test_table_raw_text_fallback_when_no_rows():
    for rows in (None, [], [["", " "]]):
        t = table_text(_table(rows=rows, raw_text="Method  Acc\nOurs 91.3"), 1000)
        assert t.body == "Method Acc Ours 91.3"


def test_table_caption_only():
    t = table_text(_table(raw_text="   "), 1000)
    assert t.body == NO_CELLS_NOTE
    text = render(build_context(_paper(tables=[_table()])))
    assert f"### Table 1: Results.\n{NO_CELLS_NOTE}" in text


def test_table_rows_truncated_at_row_boundary_within_budget():
    rows = [[f"row{i}", "12.5", "13.75"] for i in range(10)]
    line_len = len("row0 | 12.5 | 13.75")  # 19
    budget = 3 * line_len + 2 + 5  # room for 3 rows (with newlines) but not 4
    body = table_text(_table(rows=rows), budget).body
    lines = body.split("\n")
    assert lines[:3] == ["row0 | 12.5 | 13.75", "row1 | 12.5 | 13.75", "row2 | 12.5 | 13.75"]
    assert lines[3:] == ["[… 7 more rows not shown]"]
    assert len("\n".join(lines[:3])) <= budget


def test_table_rows_fitting_exactly_are_not_truncated():
    rows = [["a", "1"], ["b", "2"]]
    assert table_text(_table(rows=rows), len("a | 1\nb | 2")).body == "a | 1\nb | 2"


def test_table_raw_text_truncated():
    raw = " ".join(f"tok{i}" for i in range(500))
    body = table_text(_table(raw_text=raw), 100).body
    assert body.endswith(" [… truncated]")
    kept = body.removesuffix(" [… truncated]")
    assert len(kept) <= 100 and raw.startswith(kept) and not kept.endswith("tok")


def test_budget_is_passed_through_and_defaults_to_config():
    assert DEFAULT_TABLE_CHAR_BUDGET == Settings(_env_file=None).table_char_budget == 1500
    rows = [[f"r{i}", "1.0"] for i in range(1000)]
    paper = _paper(tables=[_table(rows=rows), _table(label="Table 9", raw_text="x " * 5000)])
    for budget in (DEFAULT_TABLE_CHAR_BUDGET, 200):
        ctx = build_context(paper, table_char_budget=budget)
        for t in ctx.tables:
            content = t.body.rsplit("\n[…", 1)[0].removesuffix(" [… truncated]")
            assert len(content) <= budget
            assert "…" in t.body
    assert build_context(paper) == build_context(paper, table_char_budget=1500)


def test_subsections_of_methods_section_contribute_openers():
    from reviewer.claims.prompt_builder import build_context, render
    from reviewer.models import ParsedPaper, Section

    paper = ParsedPaper(
        paper_id="x",
        source_path="x.pdf",
        title="T",
        sections=[
            Section(heading="Introduction", number="1", paragraphs=["Intro."]),
            Section(heading="Methods", number="2", level=1, paragraphs=[]),
            Section(
                heading="Masking to Reduce Copying",
                number="2.2",
                level=2,
                paragraphs=["We mask RAKE keywords.", "SECOND PARAGRAPH SENTINEL"],
            ),
            Section(heading="Datasets", number="3", paragraphs=["NOT A METHOD SENTINEL"]),
        ],
    )
    text = render(build_context(paper))
    assert "We mask RAKE keywords." in text
    assert "SECOND PARAGRAPH SENTINEL" not in text
    assert "NOT A METHOD SENTINEL" not in text


# --------------------------------------------------------------------------- v3 prompts


def test_v3_prompts_explain_fallback_pages():
    for name in ("extract_v3", "critic_v3"):
        text = load_prompt(name).text
        assert "<fallback_pages>" in text and "under-recovered" in text
        assert "garbled math" in text and "duplicated" in text
        assert "clearly attached to the claim it supports" in text
        assert "<tables>" in text and "1 to 4" in text and "2 to 4" not in text


def test_critic_v3_truism_rule_and_defects():
    text = load_prompt("critic_v3").text
    assert "would any competent practitioner already know this is true" in text
    assert "A truism must be rejected, never rewritten." in text
    assert "purely qualitative" in text and "lowering its confidence" in text
    for defect in Defect:
        assert defect.value in text
    # examples stay invented: nothing from the corpus paper that motivated the rule
    assert "DeepSpeed" not in text and "Nebula" not in text and "GPU" not in text


def test_extract_v3_is_v2_plus_fallback_only():
    v2, v3 = load_prompt("extract_v2").text, load_prompt("extract_v3").text
    fallback = v3[v3.index("## Fallback pages") : v3.index("## Each claim")]
    stripped = v3.replace(fallback, "").replace(
        ", the opening paragraph of each method and experiment section, and, when present, raw "
        "text for pages the parser under-recovered in <fallback_pages>.",
        ", and the opening paragraph of each method and experiment section.",
    )
    assert stripped == v2


# --------------------------------------------------------------------------- fallback pages


def _fallback(page: int, *paragraphs: str) -> Section:
    return Section(
        heading=f"Page {page}", paragraphs=list(paragraphs), fallback=True, fallback_page=page
    )


def test_no_fallback_block_without_fallback_sections():
    text = render(build_context(_paper()))
    assert "<fallback_pages>" not in text and "pdftotext" not in text


def test_fallback_pages_rendered_in_own_block():
    paper = _paper()
    paper = paper.model_copy(
        update={
            "sections": [
                *paper.sections,
                _fallback(4, "Result 91.2 on dev.", "Second para."),
                _fallback(9, "  "),
                _fallback(11, "Conclusion garbled ∑ text."),
            ]
        }
    )
    ctx = build_context(paper)
    assert [s.heading for s in ctx.fallback_pages] == [
        "Page 4 (pdftotext fallback, may be garbled or duplicate other text):",
        "Page 11 (pdftotext fallback, may be garbled or duplicate other text):",
    ]
    text = render(ctx)
    assert text.endswith(
        "<fallback_pages>\n"
        "Page 4 (pdftotext fallback, may be garbled or duplicate other text):\n"
        "Result 91.2 on dev.\n\nSecond para.\n\n"
        "Page 11 (pdftotext fallback, may be garbled or duplicate other text):\n"
        "Conclusion garbled ∑ text.\n"
        "</fallback_pages>"
    )


def test_fallback_sections_never_intro_conclusion_openers_or_headings():
    # A paper whose only GROBID section is odd; fallback headings that would otherwise match.
    fallbacks = [
        Section(heading="Introduction", paragraphs=["FB intro"], fallback=True, fallback_page=1),
        Section(heading="Method", paragraphs=["FB method"], fallback=True, fallback_page=2),
        Section(heading="Conclusion", paragraphs=["FB concl"], fallback=True, fallback_page=3),
    ]
    paper = _paper()
    paper = paper.model_copy(update={"sections": [*paper.sections, *fallbacks]})
    ctx = build_context(paper)
    assert ctx == build_context(_paper()).model_copy(update={"fallback_pages": ctx.fallback_pages})
    assert len(ctx.fallback_pages) == 3
    # nothing but fallback sections: intro falls back to nothing, not to a fallback page
    only = build_context(_paper(sections=fallbacks))
    assert only.introduction == only.conclusion == only.method_experiment_openers == []
    assert only.section_headings == []
    text = render(only)
    before, block = text.split("<fallback_pages>")
    assert "FB intro" not in before and "FB method" not in before and "FB concl" not in before
    assert all(s in block for s in ("FB intro", "FB method", "FB concl"))


def test_fallback_per_page_cap():
    assert DEFAULT_FALLBACK_CHAR_BUDGET == 4000
    long = " ".join(f"w{i}" for i in range(3000))
    paper = _paper(sections=[_fallback(2, long), _fallback(3, "short page")])
    ctx = build_context(paper)
    body = ctx.fallback_pages[0].text
    assert body.endswith(f" {TRUNCATED_NOTE}")
    kept = body.removesuffix(f" {TRUNCATED_NOTE}")
    assert len(kept) <= DEFAULT_FALLBACK_CHAR_BUDGET and long.startswith(kept)
    assert ctx.fallback_pages[1].text == "short page"
    small = build_context(paper, fallback_char_budget=50).fallback_pages[0].text
    assert len(small.removesuffix(f" {TRUNCATED_NOTE}")) <= 50 and TRUNCATED_NOTE in small
