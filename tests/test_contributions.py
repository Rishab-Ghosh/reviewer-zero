"""Contributions-statement detection on synthetic section lists."""

from reviewer.models import Section
from reviewer.parse.contributions import find_contributions, is_bullet_like

LONG = (
    "Large language models have transformed natural language processing over the last few "
    "years, and a large body of work studies how to adapt them efficiently to new domains, "
    "tasks, and languages while keeping compute and memory requirements manageable for "
    "practitioners who do not have access to large clusters of accelerators."
)


def _intro(*paragraphs: str, heading: str = "Introduction") -> Section:
    return Section(heading=heading, number="1", level=1, paragraphs=list(paragraphs))


def test_bulleted_list_after_trigger() -> None:
    sections = [
        _intro(
            LONG,
            "Our main contributions are as follows:",
            "• We propose FooNet, a sparse adapter for long-context retrieval.",
            "• We show FooNet improves recall@10 by 4.2 points on BEIR.",
            "• We release code and data.",
            LONG,
        ),
        Section(heading="Related Work", number="2", paragraphs=[LONG]),
    ]
    found = find_contributions(sections)
    assert found is not None
    lines = found.split("\n")
    assert lines[0] == "Our main contributions are as follows:"
    assert len(lines) == 4
    assert LONG not in found


def test_unmarked_short_items_after_colon() -> None:
    # GROBID often drops bullet glyphs; short paragraphs after "...:" still count as items.
    sections = [
        _intro(
            LONG,
            "We make the following contributions:",
            "A formal definition of stepwise calibration error for agents.",
            "An estimator with finite-sample guarantees.",
            LONG,
        )
    ]
    found = find_contributions(sections)
    assert found is not None
    assert found.count("\n") == 2
    assert "finite-sample" in found


def test_inline_threefold() -> None:
    para = (
        "Our contributions are threefold: (1) we introduce BarBench, a benchmark of 3,000 "
        "multi-hop questions; (2) we evaluate 12 open models; (3) we find retrieval helps "
        "only below 7B parameters."
    )
    found = find_contributions([_intro(LONG, para, LONG)])
    assert found == para


def test_ordinal_items_and_numbered_items() -> None:
    sections = [
        _intro(
            "In summary, this paper makes two contributions.",
            "First, we derive a closed-form bound on the error of low-rank adapters.",
            "(ii) We validate the bound on GLUE with RoBERTa-base.",
            LONG,
        )
    ]
    found = find_contributions(sections)
    assert found is not None
    assert found.count("\n") == 2


def test_negative_no_trigger() -> None:
    sections = [
        _intro(LONG, "In this work we propose the Transformer, a new architecture.", LONG),
        Section(heading="Related Work", number="2", paragraphs=["Our contributions differ."]),
    ]
    # the only trigger is after related work, which is out of scope
    assert find_contributions(sections) is None


def test_equal_contribution_footnote_is_not_a_trigger() -> None:
    assert find_contributions([_intro("* Equal contribution. Listing order is random.")]) is None


def test_intro_subsections_searched() -> None:
    sections = [
        _intro(LONG),
        Section(
            heading="Contributions",
            number="1.1",
            level=2,
            paragraphs=["This work contributes a new dataset of 10k annotated proofs."],
        ),
        Section(heading="Method", number="2", paragraphs=["Our contributions are not here."]),
    ]
    assert find_contributions(sections) == (
        "This work contributes a new dataset of 10k annotated proofs."
    )


def test_intro_found_by_heading_not_position() -> None:
    sections = [
        Section(heading="Preface", paragraphs=[LONG]),
        _intro("To summarize, we contribute X and Y."),
    ]
    assert find_contributions(sections) == "To summarize, we contribute X and Y."


def test_fallback_before_related_work() -> None:
    sections = [
        Section(heading="Motivation", paragraphs=[LONG]),  # no 'Introduction' heading
        Section(heading="Overview", paragraphs=["Our key contributions include Z."]),
        Section(heading="Related Work", paragraphs=["Our contributions again."]),
    ]
    assert find_contributions(sections) == "Our key contributions include Z."


def test_empty() -> None:
    assert find_contributions([]) is None


def test_is_bullet_like() -> None:
    assert is_bullet_like("- item")
    assert is_bullet_like("1. item")
    assert is_bullet_like("(iii) item")
    assert is_bullet_like("We introduce a thing.")
    assert not is_bullet_like(LONG)
    assert not is_bullet_like("We " + "word " * 130)


def test_trigger_paragraph_trimmed_to_trigger_sentence():
    from reviewer.models import Section
    from reviewer.parse.contributions import find_contributions

    intro = Section(
        heading="Introduction",
        paragraphs=[
            "We study X. Prior methods fail at Y. The contributions of our paper are as follows:",
            "• We propose Z, a method for Y.",
            "• We show Z beats W by 3 points.",
            "A long normal paragraph " * 20,
        ],
    )
    found = find_contributions([intro])
    assert found is not None
    assert found.startswith("The contributions of our paper are as follows:")
    assert "We study X" not in found
    assert "beats W by 3 points" in found


def test_contributions_paragraph_heading_split_into_own_section():
    from reviewer.models import Section
    from reviewer.parse.contributions import find_contributions

    sections = [
        Section(heading="Introduction", number="1", paragraphs=["We study X. It matters."]),
        Section(heading="Contributions.", paragraphs=["• We propose Z.", "• We release D."]),
        Section(heading="Methods", number="2", paragraphs=["Z works like so."]),
    ]
    assert find_contributions(sections) == "• We propose Z.\n• We release D."


def test_contributions_heading_deep_in_paper_is_ignored():
    from reviewer.models import Section
    from reviewer.parse.contributions import find_contributions

    sections = [Section(heading=f"S{i}", paragraphs=["text"]) for i in range(8)]
    sections.append(Section(heading="Contributions", paragraphs=["Author contributions: A did B."]))
    assert find_contributions(sections) is None
