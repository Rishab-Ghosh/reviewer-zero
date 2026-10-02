from reviewer.gold.author import parse_blocks, resolve_as_above
from reviewer.gold.famous import abstract_result_sentences, contribution_items
from reviewer.gold.pdf_text import join_lines, joins_mid_word, span


def test_parse_blocks_and_as_above():
    lines = [
        "Title",
        "Claim 1",
        "Claim: First claim with",
        "a wrapped line.",
        "Setting: Ten datasets.",
        "Result: 3 of 4 cells.",
        "Claim 2",
        "Claim: Second.",
        "Setting: As above; 6 cells.",
        "What's new: The thing.",
    ]
    blocks = parse_blocks(lines)
    assert blocks[0]["statement"] == "First claim with a wrapped line."
    assert blocks[1]["novelty_assertion"] == "The thing."
    resolved = resolve_as_above(blocks)
    assert resolved[1][0]["setting"] == "Ten datasets.; 6 cells."
    assert "resolved" in resolved[1][1][0]


def test_unresolvable_as_above_left_verbatim():
    blocks = [
        {"statement": "a", "setting": "As above; 30 cells."},
        {"statement": "b", "setting": "As above; 6."},
    ]
    out = resolve_as_above(blocks)
    assert out[0][0]["setting"] == "As above; 30 cells." and "UNRESOLVED" in out[0][1][0]
    assert out[1][0]["setting"] == "As above; 6." and "UNRESOLVED" in out[1][1][0]


def test_join_lines_and_span():
    assert join_lines(["chain-of-", "thought works", "well."]) == "chain-of-thought works well."
    text = "alpha beta. Gamma delta epsilon. Zeta."
    assert span(text, "Gamma", "epsilon.") == "Gamma delta epsilon."


def test_mid_word_join_rule():
    assert joins_mid_word("transiti", "on", prev_reaches_edge=False)
    assert not joins_mid_word("Depends", "on", prev_reaches_edge=False)
    assert joins_mid_word("empiric", "al", prev_reaches_edge=True)
    assert not joins_mid_word("Threshold", "is", prev_reaches_edge=True)
    assert joins_mid_word("Trial-answ", "er", prev_reaches_edge=True)


def test_famous_helpers():
    abstract = (
        "We propose X. Following Kaplan et al. (2020) we train it. "
        "X reaches 28.4 BLEU on WMT 2014. It is simple."
    )
    assert abstract_result_sentences(abstract) == ["X reaches 28.4 BLEU on WMT 2014."]
    items = contribution_items("The contributions are as follows:\n• We do A.\n• We do B.")
    assert items == ["• We do A.", "• We do B."]


def test_setting_overrides_replace_as_above_verbatim():
    from reviewer.gold.author import apply_setting_overrides

    blocks = [
        {"statement": "a", "setting": "As above; 30 cells."},
        {"statement": "b", "setting": "S2"},
    ]
    resolved = resolve_as_above(blocks)
    out = apply_setting_overrides(
        resolved, {1: "Four LLMs; 30 cells; $214."}, "supplied in session"
    )
    assert out[0][0]["setting"] == "Four LLMs; 30 cells; $214."
    assert not any("UNRESOLVED" in n for n in out[0][1])
    assert "supplied in session" in out[0][1][-1] and "As above; 30 cells." in out[0][1][-1]
    assert out[1][0]["setting"] == "S2"
