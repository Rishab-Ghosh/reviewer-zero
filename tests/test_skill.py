import re
from pathlib import Path

import anyio
from mcp.client import Client

from reviewer.mcp.server import build_server
from reviewer.meta.leak_filter import is_score_leak

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "skills" / "reviewer-zero" / "SKILL.md").read_text()
METHODS = (ROOT / "src" / "reviewer" / "methods" / "prompts" / "methods_v2.md").read_text()


def test_frontmatter():
    head = SKILL.split("---")[1]
    assert re.search(r"^name: reviewer-zero$", head, re.M)
    assert re.search(r"^description: .{40,}", head, re.M)


def test_every_server_tool_is_in_the_skill():
    async def names():
        async with Client(build_server()) as c:
            return {t.name for t in (await c.list_tools()).tools}

    tools = anyio.run(names)
    assert tools >= {
        "check_format",
        "check_citations",
        "find_prior_work",
        "paper",
        "citations",
        "verify_quotes",
        "review_paper",
    }
    missing = [t for t in tools if f"`{t}" not in SKILL]
    assert not missing, missing


def test_methodology_checklist_matches_the_pipeline_prompt():
    prompt_items = re.findall(r"^\s*\d+\. ([a-z_]+):", METHODS, re.M)
    skill_items = re.findall(r"^\s*\d+\. \*\*([a-z_]+)\*\*", SKILL, re.M)
    assert len(prompt_items) == 10 and skill_items == prompt_items


def test_forbidden_phrases_in_the_skill_are_ones_the_leak_filter_catches():
    for phrase in (
        "This would be a weak accept.",
        "The paper is publishable.",
        "I would raise my rating.",
        "It is likely to be accepted.",
    ):
        assert is_score_leak(phrase) is not None, phrase
        key = re.sub(r"[.]$", "", phrase).split()[-2:]
        assert " ".join(key).lower().strip(".") in SKILL.lower() or key[-1].lower() in SKILL.lower()
