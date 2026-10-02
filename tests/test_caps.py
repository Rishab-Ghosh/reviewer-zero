import json

import pytest

from reviewer.config import PROJECT_ROOT, Settings
from reviewer.models import ReviewStep
from reviewer.review.caps import OBSERVED_FILE, load_caps, observed_max, tag_step, write_observed
from reviewer.review.orchestrator import STEP_MAX_TOKENS

DEV_LLM_CACHE = PROJECT_ROOT / "data" / "cache" / "llm"


def test_tags_map_to_steps_by_active_prompt():
    s = Settings(_env_file=None)
    assert tag_step("extract_v3+critic_v4", s) == ReviewStep.claims
    assert tag_step("methods_v2#rep1", s) == ReviewStep.methods
    assert tag_step("methods_v2+repair_v1#rep1", s) == ReviewStep.methods
    assert tag_step("support_v5+verify", s) == ReviewStep.citations
    assert tag_step("meta_v1:ICLR", s) == ReviewStep.meta
    assert tag_step("rerank_v3", s) == ReviewStep.novelty
    assert tag_step("rerank_v2", s) is None  # an older prompt no web step runs
    assert tag_step("baseline_full_v1", s) is None  # D67: never a web call


def test_caps_are_twice_the_committed_observed_max():
    body = json.loads(OBSERVED_FILE.read_text())
    caps = load_caps()
    for step in (
        ReviewStep.claims,
        ReviewStep.citations,
        ReviewStep.novelty,
        ReviewStep.methods,
        ReviewStep.writing,
        ReviewStep.meta,
    ):
        assert caps[step] == 2 * body[step.value]["max_output_tokens"] > 0
    assert caps[ReviewStep.parse] == caps[ReviewStep.format] == 0
    assert caps == STEP_MAX_TOKENS


def test_observed_max_reads_counts_only(tmp_path):
    cache = tmp_path / "llm"
    cache.mkdir()
    for i, (tag, out) in enumerate(
        [("meta_v1:ICLR", 300), ("meta_v1:ICLR", 900), ("judge_v2", 5000)]
    ):
        (cache / f"{i}.json").write_text(
            json.dumps({"cache_tag": tag, "usage": {"output_tokens": out}, "parsed": {}})
        )
    (cache / "bad.json").write_text("{")
    obs = observed_max(cache, Settings(_env_file=None))
    assert obs == {ReviewStep.meta: (900, 2)}
    write_observed(obs, tmp_path / "o.json")
    assert load_caps(tmp_path / "o.json")[ReviewStep.meta] == 1800


@pytest.mark.skipif(not DEV_LLM_CACHE.is_dir(), reason="dev LLM cache not present")
def test_no_cached_response_exceeds_its_step_cap():
    obs = observed_max(DEV_LLM_CACHE, Settings(_env_file=None))
    over = {step: top for step, (top, _) in obs.items() if top > STEP_MAX_TOKENS[step]}
    assert over == {}
