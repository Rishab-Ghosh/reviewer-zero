"""Per-step `max_tokens` from observed output sizes (BP §4, task 1.8).

Each step's cap is 2x the largest output of any of its active-prompt calls in the dev LLM cache.
The observed maxima are committed (`observed_output_tokens.json`: counts only, no content) so
the caps are reproducible without the cache; `reviewer review-caps` refreshes them.
"""

import json
import re
from pathlib import Path

from reviewer.config import Settings
from reviewer.models import ReviewStep

OBSERVED_FILE = Path(__file__).with_name("observed_output_tokens.json")
CAP_FACTOR = 2
_TAG_SPLIT = re.compile(r"[+#|:]")


def step_prompts(s: Settings) -> dict[ReviewStep, set[str]]:
    """Prompt versions each web step calls with the current settings."""
    return {
        ReviewStep.claims: {s.extract_prompt, s.critic_prompt}
        | ({s.verify_prompt} if s.verify_prompt else set()),
        ReviewStep.citations: {s.citation_prompt},
        ReviewStep.novelty: {
            s.novelty_prompt,
            s.novelty_severity_prompt,
            s.novelty_query_prompt,
            s.novelty_listwise_prompt,
        },
        ReviewStep.methods: {s.methods_prompt},
        ReviewStep.writing: {s.writing_prompt},
        ReviewStep.meta: {s.meta_prompt},
    }


def tag_step(cache_tag: str, s: Settings) -> ReviewStep | None:
    """The step an LLM cache entry belongs to, by its leading prompt version; None for calls
    no web step makes (judge, baselines, older prompt versions)."""
    head = _TAG_SPLIT.split(cache_tag, maxsplit=1)[0]
    for step, prompts in step_prompts(s).items():
        if head in prompts:
            return step
    return None


def observed_max(llm_cache_dir: Path, s: Settings) -> dict[ReviewStep, tuple[int, int]]:
    """{step: (max output tokens, n responses)} over the cache's active-prompt entries."""
    out: dict[ReviewStep, tuple[int, int]] = {}
    for f in llm_cache_dir.glob("*.json"):
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
            step = tag_step(entry["cache_tag"], s)
            tokens = int(entry["usage"]["output_tokens"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if step is None:
            continue
        top, n = out.get(step, (0, 0))
        out[step] = (max(top, tokens), n + 1)
    return out


def write_observed(observed: dict[ReviewStep, tuple[int, int]], path: Path = OBSERVED_FILE) -> None:
    body = {
        step.value: {"max_output_tokens": top, "n_responses": n}
        for step, (top, n) in sorted(observed.items())
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_caps(path: Path = OBSERVED_FILE) -> dict[ReviewStep, int]:
    """Step caps = 2x observed max. Steps without LLM calls get 0 (no call is allowed)."""
    body = json.loads(path.read_text(encoding="utf-8"))
    caps = {step: 0 for step in ReviewStep}
    for key, row in body.items():
        caps[ReviewStep(key)] = CAP_FACTOR * int(row["max_output_tokens"])
    return caps
