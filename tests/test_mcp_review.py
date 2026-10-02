import json
from pathlib import Path

import anyio
import pytest
from mcp.client import Client

import reviewer.mcp.format_tool as ft
from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.prior_work import QueryError
from reviewer.mcp.review_tool import NO_GROBID, NO_KEY, run_review_paper
from reviewer.mcp.server import build_server
from reviewer.models import ReviewStep
from reviewer.parse.grobid_client import GrobidClient
from reviewer.review import orchestrator
from reviewer.review.estimate import NOVELTY_LITE_MEDIAN_USD, STEP_MEDIAN_USD

DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo" / "spindle.pdf"
FORBIDDEN_KEYS = {
    "rating",
    "raw_rating",
    "score",
    "soundness",
    "presentation",
    "contribution",
    "confidence",
    "verdict_paper",
    "judgments",
    "calibration",
    "p_accept",
}

OUTPUTS = {
    ReviewStep.claims: {"claims": [{"statement": "SPINDLE matches full fine-tuning."}]},
    ReviewStep.novelty: {
        "claims": [
            {
                "query": {"statement": "SPINDLE matches full fine-tuning."},
                "verdict": "already_done",
                "severity": {"score": 3, "drivers": ["2106.09685"], "reason": "LoRA is the same."},
                "ranked": [
                    {
                        "candidate": {
                            "paper": {"key": "2106.09685", "title": "LoRA", "year": 2021}
                        },
                        "label": "same",
                        "reason": "Identical update.",
                    }
                ],
            }
        ]
    },
    ReviewStep.methods: {
        "findings": [
            {
                "item": "seeds_variance",
                "verdict": "no",
                "agreement": "confirmed",
                "concern": "Single seed.",
                "quotes": [{"quote": "single run with seed 0", "verified": True}],
            },
            {"item": "data_disclosure", "verdict": "yes", "quotes": []},
        ]
    },
    ReviewStep.writing: {
        "findings": [
            {
                "check": "abstract_overclaim",
                "quote": "every task",
                "location": "abstract",
                "detail": "MNLI is lower.",
            }
        ]
    },
    ReviewStep.citations: {
        "references": [
            {
                "raw_string": "Vaswani 2015.",
                "flags": [{"ref_id": "b0", "type": "wrong_year", "detail": "cites 2015"}],
            }
        ],
        "support": [
            {
                "ref_id": "b0",
                "sentence": "S.",
                "verdict": "contradicted",
                "quote": "q",
                "rationale": "r",
            }
        ],
    },
}


def _meta_output() -> dict:
    from reviewer.meta.calibration import calibrate
    from reviewer.models import MetaReview, TokenUsage
    from tests.test_meta import _judgment

    j = _judgment(rating=6, evidence_ids=["claim:1"]).model_copy(
        update={
            "summary": "SPINDLE re-derives LoRA. This paper would be accepted at a workshop.",
            "would_change_my_mind": "Two things would raise my rating: multi-seed results.",
        }
    )
    review = MetaReview(
        paper_id="p",
        title="T",
        venue="ICLR",
        venue_year=2027,
        model="m",
        prompt_version="v",
        generated="2026-10-02T00:00:00Z",
        judgments=[j],
        calibration=calibrate([j]),
        checker_status=[],
        token_usage=TokenUsage(),
    )
    return json.loads(review.model_dump_json())


OUTPUTS[ReviewStep.meta] = _meta_output()


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GROBID_URL", "http://localhost:8070")
    monkeypatch.setattr(GrobidClient, "is_alive", lambda self: True)
    monkeypatch.setattr(ft, "grobid_paper", lambda pdf, config: None)
    monkeypatch.setattr(HostedIndex, "paper", lambda self, key: None)
    return McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path)


def _real_path(step: ReviewStep, s, pid: str) -> Path:
    """Where the real checker writes, i.e. where the plan looks for finished steps."""
    return {
        ReviewStep.parse: s.parsed_dir,
        ReviewStep.claims: s.claims_dir,
        ReviewStep.novelty: s.novelty_dir,
        ReviewStep.methods: s.methods_dir,
        ReviewStep.meta: s.meta_dir,
        ReviewStep.format: s.private_reports_dir / "format",
        ReviewStep.citations: s.private_reports_dir / "citations",
        ReviewStep.writing: s.private_reports_dir / "writing",
    }[step] / f"{pid}.json"


def _fake_steps(monkeypatch, tmp_path) -> list[ReviewStep]:
    ran: list[ReviewStep] = []
    for step in ReviewStep:

        def fn(ctx, s, step=step):
            ran.append(step)
            body = OUTPUTS.get(step, {})
            out = _real_path(step, s, ctx.paper_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(body))
            return out

        monkeypatch.setitem(orchestrator.STEPS, step, fn)
    return ran


def test_no_key_names_the_free_tools(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(QueryError) as e:
        run_review_paper(str(DEMO), "iclr2027", config=McpConfig(cache=tmp_path))
    assert str(e.value) == NO_KEY and "check_format" in NO_KEY


def test_no_grobid_and_remote_grobid(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(GrobidClient, "is_alive", lambda self: False)
    with pytest.raises(QueryError, match="docker run"):
        run_review_paper(str(DEMO), "iclr2027", config=McpConfig(cache=tmp_path))
    monkeypatch.setenv("GROBID_URL", "https://grobid.example.com")
    with pytest.raises(QueryError, match="must point at this machine"):
        run_review_paper(str(DEMO), "iclr2027", config=McpConfig(cache=tmp_path))
    assert "docker run" in NO_GROBID


def test_plan_spends_nothing(monkeypatch, env):
    def boom(*a, **k):
        raise AssertionError("network or model touched in the plan phase")

    monkeypatch.setattr("reviewer.llm.client.anthropic_transport", boom)
    monkeypatch.setattr(HostedIndex, "paper", boom)
    r = run_review_paper(str(DEMO), "iclr2027", confirm=False, config=env)
    assert r.phase == "plan" and r.spent_usd is None and r.findings is None
    assert r.novelty_mode == "full"
    assert r.estimate_usd == pytest.approx(round(sum(STEP_MEDIAN_USD.values()), 2))
    assert 0 < r.estimate_usd < r.limit_usd <= 6
    assert r.steps_already_done == [] and len(r.steps_to_run) == 8
    assert r.format.rules == "ICLR 2027"


def test_confirmed_review_runs_every_step_and_never_exposes_a_score(monkeypatch, tmp_path, env):
    ran = _fake_steps(monkeypatch, tmp_path)
    seen = []
    r = run_review_paper(str(DEMO), "iclr2027", confirm=True, config=env, on_step=seen.append)
    assert r.phase == "review" and set(ran) == set(ReviewStep) and len(seen) == 8
    assert all(s.status == "done" for s in r.steps) and r.spent_usd == 0
    f = r.findings
    assert f["claims"][0]["statement"].startswith("SPINDLE")
    pw = f["prior_work"][0]
    assert pw["possible_prior_claim"]["work"] == "2106.09685"
    assert pw["works"][0]["label"] == "check first"
    assert [m["item"] for m in f["methodology"]] == ["seeds_variance"]
    assert f["methodology"][0]["quotes"] == ["single run with seed 0"]
    assert f["citations"]["flags"][0]["type"] == "wrong_year"
    assert f["citations"]["possible_contradictions"][0]["ref_id"] == "b0"
    blob = r.model_dump_json()
    assert r.meta_review is not None
    assert r.meta_review.summary == "SPINDLE re-derives LoRA."
    assert r.meta_review.would_change_my_mind == "multi-seed results."
    assert "'What would change my mind'" in r.meta_review.left_out_note
    assert "would be accepted" not in blob and "raise my rating" not in blob
    assert '"already_done"' not in blob  # the novelty verdict value (D14)

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    assert not FORBIDDEN_KEYS & set(keys(json.loads(blob)))


def test_second_call_resumes_from_what_is_done(monkeypatch, tmp_path, env):
    _fake_steps(monkeypatch, tmp_path)
    run_review_paper(str(DEMO), "iclr2027", confirm=True, config=env)
    again = run_review_paper(str(DEMO), "iclr2027", confirm=False, config=env)
    assert len(again.steps_already_done) == 8 and again.steps_to_run == []
    assert again.estimate_usd == 0


def test_progress_notifications_through_an_mcp_client(monkeypatch, tmp_path, env):
    _fake_steps(monkeypatch, tmp_path)
    monkeypatch.setenv("REVIEWER_ZERO_INDEX_URL", "https://index.test")
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path / "c"))
    events: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress, total, message):
        events.append((progress, total, message))

    async def go():
        async with Client(build_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool(
                "review_paper",
                {"pdf_path": str(DEMO), "venue": "iclr2027", "confirm": True},
                progress_callback=on_progress,
            )
            return names, res

    names, res = anyio.run(go)
    assert "review_paper" in names and not res.is_error
    assert len(events) == 8 and events[-1][0] == 8 and events[0][1] == 8


def test_lite_mode_estimate_and_pool_sizes(monkeypatch, tmp_path, env):
    lite = run_review_paper(str(DEMO), "iclr2027", config=env, novelty="lite")
    rest = sum(STEP_MEDIAN_USD.values()) - STEP_MEDIAN_USD[ReviewStep.novelty]
    assert lite.novelty_mode == "lite"
    assert lite.estimate_usd == pytest.approx(round(rest + NOVELTY_LITE_MEDIAN_USD, 2))
    pools = {}
    _fake_steps(monkeypatch, tmp_path)
    orig = orchestrator.STEPS[ReviewStep.novelty]

    def spy(ctx, s):
        pools[ctx.settings.novelty_pool_size] = s.novelty_pool_size
        return orig(ctx, s)

    monkeypatch.setitem(orchestrator.STEPS, ReviewStep.novelty, spy)
    run_review_paper(str(DEMO), "iclr2027", confirm=True, config=env, novelty="lite")
    run_review_paper(str(DEMO), "iclr2027", confirm=True, config=env, novelty="full")
    assert pools == {60: 60, 150: 150}


def test_switching_mode_reruns_novelty_and_meta(monkeypatch, tmp_path, env):
    _fake_steps(monkeypatch, tmp_path)
    run_review_paper(str(DEMO), "iclr2027", confirm=True, config=env, novelty="lite")
    same = run_review_paper(str(DEMO), "iclr2027", config=env, novelty="lite")
    assert same.steps_to_run == [] and same.estimate_usd == 0
    other = run_review_paper(str(DEMO), "iclr2027", config=env, novelty="full")
    assert other.steps_to_run == ["novelty", "meta"]
    assert other.estimate_usd == pytest.approx(
        round(STEP_MEDIAN_USD[ReviewStep.novelty] + STEP_MEDIAN_USD[ReviewStep.meta], 2)
    )


def test_no_key_message_reaches_the_client(monkeypatch, tmp_path):
    """Through the server (task group + progress pump), the ToolError text must reach the
    client, not the SDK's generic 'Error executing tool'."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("REVIEWER_ZERO_CACHE", str(tmp_path))

    async def go():
        async with Client(build_server()) as c:
            return await c.call_tool("review_paper", {"pdf_path": str(DEMO), "venue": "iclr2027"})

    res = anyio.run(go)
    assert res.is_error and "ANTHROPIC_API_KEY" in res.content[0].text
