import json
import time
from pathlib import Path

import pytest

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import ReviewStep, StepStatus, TokenUsage
from reviewer.review import orchestrator
from reviewer.review.budget import Budget
from reviewer.review.orchestrator import VersionContext, run_step
from reviewer.review.profile import WEB_OVERRIDES, paper_settings, web_settings
from tests.test_meta import _judgment, _paper

BASELINE_KEYS = {"baseline_abstract", "baseline_full", "contamination_probe"}


class CountingTransport:
    def __init__(self, parsed, output_tokens: int = 50) -> None:
        self.parsed = parsed
        self.output_tokens = output_tokens
        self.requests: list[TransportRequest] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.requests.append(req)
        return TransportResponse(
            parsed=self.parsed,
            usage=TokenUsage(input_tokens=1000, output_tokens=self.output_tokens),
            stop_reason="end_turn",
        )


@pytest.fixture
def base(tmp_path) -> Settings:
    # The dev defaults (meta_run_baselines=True) are what the web profile must override.
    return Settings(_env_file=None, data_dir=tmp_path / "dev", anthropic_api_key="test-key")


@pytest.fixture
def ctx(tmp_path, base) -> VersionContext:
    s = paper_settings(base, tmp_path / "work", tmp_path / "public", tmp_path / "papercache")
    paper = _paper()
    s.parsed_dir.mkdir(parents=True)
    (s.parsed_dir / f"{paper.paper_id}.json").write_text(paper.model_dump_json())
    return VersionContext(paper_id=paper.paper_id, settings=s, venue="ICLR", venue_year=2025)


def _fake_transport(monkeypatch, fake) -> None:
    monkeypatch.setattr("reviewer.llm.client.anthropic_transport", lambda *a, **k: fake)


def test_web_profile_pins_d67_even_when_dev_defaults_differ(base):
    assert base.meta_run_baselines is True
    s = web_settings(base)
    assert (s.meta_run_baselines, s.meta_k, s.meta_blind_decision_status) == (False, 1, False)
    assert s.parse_docling_tables is False and s.llm_retry_truncated is True
    assert set(WEB_OVERRIDES) >= {"meta_run_baselines", "meta_k", "meta_blind_decision_status"}


def test_d67_meta_step_makes_exactly_one_call_and_stores_no_baselines(monkeypatch, ctx):
    fake = CountingTransport(_judgment(rating=6, evidence_ids=["claims:none"]))
    _fake_transport(monkeypatch, fake)
    result = run_step(ReviewStep.meta, ctx, Budget(limit_usd=5.0))
    assert result.status == StepStatus.done, result
    assert len(fake.requests) == ctx.settings.meta_k == 1
    assert result.n_calls == 1 and result.cost_usd > 0
    stored = json.loads(Path(result.output).read_text())
    assert all(stored.get(k) is None for k in BASELINE_KEYS)
    assert fake.requests[0].max_tokens == orchestrator.STEP_MAX_TOKENS[ReviewStep.meta]


def test_meta_step_never_calls_run_meta_review(monkeypatch, ctx):
    import reviewer.meta.commands as meta_commands

    def boom(*a, **k):
        raise AssertionError("the web meta step must not use run_meta_review")

    monkeypatch.setattr(meta_commands, "run_meta_review", boom)
    _fake_transport(
        monkeypatch, CountingTransport(_judgment(rating=6, evidence_ids=["claims:none"]))
    )
    assert run_step(ReviewStep.meta, ctx, Budget(limit_usd=5.0)).status == StepStatus.done


def test_llm_cache_is_rooted_per_paper(ctx, tmp_path):
    client = LLMClient(ctx.settings)
    assert client.cache.dir == tmp_path / "papercache" / "llm"
    assert ctx.settings.cache_dir == tmp_path / "public"


def test_budget_refusal_fails_the_step_without_a_call(monkeypatch, ctx):
    fake = CountingTransport(_judgment(rating=6, evidence_ids=["claims:none"]))
    _fake_transport(monkeypatch, fake)
    result = run_step(ReviewStep.meta, ctx, Budget(limit_usd=0.01))
    assert result.status == StepStatus.failed and result.error_kind == "budget"
    assert fake.requests == [] and "meta" in ctx.failures


def test_deadline_passed_times_out_the_step_without_a_call(monkeypatch, ctx):
    fake = CountingTransport(_judgment(rating=6, evidence_ids=["claims:none"]))
    _fake_transport(monkeypatch, fake)
    result = run_step(ReviewStep.meta, ctx, Budget(limit_usd=5.0), deadline=time.monotonic() - 1)
    assert result.status == StepStatus.timed_out and fake.requests == []


def test_truncated_call_retries_once_at_double_max_tokens(monkeypatch, ctx):
    good = _judgment(rating=6, evidence_ids=["claims:none"])

    class TruncateOnce(CountingTransport):
        def __call__(self, req):
            self.requests.append(req)
            stop = "max_tokens" if len(self.requests) == 1 else "end_turn"
            return TransportResponse(
                parsed=None if stop == "max_tokens" else good,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                stop_reason=stop,
            )

    fake = TruncateOnce(good)
    _fake_transport(monkeypatch, fake)
    result = run_step(ReviewStep.meta, ctx, Budget(limit_usd=5.0))
    assert result.status == StepStatus.done
    cap = orchestrator.STEP_MAX_TOKENS[ReviewStep.meta]
    assert [r.max_tokens for r in fake.requests] == [cap, 2 * cap]


def test_cache_hit_bypasses_the_budget(monkeypatch, ctx):
    fake = CountingTransport(_judgment(rating=6, evidence_ids=["claims:none"]))
    _fake_transport(monkeypatch, fake)
    assert run_step(ReviewStep.meta, ctx, Budget(limit_usd=5.0)).n_calls == 1
    again = run_step(ReviewStep.meta, ctx, Budget(limit_usd=0.0))
    assert again.status == StepStatus.done and again.n_calls == 0 and again.cost_usd == 0
    assert len(fake.requests) == 1


def test_error_kind_never_carries_exception_text(monkeypatch, ctx):
    def leaky(*a, **k):
        raise ValueError("a sentence from the paper")

    monkeypatch.setitem(orchestrator.STEPS, ReviewStep.writing, leaky)
    result = run_step(ReviewStep.writing, ctx, Budget(limit_usd=1.0))
    assert result.error_kind == "ValueError"


def test_api_error_without_usage_is_charged_the_hold(monkeypatch, ctx):
    from reviewer.llm.client import LLMAPIError

    def failing(req):
        raise LLMAPIError("connection reset")

    _fake_transport(monkeypatch, failing)
    budget = Budget(limit_usd=5.0)
    result = run_step(ReviewStep.meta, ctx, budget)
    assert result.status == StepStatus.failed
    assert result.n_calls >= 1 and result.cost_usd > 0
    assert budget.spent_usd == pytest.approx(result.cost_usd)


def test_missing_api_key_is_charged_nothing(ctx):
    s = ctx.settings.model_copy(update={"anthropic_api_key": None})
    ctx.settings = s
    budget = Budget(limit_usd=5.0)
    result = run_step(ReviewStep.meta, ctx, budget)
    assert result.status == StepStatus.failed and result.cost_usd == 0 and budget.spent_usd == 0


def test_paper_settings_keep_the_committed_novelty_model(ctx, base):
    from reviewer.config import PACKAGED_NOVELTY_MODELS

    assert ctx.settings.first_stage_model_file == PACKAGED_NOVELTY_MODELS / "first_stage_lr.json"
    assert ctx.settings.first_stage_model_file.exists()
    assert ctx.settings.claims_dir != base.claims_dir


def test_error_kind_names_the_cause_type(monkeypatch, ctx):
    def wrapped(*a, **k):
        try:
            raise KeyError("text from the paper")
        except KeyError as e:
            raise RuntimeError("more text") from e

    monkeypatch.setitem(orchestrator.STEPS, ReviewStep.writing, wrapped)
    assert run_step(ReviewStep.writing, ctx, Budget(limit_usd=1.0)).error_kind == (
        "RuntimeError(KeyError)"
    )
