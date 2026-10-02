from pathlib import Path

import pytest

from reviewer.config import Settings
from reviewer.models import ReviewStep, StepStatus
from reviewer.review import orchestrator
from reviewer.review.budget import Budget
from reviewer.review.orchestrator import VersionContext, run_review
from reviewer.review.profile import paper_settings


@pytest.fixture
def ctx(tmp_path) -> VersionContext:
    base = Settings(_env_file=None, data_dir=tmp_path / "dev", anthropic_api_key=None)
    s = paper_settings(base, tmp_path / "work", tmp_path / "pub", tmp_path / "pc")
    return VersionContext(paper_id="p1", settings=s, pdf=tmp_path / "p1.pdf")


def _fake_steps(monkeypatch, tmp_path: Path, fail: set[ReviewStep]) -> list[ReviewStep]:
    ran: list[ReviewStep] = []
    for step in ReviewStep:

        def fn(ctx, s, step=step):
            ran.append(step)
            if step in fail:
                return None  # checker produced no output -> failed
            out = tmp_path / f"{step.value}.json"
            out.write_text("{}")
            return out

        monkeypatch.setitem(orchestrator.STEPS, step, fn)
    return ran


def _status(results) -> dict[ReviewStep, StepStatus]:
    return {r.step: r.status for r in results}


def test_full_dag_runs_every_step_meta_last(monkeypatch, tmp_path, ctx):
    ran = _fake_steps(monkeypatch, tmp_path, fail=set())
    results = run_review(ctx, Budget(limit_usd=1.0))
    assert all(st == StepStatus.done for st in _status(results).values())
    assert ran[0] == ReviewStep.parse and ran[-1] == ReviewStep.meta
    assert ran.index(ReviewStep.novelty) > ran.index(ReviewStep.claims)
    assert len(results) == len(ReviewStep)


def test_parse_failure_stops_the_job(monkeypatch, tmp_path, ctx):
    ran = _fake_steps(monkeypatch, tmp_path, fail={ReviewStep.parse})
    results = run_review(ctx, Budget(limit_usd=1.0))
    assert ran == [ReviewStep.parse] and len(results) == 1
    assert results[0].status == StepStatus.failed


def test_claims_failure_skips_novelty_and_meta(monkeypatch, tmp_path, ctx):
    ran = _fake_steps(monkeypatch, tmp_path, fail={ReviewStep.claims})
    st = _status(run_review(ctx, Budget(limit_usd=1.0)))
    assert st[ReviewStep.novelty] == StepStatus.skipped
    assert st[ReviewStep.meta] == StepStatus.skipped
    assert ReviewStep.novelty not in ran and ReviewStep.meta not in ran


def test_meta_needs_three_other_checkers(monkeypatch, tmp_path, ctx):
    fail = {ReviewStep.citations, ReviewStep.methods, ReviewStep.writing}
    _fake_steps(monkeypatch, tmp_path, fail=fail)
    st = _status(run_review(ctx, Budget(limit_usd=1.0)))
    assert st[ReviewStep.meta] == StepStatus.skipped  # only novelty + format ok (2 of 5)

    fail = {ReviewStep.citations, ReviewStep.methods}
    _fake_steps(monkeypatch, tmp_path, fail=fail)
    st = _status(run_review(ctx, Budget(limit_usd=1.0)))
    assert st[ReviewStep.meta] == StepStatus.done  # 3 of 5
