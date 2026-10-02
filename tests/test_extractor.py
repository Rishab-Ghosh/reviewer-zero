import json

import pytest
import typer

from reviewer.claims.commands import run_extract
from reviewer.claims.extractor import (
    ExtractionError,
    critic_consistency_problems,
    draft_problems,
    extract_claims,
    reconcile_kept,
)
from reviewer.claims.prompts import load_prompt
from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    Claim,
    ClaimsDraft,
    ClaimVerdict,
    CriticNote,
    CriticOutput,
    Defect,
    PaperClaims,
    ParsedPaper,
    Section,
    Table,
    TokenUsage,
)

K, RW, RJ, AD = (
    ClaimVerdict.kept,
    ClaimVerdict.rewritten,
    ClaimVerdict.rejected,
    ClaimVerdict.added,
)


def _claim(tag: str) -> Claim:
    return Claim(
        statement=f"{tag} statement",
        contribution_type="new_method",
        method="m",
        problem="p",
        setting="s",
        result="r",
        novelty_assertion="n",
        evidence=["Table 1"],
        confidence=0.8,
    )


def _draft(tag: str, n_claims: int = 3, n_keywords: int = 6) -> ClaimsDraft:
    return ClaimsDraft(
        one_line_summary=f"{tag} summary",
        claims=[_claim(f"{tag}{i}") for i in range(n_claims)],
        keywords=[f"kw{i}" for i in range(n_keywords)],
    )


def _note(
    d: int | None,
    f: int | None,
    verdict: ClaimVerdict,
    defects: list[Defect] | None = None,
) -> CriticNote:
    """A review note; rewritten and rejected notes default to one (vague) defect."""
    if defects is None:
        defects = [Defect.vague] if verdict in (RW, RJ) else []
    return CriticNote(
        draft_index=d, final_index=f, verdict=verdict, defects=defects, reason="because"
    )


def _critic(final: ClaimsDraft, n_draft: int = 3) -> CriticOutput:
    """Consistent notes for a `final` whose claims all differ from an `n_draft`-claim draft:
    draft claim i is rewritten into final claim i; surplus draft claims are rejected."""
    n_final = len(final.claims)
    review = [
        _note(i, i if i < n_final else None, RW if i < n_final else RJ) for i in range(n_draft)
    ]
    review += [_note(None, j, AD) for j in range(n_draft, n_final)]
    return CriticOutput(review=review, final=final)


class ScriptedTransport:
    """Returns queued outputs per schema; records every request."""

    def __init__(self, drafts: list[ClaimsDraft], critics: list[CriticOutput]) -> None:
        self.queues = {"ClaimsDraft": list(drafts), "CriticOutput": list(critics)}
        self.calls: list[TransportRequest] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls.append(req)
        parsed = self.queues[req.schema.__name__].pop(0)
        return TransportResponse(
            parsed=parsed,
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
            stop_reason="end_turn",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        critic_prompt="critic_v3",
        number_grounding=False,
        data_dir=tmp_path,
        anthropic_api_key=None,
        llm_max_retries=2,
    )


@pytest.fixture
def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p1",
        source_path="p1.pdf",
        title="Paper One",
        abstract="Abstract.",
        sections=[Section(heading="Introduction", number="1", paragraphs=["Intro."])],
    )


def test_happy_path_final_is_critic_version(settings, paper):
    pass1, final = _draft("draft"), _draft("final", n_claims=2)
    fake = ScriptedTransport([pass1], [_critic(final)])
    claims, draft = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)

    assert draft == pass1
    assert [c.statement for c in claims.claims] == [c.statement for c in final.claims]
    assert claims.one_line_summary == "final summary"
    meta = claims.extraction_metadata
    assert meta.prompt_version == "extract_v3+critic_v3"
    assert meta.retries == 0
    assert meta.critic_review[0].verdict == ClaimVerdict.rewritten
    assert set(meta.pass_usage) == {"extract", "critic"}
    assert meta.token_usage.input_tokens == 2000
    assert meta.token_usage.cost_usd == pytest.approx(
        meta.pass_usage["extract"].cost_usd + meta.pass_usage["critic"].cost_usd
    )
    # critic sees the context plus the pass-one draft
    critic_req = fake.calls[1]
    assert "<pass_one_draft>" in critic_req.user and "draft0 statement" in critic_req.user
    assert "<title>" in critic_req.user


def test_retry_on_too_many_claims(settings, paper):
    fake = ScriptedTransport(
        [_draft("bad", n_claims=5), _draft("good")], [_critic(_draft("final"))]
    )
    claims, draft = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert draft.one_line_summary == "good summary"
    assert claims.extraction_metadata.retries == 1
    assert "expected 1 to 4 claims, got 5" in fake.calls[1].user
    assert claims.extraction_metadata.pass_usage["extract"].input_tokens == 2000


def test_hard_failure_after_retries(settings, paper):
    bad = [_draft("bad", n_claims=5) for _ in range(3)]
    fake = ScriptedTransport(bad, [])
    with pytest.raises(ExtractionError, match="still invalid after 2 retries"):
        extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(fake.calls) == 3


def test_critic_failure_falls_back_to_pass1_claims_marked_unreviewed(settings, paper):
    draft = _draft("draft")
    critics = [_critic(_draft("f", n_keywords=11)) for _ in range(3)]
    fake = ScriptedTransport([draft], critics)
    claims, pass1 = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert pass1 == draft
    assert claims.claims == draft.claims and claims.keywords == draft.keywords
    meta = claims.extraction_metadata
    assert (
        meta.critic_fallback and "critic pass still invalid after 2 retries" in meta.critic_fallback
    )
    assert meta.critic_review == []
    assert meta.pass_usage["critic"].output_tokens > 0  # failed critic attempts still counted


def test_run_extract_writes_files_and_records_failures(settings, paper):
    settings.parsed_dir.mkdir(parents=True)
    (settings.parsed_dir / "p1.json").write_text(paper.model_dump_json())
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    client = LLMClient(settings, fake)

    run_extract(ids=["p1", "missing"], settings=settings, client=client)

    out = settings.claims_dir
    assert PaperClaims.model_validate_json((out / "p1.json").read_text()).claims
    assert ClaimsDraft.model_validate_json((out / "p1.pass1.json").read_text())
    failures = [json.loads(x) for x in settings.failures_file.read_text().splitlines()]
    assert [(f["stage"], f["paper_id"]) for f in failures] == [("extract", "missing")]

    # second run skips the existing output without calling the transport
    run_extract(ids=["p1"], settings=settings, client=client)
    assert len(fake.calls) == 2


def test_rerun_is_fully_cached(settings, paper):
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    first, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    again, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(fake.calls) == 2
    usage = again.extraction_metadata.token_usage
    assert usage.from_cache and not first.extraction_metadata.token_usage.from_cache
    assert usage.cost_usd == pytest.approx(first.extraction_metadata.token_usage.cost_usd)


def test_run_extract_exits_without_api_key(settings, paper):
    settings.parsed_dir.mkdir(parents=True)
    (settings.parsed_dir / "p1.json").write_text(paper.model_dump_json())
    with pytest.raises(typer.Exit):
        run_extract(settings=settings)
    assert not settings.failures_file.exists()


# --------------------------------------------------------------------------- claim counts


def test_draft_problems_counts():
    assert draft_problems(_draft("a", n_claims=1)) == []
    assert draft_problems(_draft("a", n_claims=4, n_keywords=10)) == []
    assert draft_problems(_draft("a", n_claims=0)) == ["expected 1 to 4 claims, got 0"]
    assert draft_problems(_draft("a", n_claims=5)) == ["expected 1 to 4 claims, got 5"]
    assert draft_problems(_draft("a", n_keywords=4)) == ["expected 5 to 10 keywords, got 4"]
    assert draft_problems(_draft("a", n_keywords=11)) == ["expected 5 to 10 keywords, got 11"]


def test_one_claim_draft_accepted_end_to_end(settings, paper):
    one = _draft("draft", n_claims=1)
    fake = ScriptedTransport([one], [CriticOutput(review=[_note(0, 0, K)], final=one)])
    claims, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(claims.claims) == 1 and claims.extraction_metadata.retries == 0


# --------------------------------------------------------------------------- critic notes


def _final_from(draft: ClaimsDraft, picks: list[int | str]) -> ClaimsDraft:
    """Final set: an int keeps that draft claim verbatim, a str makes a new claim."""
    claims = [draft.claims[p] if isinstance(p, int) else _claim(p) for p in picks]
    return draft.model_copy(update={"claims": claims})


DRAFT = _draft("d", n_claims=3)


def _problems(picks: list[int | str], notes: list[CriticNote]) -> list[str]:
    return critic_consistency_problems(
        DRAFT, CriticOutput(review=notes, final=_final_from(DRAFT, picks))
    )


def test_consistent_kept_rejected_rewritten_added():
    notes = [_note(0, 0, K), _note(1, None, RJ), _note(2, 1, RW), _note(None, 2, AD)]
    assert _problems([0, "rewritten2", "new"], notes) == []


def test_critic_may_return_fewer_claims():
    notes = [_note(0, None, RJ), _note(1, 0, K), _note(2, None, RJ)]
    assert _problems([1], notes) == []


def test_final_order_may_differ_from_draft_order():
    assert _problems([2, 0, 1], [_note(0, 1, K), _note(1, 2, K), _note(2, 0, K)]) == []


def test_missing_note_for_draft_claim():
    problems = _problems([0, 1], [_note(0, 0, K), _note(1, 1, K)])
    assert "draft claim 2 has 0 review notes; it needs exactly one" in problems


def test_duplicate_note_for_draft_claim():
    notes = [_note(0, 0, K), _note(0, None, RJ), _note(1, 1, K), _note(2, 2, K)]
    assert "draft claim 0 has 2 review notes; it needs exactly one" in _problems([0, 1, 2], notes)


def test_added_note_when_nothing_was_added_is_rejected():
    # the week-1 bug: every draft claim kept, plus an "added" note for nothing
    kept = [_note(0, 0, K), _note(1, 1, K), _note(2, 2, K)]
    for phantom in (_note(None, None, AD), _note(None, 2, AD)):
        assert _problems([0, 1, 2], [*kept, phantom])
    assert any(
        "final_index must be null exactly when" in p
        for p in _problems([0, 1, 2], [*kept, _note(None, None, AD)])
    )
    assert (
        "final claim 2 is referenced by 2 review notes; it needs exactly one "
        "(its kept, rewritten, or added note)" in _problems([0, 1, 2], [*kept, _note(None, 2, AD)])
    )


def test_null_draft_index_requires_added_and_vice_versa():
    notes = [_note(0, 0, K), _note(1, 1, K), _note(2, 2, K), _note(None, 3, RW)]
    assert any(
        "draft_index must be null exactly when" in p for p in _problems([0, 1, 2, "x"], notes)
    )
    notes = [_note(0, 0, K), _note(1, 1, K), _note(2, 2, AD)]
    assert any("draft_index must be null exactly when" in p for p in _problems([0, 1, "x"], notes))


def test_rejected_iff_null_final_index():
    notes = [_note(0, 0, K), _note(1, 1, RJ), _note(2, 2, K)]
    assert any("final_index must be null exactly when" in p for p in _problems([0, 1, 2], notes))
    notes = [_note(0, 0, K), _note(1, None, RW), _note(2, 1, K)]
    assert any("final_index must be null exactly when" in p for p in _problems([0, 2], notes))


def test_unreferenced_final_claim():
    notes = [_note(0, 0, K), _note(1, None, RJ), _note(2, None, RJ)]
    assert _problems([0, "extra"], notes) == [
        "final claim 1 is referenced by 0 review notes; it needs exactly one "
        "(its kept, rewritten, or added note)"
    ]


def test_kept_but_edited_is_accepted_and_reconciled_to_rewritten():
    changed = DRAFT.claims[1].model_copy(update={"confidence": 0.5})
    final = DRAFT.model_copy(update={"claims": [DRAFT.claims[0], changed, DRAFT.claims[2]]})
    notes = [_note(0, 0, K), _note(1, 1, K), _note(2, 2, K)]
    critic = CriticOutput(review=notes, final=final)
    assert critic_consistency_problems(DRAFT, critic) == []
    fixed, idx = reconcile_kept(DRAFT, critic)
    assert idx == [1]
    assert [n.verdict for n in fixed.review] == [K, RW, K]
    assert fixed.final == final


def test_kept_but_edited_truism_is_still_rejected():
    changed = DRAFT.claims[1].model_copy(update={"confidence": 0.5})
    final = DRAFT.model_copy(update={"claims": [DRAFT.claims[0], changed, DRAFT.claims[2]]})
    notes = [
        _note(0, 0, K),
        _note(1, 1, K).model_copy(update={"defects": [Defect.truism]}),
        _note(2, 2, K),
    ]
    problems = critic_consistency_problems(DRAFT, CriticOutput(review=notes, final=final))
    assert any("truism" in p for p in problems)


def test_rewritten_must_differ():
    notes = [_note(0, 0, RW), _note(1, 1, K), _note(2, 2, K)]
    problems = _problems([0, 1, 2], notes)
    assert len(problems) == 1 and "must be 'kept'" in problems[0]


def test_indices_out_of_range():
    notes = [_note(0, 0, K), _note(1, 1, K), _note(2, 2, K), _note(7, 5, RW)]
    problems = _problems([0, 1, 2], notes)
    assert "review[3] (rewritten): draft_index 7 out of range (draft has 3 claims)" in problems
    assert "review[3] (rewritten): final_index 5 out of range (final has 3 claims)" in problems
    notes = [_note(-1, 0, K), _note(1, 1, K), _note(2, 2, K)]
    assert any("draft_index -1 out of range" in p for p in _problems([0, 1, 2], notes))


def test_inconsistent_critic_is_retried_then_accepted(settings, paper):
    draft = _draft("draft")
    final = _final_from(draft, [0, 1, 2])
    bad = CriticOutput(
        review=[_note(0, 0, K), _note(1, 1, K), _note(2, 2, K), _note(None, None, AD)],
        final=final,
    )
    good = CriticOutput(review=bad.review[:3], final=final)
    fake = ScriptedTransport([draft], [bad, good])
    claims, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(fake.calls) == 3
    retry = fake.calls[2].user
    assert "<previous_attempt_error>\nAttempt 2:" in retry
    assert "final_index must be null exactly when the verdict is 'rejected'" in retry
    assert claims.extraction_metadata.retries == 1
    assert [n.verdict for n in claims.extraction_metadata.critic_review] == [K, K, K]


def test_inconsistent_critic_retries_then_falls_back(settings, paper):
    draft = _draft("draft")
    bad = CriticOutput(review=[_note(0, 0, K)], final=_final_from(draft, [0, 1, 2]))
    fake = ScriptedTransport([draft], [bad, bad, bad])
    claims, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert "critic pass still invalid after 2 retries" in claims.extraction_metadata.critic_fallback
    assert claims.claims == draft.claims
    assert len(fake.calls) == 4
    # each retry is a distinct request (and so a distinct cache key)
    assert len({c.user for c in fake.calls[1:]}) == 3


def test_critic_with_fewer_claims_end_to_end(settings, paper):
    draft = _draft("draft")
    final = _final_from(draft, [2])
    notes = [_note(0, None, RJ), _note(1, None, RJ), _note(2, 0, K)]
    fake = ScriptedTransport([draft], [CriticOutput(review=notes, final=final)])
    claims, pass1 = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(pass1.claims) == 3 and [c.statement for c in claims.claims] == ["draft2 statement"]


# --------------------------------------------------------------------------- prompt versions


class TagRecordingClient(LLMClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tags: list[str] = []

    def structured(self, *, cache_tag: str, **kwargs):
        self.tags.append(cache_tag)
        return super().structured(cache_tag=cache_tag, **kwargs)


@pytest.mark.parametrize(
    ("extract", "critic"), [("extract_v2", "critic_v2"), ("extract_v1", "critic_v1")]
)
def test_prompts_load_by_configured_version(tmp_path, paper, extract, critic):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        anthropic_api_key=None,
        extract_prompt=extract,
        critic_prompt=critic,
    )
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    client = TagRecordingClient(settings, fake)
    claims, _ = extract_claims(paper, "claude-sonnet-5", client, settings)
    assert claims.extraction_metadata.prompt_version == f"{extract}+{critic}"
    assert fake.calls[0].system == load_prompt(extract).text
    assert fake.calls[1].system == load_prompt(critic).text
    assert client.tags == [extract, f"{extract}+{critic}"]


def test_tables_reach_both_passes_with_configured_budget(tmp_path, paper):
    settings = Settings(
        _env_file=None,
        critic_prompt="critic_v3",
        number_grounding=False,
        data_dir=tmp_path,
        anthropic_api_key=None,
    )
    settings.table_char_budget = 40
    rows = [["Model", "Acc"]] + [[f"m{i}", f"{i}.5"] for i in range(20)]
    paper = paper.model_copy(update={"tables": [Table(label="Table 1", caption="R.", rows=rows)]})
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    for call in fake.calls:
        assert "<tables>\n### Table 1: R.\nModel | Acc\nm0 | 0.5\n" in call.user
        assert "more rows not shown]" in call.user and "m19 | 19.5" not in call.user


# --------------------------------------------------------------------------- defects


def test_defect_rules_on_consistent_notes():
    ok = [_note(0, 0, K), _note(1, None, RJ, [Defect.truism]), _note(2, 1, RW)]
    assert _problems([0, "rewritten2"], ok) == []


def test_kept_with_defects_is_a_problem():
    notes = [_note(0, 0, K, [Defect.missing_numbers]), _note(1, 1, K), _note(2, 2, K)]
    assert _problems([0, 1, 2], notes) == [
        "review[0] (kept): a kept claim must have no defects, got ['missing_numbers']; "
        "fix it (rewritten) or reject it"
    ]


@pytest.mark.parametrize("verdict", [RW, RJ])
def test_rewritten_or_rejected_needs_a_defect(verdict):
    final_index = 1 if verdict == RW else None
    picks: list[int | str] = [0, "changed"] if verdict == RW else [0]
    notes = [_note(0, 0, K), _note(1, final_index, verdict, []), _note(2, None, RJ)]
    assert _problems(picks, notes) == [
        f"review[1] ({verdict.value}): a {verdict.value} claim must list at least one defect"
    ]


@pytest.mark.parametrize("verdict", [K, RW])
def test_truism_must_be_rejected(verdict):
    picks: list[int | str] = [0, 1 if verdict == K else "changed", 2]
    notes = [_note(0, 0, K), _note(1, 1, verdict, [Defect.truism]), _note(2, 2, K)]
    assert any(
        p.startswith(
            f"review[1] ({verdict.value}): a claim with the truism defect must be rejected"
        )
        for p in _problems(picks, notes)
    )


# The v1/v2 run's DeepSpeed ViT truism: the critic rewrote it instead of rejecting it.
DEEPSPEED_TRUISM = (
    "Increasing the number of GPUs in DeepSpeed-based distributed ViT training increases "
    "communication overhead, as observed on the Nebula cluster, which must be balanced against "
    "speedup gains."
)


def _deepspeed_draft() -> ClaimsDraft:
    specific = Claim(
        statement=(
            "DeepSpeed ZeRO stage-2 data-parallel training of ViT-B/16 on ImageNet-1k reaches a "
            "5.1x speedup on 8 GPUs over a single GPU while keeping top-1 accuracy within 0.3 "
            "points."
        ),
        contribution_type="empirical_finding",
        method="DeepSpeed ZeRO stage-2 data parallelism for Vision Transformer training",
        problem="Reducing wall-clock training time of Vision Transformers",
        setting="ViT-B/16, ImageNet-1k, 1 to 8 GPUs on the Nebula cluster",
        result="5.1x speedup on 8 GPUs vs 1 GPU; top-1 accuracy within 0.3 points",
        novelty_assertion="First systematic study of DeepSpeed scaling for ViTs on Nebula",
        evidence=["Table 2", "4 Results"],
        confidence=0.85,
    )
    truism = Claim(
        statement=DEEPSPEED_TRUISM,
        contribution_type="empirical_finding",
        method="DeepSpeed data-parallel training of Vision Transformers",
        problem="Scaling distributed ViT training across GPUs",
        setting="ViT on the Nebula cluster, increasing GPU counts",
        result="Communication overhead grows with the number of GPUs",
        novelty_assertion="The authors assert nothing specific",
        evidence=["4 Results"],
        confidence=0.6,
    )
    return ClaimsDraft(
        one_line_summary="DeepSpeed speeds up ViT training on the Nebula cluster.",
        claims=[specific, truism],
        keywords=["DeepSpeed", "ZeRO", "Vision Transformer", "data parallelism", "ImageNet"],
    )


def _deepspeed_rewrite(draft: ClaimsDraft) -> CriticOutput:
    """The v2 failure: the truism is 'rewritten' (made qualitative, confidence lowered)."""
    softened = draft.claims[1].model_copy(
        update={
            "statement": "Communication overhead increases as the number of GPUs scales up "
            "during DeepSpeed data-parallel training of ViTs.",
            "confidence": 0.5,
        }
    )
    final = draft.model_copy(update={"claims": [draft.claims[0], softened]})
    notes = [_note(0, 0, K), _note(1, 1, RW, [Defect.truism])]
    return CriticOutput(review=notes, final=final)


def _deepspeed_reject(draft: ClaimsDraft) -> CriticOutput:
    final = draft.model_copy(update={"claims": [draft.claims[0]]})
    notes = [_note(0, 0, K), _note(1, None, RJ, [Defect.truism])]
    return CriticOutput(review=notes, final=final)


def test_deepspeed_truism_rewritten_is_a_problem_and_retried(settings, paper):
    draft = _deepspeed_draft()
    assert critic_consistency_problems(draft, _deepspeed_rewrite(draft)) == [
        "review[1] (rewritten): a claim with the truism defect must be rejected (removed from "
        "final), never rewritten or kept"
    ]
    fake = ScriptedTransport([draft], [_deepspeed_rewrite(draft), _deepspeed_reject(draft)])
    claims, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert len(fake.calls) == 3
    assert "a claim with the truism defect must be rejected" in fake.calls[2].user
    assert claims.extraction_metadata.retries == 1
    assert all(c.statement != DEEPSPEED_TRUISM for c in claims.claims)
    assert all("Communication overhead" not in c.statement for c in claims.claims)


def test_deepspeed_truism_rejected_is_accepted_and_absent(settings, paper):
    draft = _deepspeed_draft()
    assert critic_consistency_problems(draft, _deepspeed_reject(draft)) == []
    fake = ScriptedTransport([draft], [_deepspeed_reject(draft)])
    claims, pass1 = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert claims.extraction_metadata.retries == 0
    assert DEEPSPEED_TRUISM in [c.statement for c in pass1.claims]
    assert [c.statement for c in claims.claims] == [draft.claims[0].statement]
    review = claims.extraction_metadata.critic_review
    assert (review[1].verdict, review[1].defects) == (RJ, [Defect.truism])


def test_deepspeed_truism_kept_with_defects_is_a_problem():
    draft = _deepspeed_draft()
    kept = CriticOutput(review=[_note(0, 0, K), _note(1, 1, K, [Defect.truism])], final=draft)
    problems = critic_consistency_problems(draft, kept)
    assert "review[1] (kept): a claim with the truism defect must be rejected" in problems[0]
    assert any("a kept claim must have no defects, got ['truism']" in p for p in problems)


def test_v3_prompts_load_and_run(tmp_path, paper):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        anthropic_api_key=None,
        extract_prompt="extract_v3",
        critic_prompt="critic_v3",
    )
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    claims, _ = extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    assert claims.extraction_metadata.prompt_version == "extract_v3+critic_v3"
    assert fake.calls[1].system == load_prompt("critic_v3").text


def test_fallback_pages_reach_both_passes(settings, paper):
    fb = Section(
        heading="Page 7", paragraphs=["Table 3 accuracy 88.2"], fallback=True, fallback_page=7
    )
    paper = paper.model_copy(update={"sections": [*paper.sections, fb]})
    fake = ScriptedTransport([_draft("draft")], [_critic(_draft("final"))])
    extract_claims(paper, "claude-sonnet-5", LLMClient(settings, fake), settings)
    for call in fake.calls:
        assert "<fallback_pages>\nPage 7 (pdftotext fallback" in call.user
        assert "Table 3 accuracy 88.2" in call.user


def test_defect_rules_only_from_critic_v3():
    from reviewer.claims.extractor import uses_defects

    assert not uses_defects("critic_v1") and not uses_defects("critic_v2")
    assert uses_defects("critic_v3") and uses_defects("critic_v12")
