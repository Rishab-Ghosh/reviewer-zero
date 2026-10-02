import json
import re
from datetime import date

import httpx
import numpy as np
import pytest
import typer

from reviewer.config import PROJECT_ROOT, Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    Candidate,
    CandidateJudgment,
    CandidateJudgmentV2,
    CandidateJudgmentV3,
    CandidateSource,
    Claim,
    ExpandCandidate,
    ExpandResponse,
    ExpandSeedLink,
    ExtractionMetadata,
    GeneratedQuery,
    IndexCaseRow,
    IndexPaper,
    JudgedCandidate,
    ListwiseItem,
    ListwiseRanking,
    NoveltyEvalCase,
    NoveltyLabel,
    NoveltyQuery,
    NoveltyVerdict,
    PaperClaims,
    PaperNovelty,
    PriorWorkRef,
    QueryHit,
    QueryKind,
    QueryPlan,
    RerankOutput,
    RerankOutputV2,
    RerankOutputV3,
    SearchQuery,
    SeverityOutput,
    TokenUsage,
)
from reviewer.novelty.checker import Checker
from reviewer.novelty.commands import run_check, run_eval, run_first_stage_eval, run_negatives
from reviewer.novelty.evaluate import (
    case_query,
    case_target,
    group_cases,
    matches,
    micro,
    recall_at,
    sample_cases,
    score_case,
    venue_cutoff,
)
from reviewer.novelty.firststage import (
    FEATURES,
    CrossValidatedScorer,
    LogisticModel,
    cross_validated_scores,
    fit_logistic,
    gather_candidates,
    retrieve_first_stage,
    rule_score,
)
from reviewer.novelty.index_client import HttpIndexClient, IndexClientError
from reviewer.novelty.queries import plan_problems
from reviewer.novelty.report import (
    TOP_SHOWN,
    duplicate_flag,
    low_evidence_note,
    render_claim,
    render_report,
)
from reviewer.novelty.rerank import (
    OVERLAP_LABELS,
    RerankError,
    judgment_problems,
    listwise_problems,
    load_rerank_prompt,
    rank,
    render_user,
    verdict,
)
from reviewer.novelty.retrieve import (
    OWN_PREPRINT_JACCARD,
    RetrievalParams,
    Target,
    arxiv_month_end,
    interleave,
    is_future,
    is_own_preprint,
    is_self,
    method_problem_query,
    resolve_v1_date,
    retrieve,
    retrieve_multi,
    target_for,
    text_jaccard,
)
from reviewer.privacy import ensure_private

S, SUB, C, D = NoveltyLabel.same, NoveltyLabel.subsumes, NoveltyLabel.close, NoveltyLabel.different


def _p(key: str, sim: float = 0.5, title: str | None = None, **kw) -> IndexPaper:
    return IndexPaper(
        key=key, title=title or f"paper {key}", abstract=f"abs {key}", similarity=sim, **kw
    )


class FakeIndex:
    def __init__(
        self,
        search: dict[str, list[IndexPaper]] | None = None,
        neighbors: dict[str, list[IndexPaper]] | None = None,
        version: str = "v1",
    ) -> None:
        self._search = search or {}
        self._neighbors = neighbors or {}
        self._version = version
        self.calls: list[tuple[str, str]] = []
        self._expand: dict[str, ExpandCandidate] = {}
        self.expand_queries: list[str] = []
        self.befores: list[date | None] = []

    def version(self) -> str:
        return self._version

    def search(self, q: str, k: int, before: date | None = None) -> list[IndexPaper]:
        self.calls.append(("search", q))
        self.befores.append(before)
        return self._search.get(q, [])[:k]

    def neighbors(self, key: str, q: str, before: date | None = None) -> list[IndexPaper]:
        self.calls.append(("neighbors", key))
        self.befores.append(before)
        return self._neighbors.get(key, [])

    def expand(
        self, seeds: list[str], queries: list[str], before: date | None, max_candidates: int
    ) -> ExpandResponse:
        """Candidates from `self._expand` (key -> ExpandCandidate) linked to any seed sent."""
        self.calls.append(("expand", ",".join(seeds)))
        self.expand_queries = queries
        out = [c for c in self._expand.values() if any(link.seed in seeds for link in c.seeds)]
        return ExpandResponse(candidates=out[:max_candidates], total=len(out))


QUERY = NoveltyQuery(statement="STMT", method="M", problem="P")
MP = "M. P"
PARAMS = RetrievalParams(search_k=50, expand_top=10, neighbors_per_node=20, pool_size=60)


# --------------------------------------------------------------------------- retrieval


def test_union_keeps_best_similarity_and_both_sources():
    index = FakeIndex(
        search={"STMT": [_p("a", 0.9), _p("b", 0.5)], MP: [_p("b", 0.8), _p("c", 0.7)]}
    )
    pool, counts = retrieve(QUERY, index, Target(), PARAMS)
    assert [c.paper.key for c in pool] == ["a", "b", "c"]
    b = pool[1]
    assert b.similarity == 0.8
    assert b.sources == [CandidateSource.search_statement, CandidateSource.search_method_problem]
    assert (counts.search_statement, counts.search_method_problem, counts.search_union) == (2, 2, 3)


def test_method_problem_query_skipped_when_empty():
    index = FakeIndex(search={"STMT": [_p("a")]})
    retrieve(NoveltyQuery(statement="STMT"), index, Target(), PARAMS)
    assert [c for c in index.calls if c[0] == "search"] == [("search", "STMT")]
    assert method_problem_query(QUERY) == "M. P"


def test_self_and_future_filtered():
    target = Target(
        arxiv_id="2504.15895", title="My Great Paper About Things", cutoff=date(2025, 4, 30)
    )
    hits = [
        _p("2504.15895", 0.99),  # self by id
        _p("openalex:W1", 0.98, title="My great paper about things!"),  # self by title
        _p("2504.15896", 0.97),  # later arXiv id, same month
        _p("2504.15000", 0.96),  # earlier arXiv id
        _p("openalex:W2", 0.95, published_date=date(2025, 5, 2)),  # after cutoff
        _p("openalex:W3", 0.94, year=2024),  # no date, year before cutoff
        _p("openalex:W5", 0.935, year=2026),  # no date, year after cutoff
        _p("openalex:W6", 0.934, year=2025),  # same year as cutoff: kept
        _p("openalex:W4", 0.93),  # neither date nor year: dropped
        IndexPaper(key="openalex:W8", title=None, abstract="a", year=2020, similarity=0.92),
        IndexPaper(
            key="openalex:W9", title="Bad record", abstract=None, year=2020, similarity=0.91
        ),
    ]
    pool, counts = retrieve(
        NoveltyQuery(statement="STMT"), FakeIndex(search={"STMT": hits}), target, PARAMS
    )
    assert [c.paper.key for c in pool] == ["2504.15000", "openalex:W3", "openalex:W6"]
    assert (counts.filtered_self, counts.filtered_future) == (2, 3)
    assert (counts.filtered_undated, counts.filtered_no_metadata) == (1, 2)


def test_cutoff_is_sent_as_before():
    index = FakeIndex(search={"STMT": [_p("a", 0.9, year=2024)], MP: [_p("b", 0.8, year=2024)]})
    retrieve(QUERY, index, Target(cutoff=date(2025, 4, 17)), PARAMS)
    assert len(index.befores) == 4 and set(index.befores) == {date(2025, 4, 17)}  # 2 + 2 seeds
    index = FakeIndex(search={"STMT": [_p("a", 0.9)]})
    retrieve(NoveltyQuery(statement="STMT"), index, Target(), PARAMS)
    assert set(index.befores) == {None}


def test_resolve_v1_date():
    target = target_for("2504.15895", "Dynamic Early Exit in Reasoning Models", cutoff=True)
    hit = _p("2504.15895", 0.9, published_date=date(2025, 4, 22))
    index = FakeIndex(search={"Dynamic Early Exit in Reasoning Models": [_p("x"), hit]})
    resolved = resolve_v1_date(index, target)
    assert resolved.cutoff == date(2025, 4, 22)
    assert resolved.cutoff_source == "arXiv v1 date (index)"
    missing = resolve_v1_date(FakeIndex(), target)
    assert missing.cutoff == date(2025, 4, 30) and "not in index" in (missing.cutoff_source or "")
    own = target_for("own-x-12345678", "t", cutoff=True)
    assert resolve_v1_date(FakeIndex(), own) == own


def test_no_cutoff_keeps_newer_work():
    target = target_for("own-mypaper-12345678", "Own paper title here", cutoff=True)
    assert target.cutoff is None and target.arxiv_id is None
    assert not is_future(_p("2609.00001"), target)
    undated = _p("openalex:W4")
    pool, counts = retrieve(
        NoveltyQuery(statement="STMT"), FakeIndex(search={"STMT": [undated]}), target, PARAMS
    )
    assert [c.paper.key for c in pool] == ["openalex:W4"] and counts.filtered_undated == 0
    assert is_self(_p("x", title="Own paper title here"), target)


def test_arxiv_month_end_and_target():
    assert arxiv_month_end("2504.15895") == date(2025, 4, 30)
    assert arxiv_month_end("2402.00001") == date(2024, 2, 29)
    assert arxiv_month_end("cs/0112017") is None
    assert target_for("2504.15895", "t", cutoff=True).cutoff == date(2025, 4, 30)


def test_expansion_counts_cap_and_pool():
    search = {"STMT": [_p(f"s{i}", 0.9 - i * 0.01) for i in range(12)]}
    neighbors = {
        "s0": [_p(f"n{i}", 0.5 - i * 0.01, relation="cites") for i in range(25)] + [_p("s5", 0.99)],
        "s1": [],
    }
    params = RetrievalParams(search_k=50, expand_top=10, neighbors_per_node=20, pool_size=25)
    pool, counts = retrieve(
        NoveltyQuery(statement="STMT"), FakeIndex(search, neighbors), Target(), params
    )
    assert counts.expansion_nodes == 10
    assert counts.expansion_nodes_empty == 9  # only s0 has neighbors
    assert counts.expansion_new == 19  # 20 kept per node, one of them (s5) already a search hit
    assert counts.pool == 25
    assert counts.pool_from_expansion == 13  # 12 search hits fill first, then 13 neighbors
    s5 = next(c for c in pool if c.paper.key == "s5")
    assert s5.via == ["s0"] and CandidateSource.expansion in s5.sources


def test_duplicate_titles_collapse():
    hits = [
        _p("2001.00001", 0.8, title="Same Title Of A Paper"),
        _p("openalex:W9", 0.7, title="Same title of a paper"),
    ]
    pool, _ = retrieve(
        NoveltyQuery(statement="STMT"), FakeIndex(search={"STMT": hits}), Target(), PARAMS
    )
    assert [c.paper.key for c in pool] == ["2001.00001"]


# --------------------------------------------------------------------------- rerank + verdict


def _judged(key: str, label: NoveltyLabel, sim: float) -> JudgedCandidate:
    cand = Candidate(paper=_p(key, sim), similarity=sim, sources=[CandidateSource.search_statement])
    return JudgedCandidate(candidate=cand, label=label, reason="r")


def test_rank_by_label_then_similarity():
    ranked = rank(
        [_judged("a", D, 0.9), _judged("b", C, 0.5), _judged("c", S, 0.1), _judged("d", C, 0.7)]
    )
    assert [j.candidate.paper.key for j in ranked] == ["c", "d", "b", "a"]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([D, D], NoveltyVerdict.novel),
        ([C, D], NoveltyVerdict.incremental),
        ([SUB, C], NoveltyVerdict.already_done),
        ([S], NoveltyVerdict.already_done),
        ([], NoveltyVerdict.novel),
    ],
)
def test_verdict(labels, expected):
    assert verdict([_judged(str(i), lab, 0.5) for i, lab in enumerate(labels)]) == expected


def test_judgment_problems():
    ok = RerankOutput(
        judgments=[CandidateJudgment(candidate_id=i, label=D, reason="r") for i in range(3)]
    )
    assert judgment_problems(ok, 3) == []
    bad = RerankOutput(
        judgments=[CandidateJudgment(candidate_id=i, label=D, reason="r") for i in (0, 0, 5)]
    )
    (problem,) = judgment_problems(bad, 3)
    assert "missing [1, 2]" in problem and "[0, 5]" in problem


def test_render_user_marks_missing_abstract():
    cand = Candidate(
        paper=IndexPaper(key="x", title="T"), similarity=0.1, sources=[CandidateSource.expansion]
    )
    text = render_user(QUERY, [cand])
    assert (
        '<candidate id="0">' in text and "[no abstract available]" in text and "Method: M" in text
    )


TAGS = {"SAME": S, "SUBS": SUB, "CLOSE": C}


class FakeLLM:
    """Labels a candidate by a tag in its title (SAME/SUBS/CLOSE; else different). `drop_first`
    answers the first N calls with a judgment missing, to exercise retries."""

    def __init__(self, drop_first: int = 0) -> None:
        self.drop_first = drop_first
        self.calls = 0
        self.query_calls: list[str] = []
        self.listwise_calls = 0
        self.severity_calls = 0

    def __call__(self, req: TransportRequest) -> TransportResponse:
        if req.schema is QueryPlan:
            return self._plan(req)
        if req.schema is ListwiseRanking:
            return self._listwise(req)
        if req.schema is SeverityOutput:
            return self._severity(req)
        self.calls += 1
        titles = re.findall(r'<candidate id="(\d+)">\nTitle: ([^\n]*)', req.user)
        judgments = [
            CandidateJudgment(
                candidate_id=int(i),
                label=next((lab for tag, lab in TAGS.items() if tag in t), D),
                reason=f"because {t}",
            )
            for i, t in titles
        ]
        if self.calls <= self.drop_first:
            judgments = judgments[1:]
        if req.schema is RerankOutputV2:
            parsed = RerankOutputV2(
                judgments=[CandidateJudgmentV2(**j.model_dump()) for j in judgments]
            )
        elif req.schema is RerankOutputV3:
            parsed = RerankOutputV3(
                judgments=[
                    CandidateJudgmentV3(
                        **j.model_dump(exclude={"label"}),
                        label=NoveltyLabel.builds_on if "BASE" in titles[k][1] else j.label,
                    )
                    for k, j in enumerate(judgments)
                ]
            )
        else:
            parsed = RerankOutput(judgments=judgments)
        return TransportResponse(
            parsed=parsed,
            usage=TokenUsage(input_tokens=100, output_tokens=10),
            stop_reason="end_turn",
        )

    def _listwise(self, req: TransportRequest) -> TransportResponse:
        """Candidates whose title has FIRST go first; the rest in reverse of the order shown."""
        self.listwise_calls += 1
        titles = re.findall(r'<candidate id="(\d+)">\nTitle: ([^\n]*)', req.user)
        order = [int(i) for i, t in titles if "FIRST" in t] + [
            int(i) for i, t in reversed(titles) if "FIRST" not in t
        ]
        ranking = ListwiseRanking(
            ranking=[ListwiseItem(candidate_id=i, reason=f"cite {i}") for i in order]
        )
        return TransportResponse(
            parsed=ranking,
            usage=TokenUsage(input_tokens=70, output_tokens=7),
            stop_reason="end_turn",
        )

    def _severity(self, req: TransportRequest) -> TransportResponse:
        """3 when a candidate is labelled same, 1 when any is close, else 0; drives = first key."""
        self.severity_calls += 1
        keys = re.findall(r'<candidate key="([^"]+)">\n[^<]*?Label: (\w+)', req.user)
        labels = [lab for _, lab in keys]
        score = 3 if "same" in labels else 1 if "close" in labels else 0
        drivers = [next(k for k, lab in keys if lab in ("same", "close"))] if score else []
        out = SeverityOutput(score=score, drivers=drivers, reason=f"severity {score}")
        return TransportResponse(
            parsed=out, usage=TokenUsage(input_tokens=40, output_tokens=4), stop_reason="end_turn"
        )

    def _plan(self, req: TransportRequest) -> TransportResponse:
        """Six queries 'GEN<i> <origin>' where origin is 'paper' or 'claim' (from the input)."""
        origin = "paper" if req.user.startswith("<paper>") else "claim"
        self.query_calls.append(origin)
        kinds = list(QueryKind)
        plan = QueryPlan(
            queries=[
                GeneratedQuery(kind=kinds[i], component=f"comp {i}", text=f"GEN{i} {origin}")
                for i in range(6)
            ]
        )
        return TransportResponse(
            parsed=plan, usage=TokenUsage(input_tokens=50, output_tokens=50), stop_reason="end_turn"
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    """v0 behaviour (baseline queries only); multi-query tests use `multi_settings`."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        novelty_model_dir=tmp_path / "data" / "novelty",
        anthropic_api_key=None,
        llm_max_retries=2,
        novelty_batch_size=3,
        novelty_pool_size=60,
        novelty_first_stage="roundrobin",
        novelty_multi_query=False,
        novelty_prompt="rerank_v1",
        novelty_listwise=False,
        novelty_severity=False,
    )


@pytest.fixture
def multi_settings(settings) -> Settings:
    return settings.model_copy(update={"novelty_multi_query": True, "novelty_pool_size": 150})


def _checker(settings: Settings, index: FakeIndex, llm: FakeLLM) -> Checker:
    client = LLMClient(settings, transport=llm)
    return Checker.from_settings(
        settings, index, client, load_rerank_prompt(settings.novelty_prompt), "claude-sonnet-5"
    )


def _index_for_claim() -> FakeIndex:
    hits = [
        _p("2401.00001", 0.9, title="Unrelated A"),
        _p("2401.00002", 0.8, title="CLOSE cousin"),
        _p("2401.00003", 0.7, title="Unrelated B"),
        _p("2401.00004", 0.6, title="Unrelated C"),
    ]
    neighbors = {"2401.00001": [_p("openalex:W7", 0.4, title="SAME idea, older", year=2019)]}
    return FakeIndex(search={"STMT": hits, MP: hits[:2]}, neighbors=neighbors)


def test_check_query_end_to_end_with_retry(settings):
    llm = FakeLLM(drop_first=1)
    result = _checker(settings, _index_for_claim(), llm).check_query(QUERY, Target(), 0)
    assert result.verdict == NoveltyVerdict.already_done
    assert [j.label for j in result.top[:2]] == [S, C]
    assert result.top[0].candidate.paper.key == "openalex:W7"
    assert result.counts.pool == 5 and result.counts.pool_from_expansion == 1
    assert len(result.top) == 5 and len(result.ranked) == 5
    assert llm.calls == 3  # 2 batches (3 + 2) plus one retry


def test_rerank_error_after_retries(settings):
    with pytest.raises(RerankError):
        _checker(settings, _index_for_claim(), FakeLLM(drop_first=99)).check_query(
            QUERY, Target(), 0
        )


# --------------------------------------------------------------------------- commands


def _claims(paper_id: str) -> PaperClaims:
    claim = Claim(
        statement="STMT",
        contribution_type="new_method",
        method="M",
        problem="P",
        setting="s",
        result="r",
        novelty_assertion="n",
        evidence=["Table 1"],
        confidence=0.9,
    )
    meta = ExtractionMetadata(
        model="m", prompt_version="v", timestamp="2026-09-25T00:00:00Z", token_usage=TokenUsage()
    )
    return PaperClaims(
        paper_id=paper_id,
        title="A lab paper title here",
        one_line_summary="s",
        claims=[claim],
        keywords=["a", "b", "c", "d", "e"],
        extraction_metadata=meta,
    )


def _write_corpus(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_file.write_text(
        "papers:\n"
        "  - {paper_id: '2504.15895', tier: lab}\n"
        "  - {paper_id: '2505.18440', tier: lab}\n"
        "  - {paper_id: '1706.03762', tier: famous}\n",
        encoding="utf-8",
    )
    settings.claims_dir.mkdir(parents=True)
    (settings.claims_dir / "2504.15895.json").write_text(_claims("2504.15895").model_dump_json())


def test_select_entries_extra_covers_ids_outside_corpus_yaml():
    from reviewer.models import CorpusEntry, PaperTier
    from reviewer.novelty.commands import select_entries

    corpus = [CorpusEntry(paper_id="1706.03762", tier=PaperTier.famous)]
    extra = [CorpusEntry(paper_id="or-AbC123", tier=PaperTier.ordinary, reason="meta dev run")]
    out = select_entries(corpus, ["or-AbC123"], None, extra)
    assert [e.paper_id for e in out] == ["or-AbC123"]


def test_select_entries_corpus_yaml_wins_over_extra_for_the_same_id():
    from reviewer.models import CorpusEntry, PaperTier
    from reviewer.novelty.commands import select_entries

    real = CorpusEntry(paper_id="1706.03762", tier=PaperTier.famous)
    stale_extra = CorpusEntry(paper_id="1706.03762", tier=PaperTier.ordinary)
    out = select_entries([real], ["1706.03762"], None, [stale_extra])
    assert out == [real]


def test_select_entries_still_raises_for_ids_in_neither():
    from reviewer.models import CorpusEntry, PaperTier
    from reviewer.novelty.commands import select_entries

    with pytest.raises(typer.BadParameter, match="not in corpus"):
        select_entries([CorpusEntry(paper_id="1706.03762", tier=PaperTier.famous)], ["nope"], None)


def test_run_check_writes_private_outputs_and_records_failures(settings):
    _write_corpus(settings)
    run_check(
        settings=settings, index=_index_for_claim(), client=LLMClient(settings, transport=FakeLLM())
    )
    out = PaperNovelty.model_validate_json((settings.novelty_dir / "2504.15895.json").read_text())
    assert out.cutoff == "2025-04-30" and out.tier == "lab" and out.index_version == "v1"
    assert out.cutoff_source == "end of arXiv submission month (v1 date not in index)"
    assert out.claims[0].verdict == NoveltyVerdict.already_done
    assert not (settings.novelty_dir / "1706.03762.json").exists()  # famous not selected
    report = (settings.private_reports_dir / "novelty_v0.md").read_text()
    assert "2504.15895" in report and "from expansion only" in report and "2505.18440" in report
    assert "*Low evidence: only 5 candidates were retrieved and judged.*" in report
    failures = settings.failures_file.read_text()
    assert "2505.18440" in failures and '"novelty"' in failures


def test_run_check_exits_when_index_down(settings):
    class Down(FakeIndex):
        def version(self) -> str:
            raise IndexClientError("connection refused")

    _write_corpus(settings)
    with pytest.raises(typer.Exit):
        run_check(settings=settings, index=Down(), client=LLMClient(settings, transport=FakeLLM()))


def test_ensure_private():
    ensure_private(PROJECT_ROOT / "reports" / "private" / "novelty_v0.md")
    with pytest.raises(RuntimeError):
        ensure_private(PROJECT_ROOT / "reports" / "novelty_v0.md")


def test_report_renders_empty_top():
    result = PaperNovelty(
        paper_id="x",
        title="t",
        tier="lab",
        cutoff=None,
        index_version="v",
        model="m",
        prompt_version="p",
        timestamp="2026-09-25T00:00:00Z",
        claims=[],
        token_usage=TokenUsage(),
    )
    assert "cutoff none (as of today)" in render_report([result], ["x: boom"])


def _claim_novelty(ranked, severity=None, pool=60):
    from reviewer.models import CandidateCounts, ClaimNovelty

    return ClaimNovelty(
        claim_index=0,
        query=QUERY,
        verdict=NoveltyVerdict.incremental,
        top=ranked[:5],
        ranked=ranked,
        counts=CandidateCounts(pool=pool),
        token_usage=TokenUsage(),
        severity=severity,
    )


def test_duplicate_flag_only_at_severity_3_and_names_the_driver():
    from reviewer.models import SeverityResult

    ranked = [_judged("2401.00001", S, 0.9), _judged("2401.00002", C, 0.8)]
    c = _claim_novelty(ranked)
    assert duplicate_flag(c) is None  # no severity computed at all

    c2 = _claim_novelty(
        ranked,
        severity=SeverityResult(
            score=2,
            drivers=["2401.00001"],
            reason="close but not the same",
            prompt_version="p",
            model="m",
        ),
    )
    assert duplicate_flag(c2) is None  # below 3: never flagged, whatever the label

    c3 = _claim_novelty(
        ranked,
        severity=SeverityResult(
            score=3,
            drivers=["2401.00001"],
            reason="this already exists",
            prompt_version="p",
            model="m",
        ),
    )
    flag = duplicate_flag(c3)
    assert flag is not None
    assert "Possible duplicate" in flag and "2401.00001" in flag and "this already exists" in flag
    assert "3" not in flag.replace("2401.00001", "")  # the raw severity number is never shown


def test_low_evidence_note_threshold():
    ranked = [_judged("2401.00001", C, 0.9)]
    assert low_evidence_note(_claim_novelty(ranked, pool=9)) is not None
    assert low_evidence_note(_claim_novelty(ranked, pool=10)) is None


def test_render_claim_shows_top_10_not_top_5_and_no_bare_verdict():
    ranked = [_judged(f"2401.0000{i}", C, 0.9 - i * 0.01) for i in range(12)]
    text = "\n".join(render_claim(_claim_novelty(ranked)))
    assert TOP_SHOWN == 10
    for i in range(10):
        assert f"2401.0000{i}" in text
    assert "2401.00010" not in text and "2401.00011" not in text  # 11th and 12th cut off
    assert "### Claim 1\n" in text  # no ": incremental" or similar after the header
    assert "incremental" not in text and "novel" not in text and "already_done" not in text


def test_render_report_possible_duplicate_column_and_no_verdict_column():
    from reviewer.models import SeverityResult

    ranked = [_judged("2401.00001", S, 0.9)]
    flagged = PaperNovelty(
        paper_id="p1",
        title="t",
        tier="lab",
        cutoff=None,
        index_version="v",
        model="m",
        prompt_version="p",
        timestamp="2026-09-25T00:00:00Z",
        token_usage=TokenUsage(),
        claims=[
            _claim_novelty(
                ranked,
                severity=SeverityResult(
                    score=3, drivers=["2401.00001"], reason="r", prompt_version="p", model="m"
                ),
            )
        ],
    )
    text = render_report([flagged], [])
    assert "| possible duplicate |" in text
    assert "| p1 | lab | 1 | yes |" in text
    assert "verdict" not in text.lower()
    assert "already_done" not in text and "incremental" not in text and "novel" not in text


# --------------------------------------------------------------------------- eval


def test_matching_and_recall():
    assert matches(PriorWorkRef(key="arXiv:2305.18290"), _p("2305.18290"))
    assert matches(
        PriorWorkRef(title="Direct Preference Optimization: Your LM"),
        _p("x", title="direct preference optimization your LM"),
    )
    assert not matches(PriorWorkRef(title="DPO"), _p("x", title="DPO"))  # too short to trust
    assert recall_at([1, 7, None, 20], 5) == 0.25
    assert recall_at([1, 7, None, 20], 20) == 0.75


def _row(sub: str, key: str | None, title: str = "T1", abstract: str = "A1", **kw) -> dict:
    return {
        "case_id": f"{sub}__{key}",
        "submission_id": sub,
        "venue": kw.pop("venue", "ICLR.cc/2025/Conference"),
        "submission_title": title,
        "submission_abstract": abstract,
        "review_excerpt": "ignored",
        "named_prior_work": {"title": kw.pop("named", f"named {key}"), "year": 2020},
        "resolution": {"status": "resolved" if key else "unresolved", "index_key": key},
    }


def test_group_cases_and_cutoffs():
    rows = [
        IndexCaseRow.model_validate(r)
        for r in (
            _row("s1", "openalex:W7"),
            _row("s1", "2401.00004"),
            _row("s1", "openalex:W7"),  # named in a second review
            _row("s1", None, named="Never Retrieved Paper Title"),
            _row("s1", "2410.00001")
            | {
                "resolution": {
                    "status": "resolved",
                    "index_key": "2410.00001",
                    "after_cutoff": True,
                }
            },
            _row("s2", None),  # nothing resolved: no case
            _row("s3", "2401.00001", title="T3", abstract="", venue="ICLR.cc/2024/Conference"),
        )
    ]
    s1, s3 = group_cases(rows)
    assert [r.key for r in s1.prior_work] == ["openalex:W7", "2401.00004"]
    assert s1.n_unresolved == 1 and s1.n_after_cutoff == 1
    assert s1.cutoff_date == date(2024, 10, 1)
    assert s3.cutoff_date == date(2023, 9, 28)
    assert venue_cutoff("NeurIPS.cc/2025/Conference") == date(2024, 12, 31)
    assert case_query(s3) == NoveltyQuery(statement="T3", problem="T3")
    assert case_target(s1).cutoff == date(2024, 10, 1) and case_target(s1).arxiv_id is None


def test_sample_cases_is_order_independent():
    cases = [NoveltyEvalCase(case_id=f"c{i}", title="t", prior_work=[]) for i in range(10)]
    a = [c.case_id for c in sample_cases(cases, 3)]
    assert a == [c.case_id for c in sample_cases(list(reversed(cases)), 3)] and len(a) == 3
    assert len(sample_cases(cases, None)) == 10


def test_run_eval_end_to_end(settings, tmp_path):
    hits = [
        _p("2401.00001", 0.9, title="Unrelated A", year=2023),
        _p("2401.00002", 0.8, title="CLOSE cousin", year=2023),
        _p("2401.00003", 0.7, title="Unrelated B", year=2023),
        _p("2401.00004", 0.6, title="Unrelated C", year=2023),
        _p("2501.00009", 0.95, title="Too new", published_date=date(2025, 1, 5)),
    ]
    neighbors = {"2401.00001": [_p("openalex:W7", 0.4, title="SAME idea, older", year=2019)]}
    index = FakeIndex(search={"T1 A1": hits, "T1": hits[:2]}, neighbors=neighbors)
    cases = tmp_path / "cases.jsonl"
    rows = [
        _row("s1", "openalex:W7"),
        _row("s1", "2401.00004"),
        _row("s1", None, named="Never Retrieved Paper Title"),
        _row("s3", "2401.00001", title="T3", abstract=""),
    ]
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = run_eval(
        "dev",
        cases_path=cases,
        settings=settings,
        index=index,
        client=LLMClient(settings, transport=FakeLLM()),
    )
    # s1: W7 rank 1 (same), 2401.00004 rank 5 (different, lowest sim); s3 retrieves nothing
    assert summary == {
        "recall_at_5": 2 / 3,
        "recall_at_10": 2 / 3,
        "recall_at_20": 2 / 3,
        "pool_recall": 2 / 3,
        "union_recall": 2 / 3,
    }
    data = json.loads((settings.private_reports_dir / "novelty_eval_dev.json").read_text())
    assert data["split"] == "dev" and len(data["commit"]) == 40
    (logged,) = [
        json.loads(line)
        for line in (settings.private_reports_dir / "novelty_eval_runs.jsonl")
        .read_text()
        .split("\n")
        if line
    ]
    assert logged["commit"] == data["commit"] and logged["summary"] == summary
    by_id = {r["case_id"]: r for r in data["results"]}
    assert by_id["s1"]["found_at"] == [1, 5] and by_id["s1"]["n_unresolved"] == 1
    assert by_id["s3"]["found_at"] == [None]
    report = (settings.private_reports_dir / "novelty_eval_dev.md").read_text()
    assert "recall@5" in report and "not in the index (excluded from recall): 1" in report


def test_run_eval_missing_cases_file(settings, tmp_path):
    with pytest.raises(typer.Exit):
        run_eval("dev", cases_path=tmp_path / "nope.jsonl", settings=settings, index=FakeIndex())


def test_test_split_needs_confirmation_and_clean_tree(settings, tmp_path, monkeypatch):
    import reviewer.novelty.commands as commands

    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps(_row("s1", "2401.00001")) + "\n")
    with pytest.raises(typer.BadParameter):  # test file under the dev split
        run_eval("dev", cases_path=test_file, settings=settings, index=FakeIndex())
    monkeypatch.setattr(commands, "git_state", lambda: ("a" * 40, False))
    with pytest.raises(typer.Exit):
        run_eval("test", cases_path=test_file, settings=settings, index=FakeIndex())
    monkeypatch.setattr(commands, "git_state", lambda: ("a" * 40, True))
    with pytest.raises(typer.Exit):
        run_eval(
            "test", confirm_test=True, cases_path=test_file, settings=settings, index=FakeIndex()
        )
    assert not (settings.private_reports_dir / "novelty_eval_runs.jsonl").exists()
    monkeypatch.setattr(commands, "git_state", lambda: ("a" * 40, False))
    run_eval(
        "test",
        confirm_test=True,
        cases_path=test_file,
        settings=settings,
        index=FakeIndex(),
        client=LLMClient(settings, transport=FakeLLM()),
    )
    logged = json.loads((settings.private_reports_dir / "novelty_eval_runs.jsonl").read_text())
    assert (logged["split"], logged["commit"], logged["dirty"]) == ("test", "a" * 40, False)


def test_row_cutoff_date_overrides_venue_deadline():
    row = IndexCaseRow.model_validate(_row("s1", "k") | {"cutoff_date": "2024-06-01"})
    (case,) = group_cases([row])
    assert case.cutoff_date == date(2024, 6, 1)


def test_micro_excludes_failed_cases():
    from reviewer.models import NoveltyEvalCaseResult

    ok = NoveltyEvalCaseResult(
        case_id="a",
        n_named=2,
        found_at=[1, None],
        in_pool=[True, False],
        recall_at_5=0.5,
        recall_at_20=0.5,
        pool_recall=0.5,
    )
    bad = NoveltyEvalCaseResult(
        case_id="b",
        n_named=3,
        found_at=[None] * 3,
        in_pool=[False] * 3,
        recall_at_5=0,
        recall_at_20=0,
        pool_recall=0,
        error="x",
    )
    assert micro([ok, bad]) == {
        "recall_at_5": 0.5,
        "recall_at_10": 0.5,
        "recall_at_20": 0.5,
        "pool_recall": 0.5,
        "union_recall": 0.0,  # in_union not set on this hand-built result
    }


def test_score_case_ranks():
    judged = [_judged("a", S, 0.9), _judged("b", D, 0.8)]
    from reviewer.models import CandidateCounts, ClaimNovelty

    result = ClaimNovelty(
        claim_index=0,
        query=QUERY,
        verdict=NoveltyVerdict.already_done,
        top=judged,
        ranked=judged,
        counts=CandidateCounts(),
        token_usage=TokenUsage(),
    )
    case = NoveltyEvalCase(
        case_id="c", title="t", prior_work=[PriorWorkRef(key="b"), PriorWorkRef(key="z")]
    )
    r = score_case(case, result)
    assert r.found_at == [2, None] and r.recall_at_5 == 0.5 and r.pool_recall == 0.5


# --------------------------------------------------------------------------- HTTP client


def _http_client(settings: Settings, handler) -> HttpIndexClient:
    return HttpIndexClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_http_client_contract_and_cache(settings):
    seen: list[str] = []
    version = {"v": "2026-09-25"}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.path}?{request.url.query.decode()}")
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": version["v"], "status": "ok"})
        if request.url.path == "/search":
            assert request.url.params["q"] == "dpo" and request.url.params["k"] == "50"
            return httpx.Response(
                200,
                json=[  # bare list, as reviewer-index serves it
                    {
                        "key": "2305.18290",
                        "year": 2023,
                        "title": "DPO",
                        "abstract": "a",
                        "published_date": "2023-05-29",
                        "similarity": 0.9,
                        "extra": 1,
                    }
                ],
            )
        if request.url.path == "/neighbors":
            if request.url.params["key"] == "missing":
                return httpx.Response(404, json={"detail": "unknown"})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"key": "1707.06347", "relation": "cites", "similarity": 0.4, "year": None}
                    ]
                },
                # wrapped form is accepted too
            )
        return httpx.Response(500)

    client = _http_client(settings, handler)
    (hit,) = client.search("dpo", 50)
    assert hit.published_date == date(2023, 5, 29)
    client.search("dpo", 50, before=date(2024, 1, 2))
    assert seen[-1].endswith("before=2024-01-02")
    assert client.neighbors("2305.18290", "dpo")[0].relation == "cites"
    assert client.neighbors("missing", "dpo") == []
    n_before = len(seen)
    client2 = _http_client(settings, handler)
    client2.search("dpo", 50)
    client2.neighbors("missing", "dpo")
    assert len(seen) == n_before + 1  # only /health; search and the 404 are cached
    version["v"] = "2026-09-26"
    client3 = _http_client(settings, handler)
    client3.search("dpo", 50)
    assert seen[-1].startswith("/search")  # new index version -> cache miss


def test_http_client_rejects_results_without_year(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": "v"})
        return httpx.Response(200, json=[{"key": "x", "title": "t", "abstract": "a"}])

    client = _http_client(settings, handler)
    with pytest.raises(IndexClientError, match="no 'year' field"):
        client.search("q", 5)
    assert not list(client.cache.dir.glob("*.json"))  # nothing cached


def test_http_client_errors(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": "v"})
        return httpx.Response(503, json={"detail": "database unavailable"})

    with pytest.raises(IndexClientError, match="HTTP 503"):
        _http_client(settings, handler).search("x", 5)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(IndexClientError, match="ConnectError"):
        _http_client(settings, refused).version()


def test_prompt_file_loads():
    assert load_rerank_prompt("rerank_v1").version == "rerank_v1"


# --------------------------------------------------------------------------- multi-query


def test_interleave_round_robin_dedupes_and_limits():
    lanes = [["a", "b", "c"], ["b", "d"], ["e"]]
    titles = {"d": "the same long title here", "e": "the same long title here"}
    assert interleave(lanes, 10, titles) == ["a", "b", "e", "c"]  # d shares e's title
    assert interleave(lanes, 2, {}) == ["a", "b"]


def test_single_query_top_hit_reaches_pool():
    """A work only one generated query finds, at rank 1, gets a pool slot ahead of the
    baseline query's long tail."""
    base = [_p(f"s{i}", 0.9 - i * 0.001, year=2020) for i in range(50)]
    niche = _p("niche", 0.3, year=2019)
    index = FakeIndex(search={"STMT": base, "GEN niche": [niche]})
    queries = [
        SearchQuery(id="b0", origin="baseline", kind="statement", text="STMT"),
        SearchQuery(id="p0", origin="paper", kind="technique", text="GEN niche"),
    ]
    params = RetrievalParams(search_k=50, query_k=30, expand_top=0, pool_size=5)
    pool, counts, unpooled = retrieve_multi(queries, index, Target(), params, "STMT")
    keys = [c.paper.key for c in pool]
    assert keys[:2] == ["s0", "niche"] and len(keys) == 5
    niche_c = pool[1]
    assert niche_c.hits == [QueryHit(query_id="p0", rank=1, similarity=0.3)]
    assert niche_c.sources == [CandidateSource.generated]
    assert counts.pool_from_generated_only == 1 and counts.n_queries == 2
    assert len(unpooled) == 46


def test_plan_problems():
    q = [GeneratedQuery(kind=QueryKind.technique, component="c", text=f"t{i}") for i in range(6)]
    assert plan_problems(QueryPlan(queries=q)) == []
    assert "expected 6 to 10" in plan_problems(QueryPlan(queries=q[:3]))[0]
    dup = QueryPlan(queries=[*q[:5], q[0]])
    assert plan_problems(dup) == ["queries must be distinct"]


def test_check_paper_generates_paper_queries_once(multi_settings):
    llm = FakeLLM()
    checker = _checker(multi_settings, _index_for_claim(), llm)
    claims = _claims("2504.15895")
    second = claims.claims[0].model_copy(update={"statement": "STMT two"})
    claims = claims.model_copy(update={"claims": [claims.claims[0], second]})
    out = checker.check_paper(claims, "lab", Target(), "an abstract")
    assert llm.query_calls == ["paper", "claim", "claim"]
    ids = [q.id for q in out.claims[0].queries]
    assert ids[:2] == ["b0", "b1"] and "c0" in ids and "p5" in ids and len(ids) == 14
    assert out.claims[0].query_usage.input_tokens == 100  # claim + paper call
    assert out.claims[1].query_usage.input_tokens == 50
    assert out.prompt_version == "rerank_v1+queries_v1"


def test_run_eval_multi_query_provenance(multi_settings, tmp_path):
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, title="Unrelated A", **older)],
            "GEN2 paper": [_p("2001.00007", 0.5, title="CLOSE component match", **older)],
            "GEN4 paper": [_p("2001.00007", 0.5, title="CLOSE component match", **older)],
        }
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(_row("s1", "2001.00007")) + "\n")
    summary = run_eval(
        "dev",
        cases_path=cases,
        settings=multi_settings,
        index=index,
        client=LLMClient(multi_settings, transport=FakeLLM()),
    )
    assert summary["union_recall"] == 1.0 and summary["recall_at_5"] == 1.0
    data = json.loads((multi_settings.private_reports_dir / "novelty_eval_dev.json").read_text())
    (res,) = data["results"]
    assert res["found_by"] == [["p2", "p4"]]
    assert set(data["query_groups"]) == {"paper:training_trick", "paper:dataset_construction"}
    assert all(row["only"] == 0 for row in data["query_groups"].values())  # two groups found it
    assert (
        "| query group |"
        in (multi_settings.private_reports_dir / "novelty_eval_dev.md").read_text()
    )


def test_replay_mode_serves_cache_only(settings):
    live_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        live_calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": "old"})
        return httpx.Response(200, json=[{"key": "a", "title": "t", "abstract": "x", "year": 2020}])

    _http_client(settings, handler).search("q", 5)  # fills the cache under version "old"
    replay = settings.model_copy(update={"index_replay_version": "old"})
    live_calls.clear()
    client = _http_client(replay, handler)
    assert client.version() == "old" and client.search("q", 5)[0].key == "a"
    with pytest.raises(IndexClientError, match="no cached response"):
        client.search("other", 5)
    assert live_calls == []


def test_eval_records_named_labels_and_diagnostics(settings, tmp_path):
    hits = [
        _p("2401.00001", 0.9, title="Unrelated A", year=2023),
        _p("2401.00002", 0.8, title="CLOSE cousin", year=2023),
    ]
    index = FakeIndex(search={"T1 A1": hits, "T1": hits})
    cases = tmp_path / "cases.jsonl"
    rows = [_row("s1", "2401.00001"), _row("s1", "2401.00002")]
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    run_eval(
        "dev",
        cases_path=cases,
        settings=settings,
        index=index,
        client=LLMClient(settings, transport=FakeLLM()),
    )
    data = json.loads((settings.private_reports_dir / "novelty_eval_dev.json").read_text())
    assert data["named_labels"] == {
        "same": 0,
        "subsumes": 0,
        "close": 1,
        "builds_on": 0,
        "different": 1,
    }
    nj = data["results"][0]["named_judgments"]
    assert [j["label"] for j in nj] == ["different", "close"] and nj[0][
        "abstract"
    ] == "abs 2401.00001"
    report = (settings.private_reports_dir / "novelty_eval_dev.md").read_text()
    assert "named works labelled `different`" in report and "because Unrelated A" in report


def test_listwise_orders_overlap_candidates_and_keeps_verdict_rule(settings):
    s2 = settings.model_copy(
        update={"novelty_prompt": "rerank_v2", "novelty_listwise": True, "novelty_listwise_cap": 3}
    )
    hits = [
        _p("2401.00001", 0.9, title="CLOSE A"),
        _p("2401.00002", 0.8, title="CLOSE B"),
        _p("2401.00003", 0.7, title="CLOSE FIRST C"),
        _p("2401.00004", 0.6, title="CLOSE D"),  # beyond the cap of 3
        _p("2401.00005", 0.95, title="Unrelated"),
    ]
    llm = FakeLLM()
    checker = _checker(s2, FakeIndex(search={"STMT": hits}), llm)
    result = checker.check_query(NoveltyQuery(statement="STMT"), Target(), 0)
    keys = [j.candidate.paper.key for j in result.ranked]
    # listwise over the top 3 close (A, B, FIRST C): C first, then B, A; D and the different
    # one keep label/similarity order after them
    assert keys == ["2401.00003", "2401.00002", "2401.00001", "2401.00004", "2401.00005"]
    assert [j.listwise_rank for j in result.ranked] == [1, 2, 3, None, None]
    assert result.ranked[0].listwise_reason == "cite 2"
    assert result.verdict == NoveltyVerdict.incremental and llm.listwise_calls == 1


def test_listwise_skipped_with_fewer_than_two_overlaps(settings):
    s2 = settings.model_copy(update={"novelty_listwise": True})
    llm = FakeLLM()
    hits = [_p("2401.00001", 0.9, title="CLOSE A"), _p("2401.00002", 0.8, title="Unrelated")]
    _checker(s2, FakeIndex(search={"STMT": hits}), llm).check_query(
        NoveltyQuery(statement="STMT"), Target(), 0
    )
    assert llm.listwise_calls == 0


def test_listwise_problems():
    items = [ListwiseItem(candidate_id=i, reason="r") for i in (0, 2, 2)]
    (problem,) = listwise_problems(ListwiseRanking(ranking=items), 3)
    assert "missing [1]" in problem and "[2]" in problem


def test_detail_report_has_full_provenance(multi_settings):
    _write_corpus(multi_settings)
    s2 = multi_settings.model_copy(update={"novelty_listwise": True, "novelty_prompt": "rerank_v2"})
    run_check(
        ["2504.15895"],
        settings=s2,
        index=_index_for_claim(),
        client=LLMClient(s2, transport=FakeLLM()),
        detail=True,
    )
    text = (s2.private_reports_dir / "novelty_detail_2504.15895.md").read_text()
    assert "| b0 | baseline | statement |" in text and "| p0 | paper |" in text
    assert "b0#1, b1#1" in text  # both baseline queries returned the top search hit
    assert "expansion via 2401.00001" in text and "cite " in text


def test_retrieval_only_eval_scores_pool_and_makes_no_llm_calls(multi_settings, tmp_path):
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, title="Unrelated A", **older)],
            "GEN2 paper": [_p("2001.00007", 0.5, title="Component match", **older)],
        }
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(_row("s1", "2001.00007")) + "\n")
    with pytest.raises(typer.Exit):  # generated queries are not cached: refuses to call the LLM
        run_eval("dev", cases_path=cases, settings=multi_settings, index=index, retrieval_only=True)
    # cache the queries with a normal run, then retrieval-only is served from the cache
    run_eval(
        "dev",
        cases_path=cases,
        settings=multi_settings,
        index=index,
        client=LLMClient(multi_settings, transport=FakeLLM()),
    )
    summary = run_eval(
        "dev", cases_path=cases, settings=multi_settings, index=index, retrieval_only=True
    )
    assert summary["pool_recall"] == 1.0 and summary["union_recall"] == 1.0
    logged = [
        json.loads(line)
        for line in (multi_settings.private_reports_dir / "novelty_eval_runs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert logged[-1]["retrieval_only"] is True and logged[-1]["llm_cost_usd_new"] == 0.0


def test_embed_model_is_sent_to_search_only(settings):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.url.path}?{request.url.query.decode()}")
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": "v"})
        return httpx.Response(200, json=[])

    large = settings.model_copy(update={"novelty_embed_model": "bge-large"})
    client = _http_client(large, handler)
    with pytest.raises(IndexClientError, match="ignores the model parameter"):
        client.search("q", 5)  # this fake server accepts any model

    def strict(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("model") == "__no_such_model__":
            return httpx.Response(422, json={"detail": "bad model"})
        return handler(request)

    seen.clear()
    client = _http_client(large, strict)
    client.search("q", 5)
    client.neighbors("k", "q")
    assert "model=bge-large" in seen[-2] and "model" not in seen[-1]
    _http_client(settings, handler).search("q", 5)
    assert "model" not in seen[-1]


# --------------------------------------------------------------------------- first stage


def _xc(key: str, seeds: list[str], cos: float, year: int = 2020, **kw) -> ExpandCandidate:
    return ExpandCandidate(
        key=key,
        title=kw.pop("title", f"expanded {key}"),
        abstract="abs",
        year=year,
        seed_count=len(seeds),
        seeds=[ExpandSeedLink(seed=sd, relation="cites") for sd in seeds],
        max_cosine=cos,
    )


def _stage_index() -> FakeIndex:
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "STMT": [_p("s1", 0.8, **older), _p("s2", 0.7, **older)],
            "GEN a": [_p("s2", 0.75, **older), _p("s3", 0.6, **older)],
        }
    )
    index._expand = {
        "x1": _xc("x1", ["s1", "s2"], 0.72),
        "s3": _xc("s3", ["s1"], 0.65),  # also a search hit
        "x2": _xc("x2", ["s2"], 0.5, year=2030),  # after the cutoff: guard drops it
    }
    return index


STAGE_QUERIES = [
    SearchQuery(id="b0", origin="baseline", kind="statement", text="STMT"),
    SearchQuery(id="p0", origin="paper", kind="technique", text="GEN a"),
]


def test_gather_candidates_features():
    index = _stage_index()
    params = RetrievalParams(search_k=50, query_k=30, expand_seeds=2, pool_size=10)
    cands, counts = gather_candidates(STAGE_QUERIES, index, Target(cutoff=date(2025, 1, 1)), params)
    by_key = {c.candidate.paper.key: c for c in cands}
    assert set(by_key) == {"s1", "s2", "s3", "x1"} and counts.expansion_new == 1
    assert ("expand", "s1,s2") in index.calls and index.expand_queries == ["GEN a"]
    s2, s3, x1 = by_key["s2"], by_key["s3"], by_key["x1"]
    assert (s2.best_rank, s2.max_cosine, s2.in_expansion) == (1, 0.75, False)
    assert (s3.in_search, s3.in_expansion, s3.seed_count, s3.max_cosine) == (True, True, 1, 0.65)
    assert (x1.best_rank, x1.seed_count, x1.in_search) == (None, 2, False)
    assert x1.candidate.via == ["s1", "s2"]
    assert by_key["s1"].max_cosine == 0.8  # baseline-only hit: falls back to its similarity
    assert s3.features()[5] == 1.0 and x1.features()[2] == 0.0


def test_rule_and_select_pool():
    index = _stage_index()
    params = RetrievalParams(search_k=50, query_k=30, expand_seeds=2, pool_size=2)
    pool, counts, unpooled = retrieve_first_stage(
        STAGE_QUERIES, index, Target(cutoff=date(2025, 1, 1)), params, rule_score
    )
    assert [c.paper.key for c in pool] == ["s1", "s2"] and counts.pool == 2
    assert {c.paper.key for c in unpooled} == {"s3", "x1"}


def test_logistic_learns_the_signal_and_cv_is_grouped():
    rng = np.random.default_rng(0)
    tables, labels = [], []
    for _ in range(10):
        x = rng.normal(size=(50, len(FEATURES)))
        y = np.zeros(50)
        y[0] = 1
        x[0, 1] += 8.0  # positives have a high seed-count feature
        tables.append(x)
        labels.append(y)
    model = fit_logistic(np.vstack(tables), np.concatenate(labels))
    assert np.argmax(np.abs(model.weights)) == 1 and model.weights[1] > 0
    scores = cross_validated_scores(tables, labels, folds=5)
    assert all(0 in np.argsort(-sc)[:3] for sc in scores)


def test_logistic_model_round_trip(tmp_path):
    model = LogisticModel(weights=[1.0] * 6, bias=0.5, mean=[0.0] * 6, std=[1.0] * 6)
    path = tmp_path / "m.json"
    path.write_text(model.to_json())
    assert LogisticModel.load(path) == model


def test_first_stage_eval_end_to_end(multi_settings, tmp_path):
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, **older)],
            "GEN0 paper": [_p("2001.00002", 0.6, **older)],
        }
    )
    index._expand = {"2001.00009": _xc("2001.00009", ["2001.00001"], 0.7)}
    cases = tmp_path / "dev.jsonl"
    rows = [_row(f"s{i}", "2001.00009") for i in range(5)]
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    settings = multi_settings.model_copy(update={"novelty_pool_size": 150})
    client = LLMClient(settings, transport=FakeLLM())
    summary = run_first_stage_eval(cases_path=cases, settings=settings, index=index, client=client)
    # named work is reached only via /expand: the current pool (no /neighbors data) misses it
    assert summary["current_pool_recall"] == 0.0 and summary["stage_union_recall"] == 1.0
    assert summary["rule_pool_recall"] == 1.0 and summary["learned_pool_recall_cv"] == 1.0
    assert settings.first_stage_model_file.exists()
    run = json.loads((settings.private_reports_dir / "novelty_first_stage_dev.json").read_text())
    assert run["n_named"] == 5 and run["type"] == "first_stage_eval"


def test_expand_request_never_sends_target(settings):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"index_version": "v"})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [], "total": 0})

    client = _http_client(settings, handler)
    client.expand(["a"], ["q"], date(2024, 1, 2), 100)
    client.expand(["a"], ["q"], date(2024, 1, 2), 100)  # cached
    assert bodies == [
        {"seeds": ["a"], "queries": ["q"], "max_candidates": 100, "before": "2024-01-02"}
    ]


def test_checker_uses_first_stage_setting(settings):
    s2 = settings.model_copy(update={"novelty_first_stage": "rule"})
    checker = _checker(s2, _stage_index(), FakeLLM())
    assert checker.first_stage is rule_score
    with pytest.raises(ValueError):
        _checker(
            settings.model_copy(update={"novelty_first_stage": "nope"}), FakeIndex(), FakeLLM()
        )


def test_cv_scorer_round_trip_and_binding(tmp_path):
    m = [
        LogisticModel(weights=[float(i)] * 6, bias=0.0, mean=[0.0] * 6, std=[1.0] * 6)
        for i in range(5)
    ]
    cv = CrossValidatedScorer(models=m, fold_of={"a": 3})
    path = tmp_path / "cv.json"
    path.write_text(cv.to_json())
    loaded = CrossValidatedScorer.load(path)
    assert loaded.for_case("a") == m[3]
    with pytest.raises(ValueError, match="not in the first-stage CV set"):
        loaded.for_case("zzz")


def test_first_stage_eval_pool_sizes_and_learned_cv_eval(multi_settings, tmp_path):
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, **older)],
            "GEN0 paper": [_p("2001.00002", 0.6, **older)],
        }
    )
    index._expand = {"2001.00009": _xc("2001.00009", ["2001.00001"], 0.7)}
    cases = tmp_path / "dev.jsonl"
    cases.write_text("\n".join(json.dumps(_row(f"s{i}", "2001.00009")) for i in range(5)) + "\n")
    settings = multi_settings.model_copy(update={"novelty_pool_size": 2})
    client = LLMClient(settings, transport=FakeLLM())
    run_first_stage_eval(
        cases_path=cases, settings=settings, index=index, client=client, pool_sizes=[3]
    )
    run = json.loads((settings.private_reports_dir / "novelty_first_stage_dev.json").read_text())
    assert set(run["by_pool_size"]) == {"2", "3"} and run["by_pool_size"]["3"]["rerank_calls"] == 1
    cv = CrossValidatedScorer.load(settings.first_stage_cv_file)
    assert sorted(cv.fold_of) == [f"s{i}" for i in range(5)]

    s2 = settings.model_copy(update={"novelty_first_stage": "learned_cv"})
    summary = run_eval("dev", cases_path=cases, settings=s2, index=index, client=client)
    assert summary["pool_recall"] == 1.0
    with pytest.raises(ValueError, match="dev eval cases"):
        _checker(s2, index, FakeLLM()).retrieve_query(NoveltyQuery(statement="T1 A1"), Target())


def test_learned_default_binds_fold_models_for_cv_dev_cases(settings):
    fold = [
        LogisticModel(weights=[float(i)] * 6, bias=0.0, mean=[0.0] * 6, std=[1.0] * 6)
        for i in range(5)
    ]
    full = LogisticModel(weights=[9.0] * 6, bias=0.0, mean=[0.0] * 6, std=[1.0] * 6)
    settings.first_stage_model_file.parent.mkdir(parents=True, exist_ok=True)
    settings.first_stage_model_file.write_text(full.to_json())
    settings.first_stage_cv_file.write_text(
        CrossValidatedScorer(models=fold, fold_of={"devcase": 2}).to_json()
    )
    learned = settings.model_copy(update={"novelty_first_stage": "learned"})
    checker = _checker(learned, FakeIndex(), FakeLLM())
    assert checker.first_stage == full  # lab papers and anything outside the CV set
    checker.bind_eval_case("devcase")
    assert checker.first_stage == fold[2]  # never the model fitted on this case
    checker.bind_eval_case("testcase")
    assert checker.first_stage == full
    cv_only = _checker(
        settings.model_copy(update={"novelty_first_stage": "learned_cv"}), FakeIndex(), FakeLLM()
    )
    with pytest.raises(ValueError, match="not in the first-stage CV set"):
        cv_only.bind_eval_case("testcase")


def test_negatives_verdicts_and_already_done_details(multi_settings, tmp_path):
    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, title="SAME idea earlier", **older)],
            "T2 A2": [_p("2001.00002", 0.9, title="Unrelated", **older)],
        }
    )
    rows = [
        {
            "submission_id": "n1",
            "venue": "ICLR.cc/2024/Conference",
            "submission_title": "T1",
            "submission_abstract": "A1",
            "cutoff_date": "2023-09-28",
            "tier": "poster",
        },
        {
            "submission_id": "n2",
            "venue": "ICLR.cc/2024/Conference",
            "submission_title": "T2",
            "submission_abstract": "A2",
            "cutoff_date": "2023-09-28",
        },
    ]
    path = tmp_path / "negatives_dev.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s2 = multi_settings.model_copy(update={"novelty_listwise": False})
    counts = run_negatives(
        cases_path=path, settings=s2, index=index, client=LLMClient(s2, transport=FakeLLM())
    )
    assert counts == {"novel": 1, "incremental": 0, "already_done": 1}
    report = (s2.private_reports_dir / "novelty_negatives_dev.md").read_text()
    assert "### 1. T1 (n1" in report and "SAME idea earlier" in report and "because SAME" in report
    with pytest.raises(typer.BadParameter):
        run_negatives(cases_path=tmp_path / "negatives_test.jsonl", settings=s2, index=index)


def test_arxiv_id_month_overrides_a_wrong_index_date():
    target = Target(title="Some submission title here", cutoff=date(2025, 2, 17))
    # the index dated 2503.17365 to 2025-02-01, but a March id cannot predate March 1
    assert is_future(_p("2503.17365", published_date=date(2025, 2, 1)), target)
    # same month as the cutoff: the id alone cannot tell, the date decides
    assert not is_future(_p("2502.00123", published_date=date(2025, 2, 3)), target)
    assert not is_future(_p("2501.00001", year=2025), target)


def test_severity_sets_verdict_and_rerank_v3_labels_builds_on(settings):
    s3 = settings.model_copy(update={"novelty_prompt": "rerank_v3", "novelty_severity": True})
    hits = [
        _p("2401.00001", 0.9, title="BASE framework we critique"),
        _p("2401.00002", 0.8, title="CLOSE component"),
        _p("2401.00003", 0.7, title="Unrelated"),
    ]
    llm = FakeLLM()
    result = _checker(s3, FakeIndex(search={"STMT": hits}), llm).check_query(
        NoveltyQuery(statement="STMT"), Target(), 0
    )
    labels = {j.candidate.paper.key: j.label for j in result.ranked}
    assert labels["2401.00001"] == NoveltyLabel.builds_on
    assert result.severity is not None and result.severity.score == 1
    assert (
        result.severity.drivers == ["2401.00002"] and result.verdict == NoveltyVerdict.incremental
    )
    assert [j.candidate.paper.key for j in result.ranked][:2] == ["2401.00002", "2401.00001"]
    assert llm.severity_calls == 1


def test_severity_problems_and_verdict_mapping():
    from reviewer.novelty.severity import severity_problems, severity_verdict

    ok = SeverityOutput(score=2, drivers=["a"], reason="r")
    assert severity_problems(ok, {"a", "b"}) == []
    assert severity_problems(ok.model_copy(update={"drivers": ["z"]}), {"a"})
    assert severity_problems(ok.model_copy(update={"score": 4}), {"a"})
    assert severity_problems(ok.model_copy(update={"drivers": []}), {"a"})
    assert severity_problems(SeverityOutput(score=0, drivers=["a"], reason="r"), {"a"})
    assert [severity_verdict(s) for s in range(4)] == [
        NoveltyVerdict.novel,
        NoveltyVerdict.incremental,
        NoveltyVerdict.incremental,
        NoveltyVerdict.already_done,
    ]


def test_severity_v2_requires_a_single_driver_for_score_2_or_3():
    """v1 kept allowing 1-2 drivers at any score >= 1 (reproducible); v2 requires exactly one
    driving work once the score reaches 2, since 'two components together cover the idea' is
    exactly the failure mode the eval turned up (combining known pieces is not itself already
    done)."""
    from reviewer.novelty.severity import severity_problems

    two_drivers_score_2 = SeverityOutput(score=2, drivers=["a", "b"], reason="together they...")
    assert severity_problems(two_drivers_score_2, {"a", "b"}, single_driver_for_overlap=False) == []
    problems = severity_problems(two_drivers_score_2, {"a", "b"}, single_driver_for_overlap=True)
    assert problems and "exactly one driving work" in problems[0]
    one_driver_score_3 = SeverityOutput(score=3, drivers=["a"], reason="r")
    assert severity_problems(one_driver_score_3, {"a"}, single_driver_for_overlap=True) == []
    two_drivers_score_1 = SeverityOutput(score=1, drivers=["a", "b"], reason="neither alone")
    assert severity_problems(two_drivers_score_1, {"a", "b"}, single_driver_for_overlap=True) == []


def test_score_severity_enforces_single_driver_only_for_v2(settings):
    """End to end: the same two-driver score-2 answer is accepted under severity_v1 and rejected
    (then retried) under severity_v2."""
    from reviewer.novelty.severity import score_severity

    class TwoDriverSeverity:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, req: TransportRequest) -> TransportResponse:
            self.calls += 1
            keys = re.findall(r'<candidate key="([^"]+)">', req.user)
            out = SeverityOutput(score=2, drivers=keys[:2], reason="A does X, B does Y together")
            return TransportResponse(
                parsed=out,
                usage=TokenUsage(input_tokens=10, output_tokens=1),
                stop_reason="end_turn",
            )

    ranked = [_judged(k, C, 0.9 - i * 0.01) for i, k in enumerate(["a1", "a2", "a3"])]
    fake_v1 = TwoDriverSeverity()
    result_v1, _ = score_severity(
        LLMClient(settings, transport=fake_v1),
        NoveltyQuery(statement="STMT"),
        ranked,
        prompt=load_rerank_prompt("severity_v1"),
        model="claude-opus-5-5",
        max_retries=1,
    )
    assert result_v1.score == 2 and fake_v1.calls == 1

    fake_v2 = TwoDriverSeverity()
    with pytest.raises(RerankError, match="exactly one driving work"):
        score_severity(
            LLMClient(settings, transport=fake_v2),
            NoveltyQuery(statement="STMT"),
            ranked,
            prompt=load_rerank_prompt("severity_v2"),
            model="claude-opus-5-5",
            max_retries=1,
        )
    assert fake_v2.calls == 2  # one retry, still rejected


def test_auc():
    from reviewer.novelty.evaluate import auc

    assert auc([3, 2], [0, 1]) == 1.0
    assert auc([1], [1]) == 0.5
    assert auc([0, 3], [1, 2]) == 0.5


def test_severity_eval_end_to_end(multi_settings, tmp_path):
    from reviewer.novelty.commands import run_severity_eval

    older = {"year": 2020}
    index = FakeIndex(
        search={
            "T1 A1": [_p("2001.00001", 0.9, title="SAME core idea", **older)],
            "T2 A2": [_p("2001.00002", 0.9, title="CLOSE component", **older)],
            "N1 B1": [_p("2001.00003", 0.9, title="SAME as negative", **older)],
            "N2 B2": [_p("2001.00004", 0.9, title="Unrelated", **older)],
        }
    )
    pos = tmp_path / "dev.jsonl"
    pos.write_text(
        json.dumps(_row("s1", "2001.00001", title="T1", abstract="A1"))
        + "\n"
        + json.dumps(_row("s2", "2001.00002", title="T2", abstract="A2"))
        + "\n"
    )
    neg = tmp_path / "negatives_dev.jsonl"
    neg.write_text(
        "\n".join(
            json.dumps(
                {
                    "submission_id": sid,
                    "venue": "ICLR.cc/2024/Conference",
                    "submission_title": t,
                    "submission_abstract": a,
                    "cutoff_date": "2023-09-28",
                }
            )
            for sid, t, a in (("n1", "N1", "B1"), ("n2", "N2", "B2"))
        )
        + "\n"
    )
    s = multi_settings.model_copy(update={"novelty_severity": True, "novelty_listwise": False})
    out = run_severity_eval(
        limit=2,
        settings=s,
        index=index,
        client=LLMClient(s, transport=FakeLLM()),
        positives_path=pos,
        negatives_path=neg,
    )
    # positives score 3 and 1, negatives 3 and 0 -> AUC = (0.5 + 1 + 0 + 1) / 4
    assert out["distributions"]["positives"] == {0: 0, 1: 1, 2: 0, 3: 1}
    assert out["distributions"]["negatives"] == {0: 1, 1: 0, 2: 0, 3: 1}
    assert out["auc"] == 0.625
    (n3,) = out["negatives_scoring_3"]
    assert n3["case_id"] == "n1" and "SAME as negative (2001.00003)" in n3["drivers"][0]
    report = (s.private_reports_dir / "novelty_severity_dev.md").read_text()
    assert "AUC of paper-level severity" in report and "N1" in report


# --------------------------------------------------------------------------- own-preprint leakage


def test_own_preprint_threshold_is_085():
    assert OWN_PREPRINT_JACCARD == 0.85


def test_text_jaccard():
    assert text_jaccard(
        "Learning Diverse Quadruped Locomotion Gaits via Reward Machines",
        "Learning Quadruped Locomotion Policies using Logical Rules",
    ) == pytest.approx(0.25, abs=0.01)
    assert text_jaccard("", "anything") == 0.0
    assert text_jaccard("same words here", "same words here") == 1.0


CODEPLAN_SUB_TITLE = "CodePlan: Unlocking Reasoning Potential in LLMs by Scaling Code-form Planning"
CODEPLAN_ARXIV_TITLE = "Unlocking Reasoning Potential in LLMs by Scaling Code-form Planning"
CODEPLAN_ABSTRACT = (
    "Despite the remarkable success of large language models on traditional tasks their "
    "planning ability remains a critical bottleneck in complex reasoning"
)
QUADRUPED_SUB_TITLE = "Learning Diverse Quadruped Locomotion Gaits via Reward Machines"
QUADRUPED_ARXIV_TITLE = "Learning Quadruped Locomotion Policies using Logical Rules"
QUADRUPED_ABSTRACT = (
    "Quadruped animals are capable of exhibiting a diverse range of locomotion gaits while "
    "progress has been made in demonstrating such gaits on robots"
)
DOMAINSTUDIO_SUB_TITLE = "DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Generation"
DOMAINSTUDIO_ARXIV_TITLE = "Few-shot Image Generation with Diffusion Models"
DOMAINSTUDIO_SUB_ABSTRACT = (
    "Denoising diffusion probabilistic models have been proven capable of synthesizing high "
    "quality images with remarkable diversity when trained on large amounts of data"
)
DOMAINSTUDIO_ARXIV_ABSTRACT = (
    "Denoising diffusion probabilistic models have been proven capable of synthesizing high "
    "quality images however few-shot generation has not been studied with DDPM approaches"
)
SYSID_SUB_TITLE = "Exact Recovery for Parameterized Nonlinear System ID under Adversarial Attacks"
SYSID_ARXIV_TITLE = "Exact Recovery for Parameterized Nonlinear System ID under Sparse Disturbances"
SYSID_SUB_ABSTRACT = (
    "In this work we study the system identification problem for parameterized nonlinear "
    "systems using basis functions under adversarial attacks motivated by security concerns"
)
SYSID_ARXIV_ABSTRACT = (
    "In this work we study the problem of learning a nonlinear dynamical system by "
    "parameterizing its dynamics using basis functions assuming disturbances occur with "
    "arbitrary probability"
)


@pytest.mark.parametrize(
    ("sub_title", "sub_abstract", "cand_title", "cand_abstract", "expect_own"),
    [
        # Renamed for the submission ("CodePlan:" prefix), near-identical abstract.
        (CODEPLAN_SUB_TITLE, CODEPLAN_ABSTRACT, CODEPLAN_ARXIV_TITLE, CODEPLAN_ABSTRACT, True),
        # Retitled for camera-ready; title alone would miss it, abstract near-identical.
        (QUADRUPED_SUB_TITLE, QUADRUPED_ABSTRACT, QUADRUPED_ARXIV_TITLE, QUADRUPED_ABSTRACT, True),
        # A genuinely different, merely topically-close prior work.
        (
            DOMAINSTUDIO_SUB_TITLE,
            DOMAINSTUDIO_SUB_ABSTRACT,
            DOMAINSTUDIO_ARXIV_TITLE,
            DOMAINSTUDIO_ARXIV_ABSTRACT,
            False,
        ),
        # Same opening sentence, but the actual problem studied diverges.
        (SYSID_SUB_TITLE, SYSID_SUB_ABSTRACT, SYSID_ARXIV_TITLE, SYSID_ARXIV_ABSTRACT, False),
    ],
)
def test_is_own_preprint_matches_the_calibration_cases(
    sub_title, sub_abstract, cand_title, cand_abstract, expect_own
):
    """Thresholds picked from 5 real dev cases where severity=3 turned out to be the submission's
    own earlier preprint under a different title, vs. 2 genuinely different close prior works."""
    target = Target(title=sub_title, abstract=sub_abstract)
    cand = IndexPaper(key="2001.00001", title=cand_title, abstract=cand_abstract, year=2020)
    assert is_own_preprint(cand, target) is expect_own


def test_is_own_preprint_requires_not_future():
    target = Target(title="X" * 20, abstract="Y" * 20, cutoff=date(2024, 1, 1))
    later = IndexPaper(
        key="2406.00001", title="X" * 20, abstract="Y" * 20, published_date=date(2024, 6, 1)
    )
    assert not is_own_preprint(later, target)  # identical text, but it postdates the target


def test_admit_filters_own_preprint_before_future_and_counts_it():
    target = Target(title=QUADRUPED_SUB_TITLE, abstract=QUADRUPED_ABSTRACT, cutoff=date(2025, 1, 1))
    own_preprint = IndexPaper(
        key="2107.10969",
        title=QUADRUPED_ARXIV_TITLE,
        abstract=QUADRUPED_ABSTRACT,
        year=2021,
        similarity=0.5,
    )
    unrelated = _p("2401.00001", title="Unrelated work", year=2024)
    pool, counts = retrieve(
        NoveltyQuery(statement="STMT"),
        FakeIndex(search={"STMT": [own_preprint, unrelated]}),
        target,
        PARAMS,
    )
    assert [c.paper.key for c in pool] == ["2401.00001"]
    assert counts.filtered_own_preprint == 1


def test_target_for_threads_abstract_through_v1_date_resolution():
    target = target_for("2504.15895", "t", cutoff=True, abstract="the abstract")
    assert target.abstract == "the abstract"
    resolved = resolve_v1_date(FakeIndex(), target)
    assert resolved.abstract == "the abstract"  # replace() must not drop it


# --------------------------------------------------------------------------- builds_on + listwise


def test_builds_on_is_ranked_by_listwise():
    assert NoveltyLabel.builds_on in OVERLAP_LABELS
    assert NoveltyLabel.different not in OVERLAP_LABELS
