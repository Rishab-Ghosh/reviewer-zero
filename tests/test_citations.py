"""Citation checker: raw-string parsing, resolution acceptance, flags, support judging, routing.
No network: sources are exercised through their pure record parsers, the LLM through a fake."""

from datetime import datetime

import pytest

from reviewer.citations import raw
from reviewer.citations.checks import (
    arxiv_published_flag,
    author_flag,
    citing_date,
    duplicate_flags,
    not_checked_reason,
    run_checks,
    unresolved_flag,
    year_flag,
)
from reviewer.citations.commands import is_private, report_dir
from reviewer.citations.resolve import Resolution, accept, hints_for
from reviewer.citations.sources import Record, crossref_record, openalex_records
from reviewer.citations.support import apply_overrides, judge_sentence, span_in_text
from reviewer.claims.prompts import Prompt
from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.models import (
    FlagType,
    InTextCitation,
    ParsedPaper,
    PropertyType,
    Reference,
    ReferenceCheck,
    ResolvedWork,
    SupportBatch,
    SupportJudgment,
    SupportVerdict,
    TokenUsage,
    WorkVersion,
)

LAYERNORM = (
    "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. "
    "arXiv preprint arXiv:1607.06450, 2016."
)
GANIN = (
    'Y. Ganin, E. Ustinova, and V. Lempitsky, "Domain-adversarial training of neural networks," '
    "The journal of machine learning research, vol. 17, no. 1, pp. 2096-2030, 2016."
)


def _j(
    verdict: SupportVerdict,
    abstract_span: str = "",
    *,
    usable: bool = True,
    prop: PropertyType = PropertyType.none,
    sentence_span: str = "LN helps",
    ref_id: str = "b1",
) -> SupportJudgment:
    return SupportJudgment(
        ref_id=ref_id,
        attributed="a",
        abstract_usable=usable,
        property_type=prop,
        sentence_span=sentence_span,
        abstract_span=abstract_span,
        explanation="e",
        verdict=verdict,
    )


# --------------------------------------------------------------------------- raw strings


def test_cited_years_ignores_page_ranges_and_arxiv_ids() -> None:
    assert raw.cited_years(GANIN) == [2016]
    assert raw.cited_years("Foo. arXiv:2006.11239, 2020.") == [2020]
    assert raw.cited_years("In NeurIPS 2019, 32.") == [2019]


def test_arxiv_ids_rejoins_line_wrapped_id() -> None:
    assert raw.arxiv_ids("URL https://arxiv.org/abs/2410. 04707. Next") == ["2410.04707"]
    assert raw.arxiv_ids("CoRR, abs/1703.03906, 2017") == ["1703.03906"]


@pytest.mark.parametrize(
    ("ref", "title"),
    [
        (LAYERNORM, "Layer normalization"),
        (GANIN, "Domain-adversarial training of neural networks"),
        (
            "Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In NIPS, 2016.",
            "Can active memory replace attention?",
        ),
        ("A. B. Smith and C. D. Jones. Deep nets. In ICML, 2019.", "Deep nets"),
    ],
)
def test_raw_title(ref: str, title: str) -> None:
    assert raw.raw_title(ref) == title


def test_arxiv_only_and_web_resource() -> None:
    assert raw.cites_arxiv_only(LAYERNORM)
    assert not raw.cites_arxiv_only("X. Title. In Proceedings of ACL, 2020. arXiv:2001.00001")
    assert raw.is_web_resource("OpenAI. Learning to reason. https://openai.com/index/x, 2024.")
    assert not raw.is_web_resource(LAYERNORM)


def test_author_chunks_and_et_al() -> None:
    names, et_al = raw.author_chunks(LAYERNORM, "Layer normalization")
    assert names == ["jimmy lei ba", "jamie ryan kiros", "geoffrey e hinton"]
    assert not et_al
    names, et_al = raw.author_chunks(
        "Daya Guo, Dejian Yang, et al. Deepseek-r1. 2025.", "Deepseek-r1"
    )
    assert names == ["daya guo", "dejian yang"] and et_al


# --------------------------------------------------------------------------- acceptance


def _rec(title: str, authors: list[str], year: int, source: str = "openalex") -> Record:
    return Record(
        WorkVersion(
            source=source,  # type: ignore[arg-type]
            work_id=f"W{abs(hash(title)) % 10**6}",
            title=title,
            year=year,
            is_preprint=True,
            authors=authors,
        )
    )


def _ref(raw_string: str, title: str | None = None, **kw: object) -> Reference:
    return Reference(ref_id="b0", raw_string=raw_string, title=title, **kw)  # type: ignore[arg-type]


def test_accept_rejects_fragment_title() -> None:
    ref = _ref(
        "A. Author. Attention is all you need in speech separation. In ICASSP, 2021.",
        title="Attention is all you need in speech separation",
    )
    frag = _rec("Attention Is All You Need", ["Ashish Vaswani"], 2017)
    assert accept(ref, hints_for(ref), frag) is None
    full = _rec("Attention is All You Need in Speech Separation", ["A. Author"], 2021)
    assert accept(ref, hints_for(ref), full) == pytest.approx(1.0)


def test_accept_ignores_consolidated_title_with_author_junk() -> None:
    ref = _ref(
        "Ian Wu, Sina Pakazad, and Graham Neubig. Better instruction-following through minimum "
        "bayes risk. arXiv preprint arXiv:2410.02902, 2024.",
        title="Sina Pakazad, and Graham Neubig. Better instruction-following through "
        "minimum bayes risk",
    )
    rec = _rec("Better Instruction-Following Through Minimum Bayes Risk", ["Ian Wu"], 2024)
    assert accept(ref, hints_for(ref), rec) is not None


def test_accept_truncated_title_needs_author_and_year() -> None:
    raw_s = "Ankur Parikh and Jakob Uszkoreit. A decomposable attention model. In EMNLP, 2016."
    ref = _ref(raw_s, title="A decomposable attention model")
    good = _rec(
        "A Decomposable Attention Model for Natural Language Inference", ["Ankur Parikh"], 2016
    )
    wrong_year = _rec(
        "A Decomposable Attention Model for Natural Language Inference", ["Ankur Parikh"], 2020
    )
    assert accept(ref, hints_for(ref), good) is not None
    assert accept(ref, hints_for(ref), wrong_year) is None


def test_printed_id_accepts_renamed_paper_with_author_agreement() -> None:
    ref = _ref(
        "Pei Ke, Bosi Wen, Zhuoer Feng. Critiquellm: Scaling llm-as-critic. "
        "corr, abs/2311.18702, 2023.",
        title="Critiquellm: Scaling llm-as-critic",
    )
    res = Resolution(ref=ref, hints=hints_for(ref))
    renamed = _rec(
        "CritiqueLLM: Towards an Informative Critique Generation Model",
        ["Pei Ke", "Bosi Wen"],
        2023,
    )
    stranger = _rec("Something else entirely", ["Nobody Known"], 2023)
    assert res.add([renamed, stranger], printed_id=True) == 0
    res.accept_by_printed_id()
    assert [r.version.title for r, _ in res.records] == [renamed.version.title]
    assert res.source == "openalex"


# --------------------------------------------------------------------------- source parsers


def test_openalex_merged_work_yields_venue_and_preprint() -> None:
    work = {
        "id": "https://openalex.org/W1",
        "title": "Long Short-Term Memory-Networks for Machine Reading",
        "publication_year": 2016,
        "type": "conference-paper",
        "doi": "https://doi.org/10.18653/v1/d16-1053",
        "authorships": [{"author": {"display_name": "Jianpeng Cheng"}}],
        "locations": [{"landing_page_url": "http://arxiv.org/abs/1601.06733v2", "source": None}],
    }
    recs = openalex_records(work)
    assert [(r.version.is_preprint, r.arxiv_id, r.doi) for r in recs] == [
        (False, "1601.06733", "10.18653/v1/d16-1053"),
        (True, "1601.06733", "10.18653/v1/d16-1053"),
    ]


def test_preprint_server_doi_is_not_a_venue() -> None:
    work = {
        "id": "W2",
        "title": "T",
        "type": "article",
        "doi": "https://doi.org/10.21203/rs.3.rs-1/v1",
    }
    assert all(r.version.is_preprint for r in openalex_records(work))
    item = {"DOI": "10.2139/ssrn.1", "title": ["T"], "type": "journal-article"}
    assert crossref_record(item).version.is_preprint


# --------------------------------------------------------------------------- flags


def _resolved(
    versions: list[WorkVersion], authors: list[str], title: str = "Layer Normalization"
) -> ResolvedWork:
    return ResolvedWork(
        source="openalex", title=title, authors=authors, versions=versions, title_similarity=1.0
    )


def _v(year: int, preprint: bool, venue: str = "V", date: str | None = None) -> WorkVersion:
    return WorkVersion(
        source="openalex",
        work_id=f"W{year}{preprint}",
        title="T",
        year=year,
        publication_date=date,
        venue=venue,
        is_preprint=preprint,
    )


def _check(
    raw_string: str, resolved: ResolvedWork | None, attempts: list[str] | None = None
) -> ReferenceCheck:
    return ReferenceCheck(
        ref_id="b1",
        raw_string=raw_string,
        cited_years=raw.cited_years(raw_string),
        cited_as_arxiv_only=raw.cites_arxiv_only(raw_string),
        resolved=resolved,
        attempts=attempts or [],
    )


def _av(year: int) -> WorkVersion:
    return WorkVersion(
        source="arxiv", work_id="1607.06450", title="T", year=year, venue="arXiv", is_preprint=True
    )


def test_year_flag_rules() -> None:
    a = ["Jimmy Ba"]
    # arXiv-only citation: must fall in the arXiv span
    assert year_flag(_check(LAYERNORM.replace("2016", "2019"), _resolved([_av(2016)], a)))
    assert year_flag(_check(LAYERNORM, _resolved([_av(2016)], a))) is None
    # venue citation, venue held: must be that venue's year (or the arXiv span)
    tot = "S. Yao. Tree of thoughts. Advances in Neural Information Processing Systems, 36, 2024."
    held = [_av(2023), _v(2023, False, "Advances in Neural Information Processing Systems 36")]
    flag = year_flag(_check(tot, _resolved(held, a)))
    assert flag is not None and "Neural Information Processing" in flag.detail
    assert year_flag(_check(tot.replace("2024", "2023"), _resolved(held, a))) is None
    # venue citation whose venue we do not hold: never flagged (no fallback rule)
    gan = "I. Goodfellow. Generative adversarial nets. In Advances in NeurIPS, 2014."
    cacm = _v(2020, False, "Communications of the ACM")
    assert year_flag(_check(gan, _resolved([cacm], a))) is None
    wiles = "O. Wiles. A fine-grained analysis on distribution shift. In ICLR, 2022."
    assert year_flag(_check(wiles.replace("2022", "2019"), _resolved([_av(2021)], a))) is None


def test_author_flag_first_author_and_short_record() -> None:
    ok = _check(
        LAYERNORM,
        _resolved([_v(2016, True)], ["Jimmy Lei Ba", "Jamie Ryan Kiros", "Geoffrey E. Hinton"]),
    )
    assert author_flag(ok) is None
    bad = _check(
        LAYERNORM,
        _resolved([_v(2016, True)], ["Someone Else", "Jamie Ryan Kiros", "Geoffrey Hinton"]),
    )
    flag = author_flag(bad)
    assert flag is not None and "first author" in flag.detail
    # Record lists only the first author (incomplete metadata): not flagged.
    short = _check(LAYERNORM, _resolved([_v(2016, True)], ["Jimmy Lei Ba"]))
    assert author_flag(short) is None


def test_arxiv_published_flag_ignores_workshops_and_reports_timing() -> None:
    ws = _check(
        LAYERNORM,
        _resolved([_v(2016, True), _v(2016, False, "NeurIPS Deep Learning Workshop")], []),
    )
    assert arxiv_published_flag(ws, citing_date("1706.03762")) is None
    pub = _check(LAYERNORM, _resolved([_v(2016, True), _v(2016, False, "EMNLP", "2016-11-01")], []))
    flag = arxiv_published_flag(pub, citing_date("1706.03762"))
    assert flag is not None and "predates" in flag.detail


def test_unresolved_not_flagged_when_search_was_skipped_or_web_page() -> None:
    skipped = _check(
        LAYERNORM, None, ["openalex title.search -> skipped (OpenAlex budget floor reached)"]
    )
    assert unresolved_flag(skipped) is None and not_checked_reason(skipped)
    web = _check("OpenAI. Learning to reason. https://openai.com/x, 2024.", None)
    assert unresolved_flag(web) is None
    assert (
        unresolved_flag(_check(LAYERNORM, None, ["openalex arxiv:1607.06450 -> 0 record(s)"]))
        is not None
    )


def test_duplicates_and_run_checks() -> None:
    r = _resolved([_v(2016, True)], ["Jimmy Lei Ba"]).model_copy(update={"arxiv_id": "1607.06450"})
    a = _check(LAYERNORM, r).model_copy(update={"ref_id": "b1"})
    b = _check(LAYERNORM, r).model_copy(update={"ref_id": "b7"})
    flags = duplicate_flags([a, b])
    assert [(f.ref_id, f.type) for f in flags] == [("b7", FlagType.duplicate)]
    run_checks([a, b], "1706.03762")
    assert [f.type for f in b.flags] == [FlagType.duplicate]


# --------------------------------------------------------------------------- support


def test_span_in_text_matches_words_in_order() -> None:
    abstract = "We propose “layer  normalization” — a simple method."
    assert span_in_text('"layer normalization" - a simple method', abstract)
    assert span_in_text("Layer normalization, a simple method", abstract)
    assert not span_in_text("a simple layer normalization", abstract)
    assert not span_in_text("batch normalization", abstract)
    assert not span_in_text("", abstract)


ABS = "Here we use x for y."
SENT = "Prior work uses z for y [1]."


@pytest.mark.parametrize(
    ("judgment", "expected", "reason_part"),
    [
        (_j(SupportVerdict.cant_tell_from_abstract), SupportVerdict.not_in_abstract, "usable"),
        (
            _j(SupportVerdict.supported, "we use x", usable=False),
            SupportVerdict.cant_tell_from_abstract,
            "unusable",
        ),
        (
            _j(SupportVerdict.cant_tell_from_abstract, usable=False),
            SupportVerdict.cant_tell_from_abstract,
            None,
        ),
        (
            _j(SupportVerdict.supported, "not there"),
            SupportVerdict.cant_tell_from_abstract,
            "abstract_span",
        ),
        (_j(SupportVerdict.not_in_abstract), SupportVerdict.not_in_abstract, None),
        # contradicted stands only with a property type and both spans verbatim
        (
            _j(
                SupportVerdict.contradicted,
                "we use x",
                prop=PropertyType.method_type,
                sentence_span="uses z for y",
            ),
            SupportVerdict.contradicted,
            None,
        ),
        (
            _j(SupportVerdict.contradicted, "we use x", sentence_span="uses z for y"),
            SupportVerdict.not_in_abstract,
            "property_type is none",
        ),
        (
            _j(
                SupportVerdict.contradicted,
                "we use x",
                prop=PropertyType.method_type,
                sentence_span="uses w",
            ),
            SupportVerdict.not_in_abstract,
            "sentence_span",
        ),
        (
            _j(
                SupportVerdict.contradicted,
                "we avoid x",
                prop=PropertyType.method_type,
                sentence_span="uses z for y",
            ),
            SupportVerdict.not_in_abstract,
            "abstract_span",
        ),
    ],
)
def test_apply_overrides(
    judgment: SupportJudgment, expected: SupportVerdict, reason_part: str | None
) -> None:
    final, reason = apply_overrides(judgment, SENT, ABS)
    assert final == expected
    if reason_part is None:
        assert reason is None
    else:
        assert reason is not None and reason_part in reason


class FakeTransport:
    """Returns `batch` for the first judge; `verify` (if given) for verification calls, which
    are recognized by their separate cache entry, i.e. only by call order here."""

    def __init__(self, batch: SupportBatch, verify: SupportBatch | Exception | None = None) -> None:
        self.batch = batch
        self.verify = verify
        self.calls: list[TransportRequest] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls.append(req)
        out = self.batch if len(self.calls) == 1 or self.verify is None else self.verify
        if isinstance(out, Exception):
            raise out
        return TransportResponse(out, TokenUsage(input_tokens=100, output_tokens=10), "end_turn")


def _support_setup(tmp_path, batch: SupportBatch, verify: SupportBatch | Exception | None = None):
    settings = Settings(_env_file=None, data_dir=tmp_path, anthropic_api_key=None)
    transport = FakeTransport(batch, verify)
    client = LLMClient(settings, transport=transport)
    with_abs = _check(
        LAYERNORM,
        _resolved([_v(2016, True)], []).model_copy(
            update={"abstract": "We transpose batch normalization into layer normalization."}
        ),
    ).model_copy(update={"ref_id": "b1"})
    no_abs = _check(LAYERNORM, _resolved([_v(2016, True)], [])).model_copy(update={"ref_id": "b2"})
    checks = {"b1": with_abs, "b2": no_abs}
    cit = InTextCitation(
        section_heading="Intro", sentence="LN helps [1, 2, 9].", ref_ids=["b1", "b2", "b9"]
    )
    return client, transport, checks, cit


def test_judge_skips_no_abstract_and_drops_unverified_contradiction(
    tmp_path,
) -> None:
    batch = SupportBatch(
        judgments=[
            _j(SupportVerdict.contradicted, "LN hurts training", prop=PropertyType.result_direction)
        ]
    )
    client, transport, checks, cit = _support_setup(tmp_path, batch)
    out = judge_sentence(
        cit, checks, "Paper", client, Prompt(version="support_vT", text="sys"), "claude-sonnet-5"
    )
    by_ref = {j.ref_id: j for j in out.judgments}
    assert set(by_ref) == {"b1", "b2"}  # b9 is unknown -> skipped
    assert by_ref["b2"].verdict == SupportVerdict.cant_tell_from_abstract
    assert not by_ref["b2"].judged_by_llm
    assert by_ref["b1"].verdict == SupportVerdict.not_in_abstract
    assert by_ref["b1"].overridden_from == SupportVerdict.contradicted
    assert "abstract_span" in (by_ref["b1"].override_reason or "")
    assert len(transport.calls) == 1 and "b2" not in transport.calls[0].user


def test_judge_keeps_verbatim_supported(tmp_path) -> None:
    batch = SupportBatch(
        judgments=[
            _j(SupportVerdict.supported, "transpose batch normalization into layer normalization")
        ]
    )
    client, _, checks, cit = _support_setup(tmp_path, batch)
    out = judge_sentence(
        cit, checks, "Paper", client, Prompt(version="support_vT", text="sys"), "claude-sonnet-5"
    )
    b1 = next(j for j in out.judgments if j.ref_id == "b1")
    assert b1.verdict == SupportVerdict.supported and b1.overridden_from is None


# --------------------------------------------------------------------------- privacy routing


def _paper(paper_id: str, source_path: str) -> ParsedPaper:
    return ParsedPaper(paper_id=paper_id, source_path=source_path, title="T")


def test_private_papers_route_to_private_reports(tmp_path) -> None:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports"
    )
    own = _paper("own-demo-deadbeef", "data/papers/own/demo.pdf")
    pub = _paper("1706.03762", "data/papers/arxiv/1706.03762.pdf")
    openreview = _paper("or-AbC123", "data/openreview/pdf/AbC123.pdf")
    assert is_private(own) and not is_private(pub)
    assert is_private(openreview)
    assert report_dir(settings, True) == tmp_path / "reports" / "private" / "citations"
    assert report_dir(settings, False) == tmp_path / "reports" / "citations"
    assert citing_date("1706.03762") == datetime(2017, 6, 1).date()
    assert citing_date("own-demo-deadbeef") is None


def test_timing_treats_openalex_january_first_as_year_only() -> None:
    cited_on = citing_date("1706.03762")
    same_year = _check(
        LAYERNORM, _resolved([_v(2016, True), _v(2017, False, "EMNLP", "2017-01-01")], [])
    )
    flag = arxiv_published_flag(same_year, cited_on)
    assert flag is not None and "same year" in flag.detail
    later = _check(
        LAYERNORM, _resolved([_v(2016, True), _v(2017, False, "EMNLP", "2017-09-07")], [])
    )
    flag = arxiv_published_flag(later, cited_on)
    assert flag is not None and "appeared after" in flag.detail


ABSTRACT_QUOTE = "transpose batch normalization into layer normalization"


def _one(verdict: SupportVerdict, quote: str = ABSTRACT_QUOTE) -> SupportBatch:
    return SupportBatch(judgments=[_j(verdict, quote, prop=PropertyType.method_type)])


def _judge(tmp_path, first: SupportBatch, verify: SupportBatch | Exception):
    client, transport, checks, cit = _support_setup(tmp_path, first, verify)
    prompt = Prompt(version="support_vT", text="sys")
    out = judge_sentence(cit, checks, "Paper", client, prompt, "claude-sonnet-5")
    return out, transport, next(j for j in out.judgments if j.ref_id == "b1")


def test_contradicted_confirmed_by_independent_rejudgment(tmp_path) -> None:
    _, transport, b1 = _judge(
        tmp_path, _one(SupportVerdict.contradicted), _one(SupportVerdict.contradicted)
    )
    assert b1.verdict == SupportVerdict.contradicted
    assert b1.verifier_verdict == SupportVerdict.contradicted and b1.overridden_from is None
    # same inputs, never the first verdict
    assert len(transport.calls) == 2
    assert transport.calls[0].user == transport.calls[1].user
    assert transport.calls[0].system == transport.calls[1].system
    assert "contradicted" not in transport.calls[1].user


def test_contradicted_not_confirmed_becomes_not_in_abstract(tmp_path) -> None:
    _, _, b1 = _judge(
        tmp_path, _one(SupportVerdict.contradicted), _one(SupportVerdict.not_in_abstract, "")
    )
    assert b1.verdict == SupportVerdict.not_in_abstract
    assert b1.overridden_from == SupportVerdict.contradicted
    assert b1.verifier_verdict == SupportVerdict.not_in_abstract
    assert "not confirmed" in (b1.override_reason or "")


def test_verifier_failure_downgrades_and_records_error(tmp_path) -> None:
    from reviewer.llm.client import LLMAPIError

    out, _, b1 = _judge(tmp_path, _one(SupportVerdict.contradicted), LLMAPIError("boom"))
    assert b1.verdict == SupportVerdict.not_in_abstract
    assert "call failed" in (b1.override_reason or "")
    assert out.error and "verifier" in out.error


def test_no_verification_call_without_contradiction(tmp_path) -> None:
    _, transport, b1 = _judge(
        tmp_path, _one(SupportVerdict.supported), _one(SupportVerdict.contradicted)
    )
    assert b1.verdict == SupportVerdict.supported and len(transport.calls) == 1


# --------------------------------------------------------------------------- reviewer-index


class _Resp:
    def __init__(self, status: int, body: object) -> None:
        self.status, self.body, self.headers, self.from_cache = status, body, {}, False


class _FakeHttp:
    def __init__(self, routes: dict[str, _Resp]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self, url: str, params: dict[str, str] | None = None, *, use_cache: bool = True
    ) -> _Resp:
        self.calls.append((url, params or {}))
        return self.routes[url.rsplit("/", 1)[1]]


def test_index_client_uses_q_only_and_parses_bare_list(tmp_path) -> None:
    from reviewer.citations.sources import IndexClient

    hit = {
        "key": "1607.06450",
        "kind": "arxiv",
        "title": "Layer Normalization",
        "abstract": "A",
        "published_date": "2016-07-21",
        "year": 2016,
        "similarity": 0.9,
    }
    http = _FakeHttp({"health": _Resp(200, {"index_version": "abc"}), "search": _Resp(200, [hit])})
    client = IndexClient(Settings(_env_file=None, data_dir=tmp_path), http=http)  # type: ignore[arg-type]
    assert client.available and client.index_version == "abc"
    assert client.search("layer normalization", k=3) == [hit]
    assert http.calls[-1][1] == {"q": "layer normalization", "k": "3"}


def test_index_client_down_without_index_version(tmp_path) -> None:
    from reviewer.citations.sources import IndexClient

    http = _FakeHttp({"health": _Resp(200, {"status": "ok"})})
    client = IndexClient(Settings(_env_file=None, data_dir=tmp_path), http=http)  # type: ignore[arg-type]
    assert not client.available and client.search("x") == []


def test_index_external_hit_is_not_venue_evidence() -> None:
    from reviewer.citations.resolve import Resolver

    class Index:
        available = True

        def search(self, q: str, k: int = 5) -> list[dict[str, object]]:
            return [
                {
                    "key": "openalex:W2952191002",
                    "kind": "external",
                    "year": 2016,
                    "title": "Layer Normalization",
                    "abstract": "We transpose batch norm.",
                }
            ]

    class OpenAlex:
        def work(self, key: str) -> list[Record]:
            return []

    ref = _ref(LAYERNORM, title="Layer normalization")
    res = Resolution(ref=ref, hints=hints_for(ref))
    Resolver(Index(), OpenAlex(), None, None)._index(res)  # type: ignore[arg-type]
    assert [r.version.is_preprint for r, _ in res.records] == [True]
    assert res.records[0][0].abstract == "We transpose batch norm."


def test_summary_samples_only_flags_and_contradicted() -> None:
    from reviewer.citations.summary import render_summary
    from reviewer.models import CitationFlag, CitationReport, CitationSupport

    def sup(verdict: SupportVerdict) -> CitationSupport:
        return CitationSupport(
            section_heading="S",
            sentence=f"{verdict.value} sentence",
            ref_id="b1",
            verdict=verdict,
            judged_by_llm=True,
        )

    ref = _check(LAYERNORM, None).model_copy(
        update={
            "flags": [
                CitationFlag(type=FlagType.unresolved, ref_id="b1", detail="d", evidence=["raw: x"])
            ]
        }
    )
    rep = CitationReport(
        paper_id="1706.03762",
        title="T",
        private=False,
        generated=datetime(2026, 9, 25),
        index_available=True,
        model="m",
        prompt_version="support_v4",
        references=[ref],
        support=[
            sup(SupportVerdict.contradicted),
            sup(SupportVerdict.not_in_abstract),
            sup(SupportVerdict.supported),
        ],
    )
    text = render_summary([rep], seed=7, n_sample=10, n_not_in=4)
    flagged = text.split("random flagged items")[1].split("not_in_abstract items")[0]
    assert "Unresolved references" in flagged and "contradicted sentence" in flagged
    assert "supported sentence" not in text
    assert "not_in_abstract sentence" in text.split("not_in_abstract items")[1]


@pytest.mark.parametrize(
    ("ref", "title"),
    [
        (
            "Tom Brown and Nick Ryder. 2020. Language models are few-shot learners. NeurIPS.",
            "Language models are few-shot learners",
        ),
        (
            "Zhu, C., Han, S., Dally, W.J.: Trained ternary quantization (2016)",
            "Trained ternary quantization",
        ),
        (
            "Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features (2009)",
            "Learning multiple layers of features",
        ),
    ],
)
def test_raw_title_author_year_and_lncs(ref: str, title: str) -> None:
    assert raw.raw_title(ref) == title


def test_parse_artifacts_and_unsearched_refs_are_not_flagged() -> None:
    toc = _check("10 Critical batch size . . . . . . . . . . . 12", None)
    assert not raw.looks_like_reference(toc.raw_string)
    assert unresolved_flag(toc) is None and "parse artifact" in (not_checked_reason(toc) or "")
    crossref_only = _check(
        LAYERNORM, None, ["crossref bibliographic(raw) -> 5 record(s), 0 accepted"]
    )
    assert unresolved_flag(crossref_only) is None
    searched = _check(
        LAYERNORM, None, ["index search 'Layer normalization' -> 5 hit(s), 0 record(s) accepted"]
    )
    assert unresolved_flag(searched) is not None


def test_accept_rejects_different_number_and_unrelated_authors() -> None:
    rte = _ref(
        "Luisa Bentivogli, Bernardo Magnini, and Ido Dagan. 2009. The fifth PASCAL recognizing "
        "textual entailment challenge. In TAC."
    )
    third = _rec("The third PASCAL recognizing textual entailment challenge", ["Ido Dagan"], 2007)
    assert accept(rte, hints_for(rte), third) is None
    gpt3 = _ref(
        "Tom B Brown, Benjamin Mann, and Nick Ryder. Language models are few-shot learners. "
        "In NeurIPS, 2020."
    )
    reupload = _rec("Language Models are Few-Shot Learners", ["Sourav Malakar"], 2026)
    real = _rec("Language Models are Few-Shot Learners", ["Tom B. Brown", "Benjamin Mann"], 2020)
    assert accept(gpt3, hints_for(gpt3), reupload) is None
    assert accept(gpt3, hints_for(gpt3), real) is not None
    # organization-authored references have no personal names to require
    bb = _ref("BIG-bench collaboration. Beyond the imitation game. In preparation, 2021.")
    rec = _rec("Beyond the Imitation Game", ["Aarohi Srivastava"], 2022)
    assert accept(bb, hints_for(bb), rec) is not None


def test_author_check_skips_merged_references() -> None:
    merged = (
        "Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for "
        "generation. ACL. Zhengzhong Liang, Steven Bethard, and Mihai Surdeanu. 2021. "
        "Explainable multi-hop verbal reasoning through internal monologue."
    )
    r = _resolved(
        [_v(2021, False)],
        ["Zhengzhong Liang", "Steven J. Bethard", "Mihai Surdeanu"],
        title="Explainable multi-hop verbal reasoning through internal monologue",
    )
    assert author_flag(_check(merged, r)) is None


def test_disk_cache_concurrent_writes_of_same_key(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from reviewer.cache import DiskCache

    cache = DiskCache(tmp_path, "ns", ".json")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: cache.put_json("k", {"i": i}), range(200)))
    assert cache.get_json("k") is not None
    assert not list((tmp_path / "ns").glob("*.tmp"))


def test_crossref_repository_copy_gets_arxiv_original() -> None:
    from reviewer.citations.resolve import Resolver

    raw_s = (
        "Ashish Vaswani, Noam Shazeer, and Niki Parmar. Attention is all you need. In NIPS, 2017."
    )
    authors = ["Ashish Vaswani", "Noam Shazeer"]

    class Index:
        available = False

    class OpenAlex:
        def search_title(self, title: str) -> None:
            return None  # budget floor

    class Crossref:
        def bibliographic(self, text: str) -> list[Record]:
            copy = _rec("Attention Is All You Need", authors, 2025, source="crossref")
            return [copy]

    class Arxiv:
        def search_title(self, title: str):
            from reviewer.citations.arxiv_meta import ArxivRecord

            return [
                ArxivRecord(
                    arxiv_id="1706.03762",
                    title="Attention Is All You Need",
                    abstract="A",
                    authors=authors,
                    published=datetime(2017, 6, 12),
                )
            ]

    ref = _ref(raw_s, title="Attention is all you need")
    res = Resolver(Index(), OpenAlex(), Crossref(), Arxiv()).resolve(ref)  # type: ignore[arg-type]
    assert res.source == "crossref"
    assert "1706.03762" in res.arxiv_ids
    assert sorted(r.version.year for r, _ in res.records) == [2017, 2025]


QWEN25 = (
    "Qwen, :, An Yang, Baosong Yang, Beichen Zhang, and Zihan Qiu. Qwen2.5 technical report, "
    "2025. URL https://arxiv.org/abs/2412.15115."
)


def test_accept_requires_matching_version_suffix() -> None:
    ref = _ref(QWEN25, title="Qwen2.5 technical report")
    qwen2 = _rec("Qwen2 Technical Report", ["An Yang", "Baosong Yang"], 2024)
    qwen25 = _rec("Qwen2.5 Technical Report", ["An Yang", "Baosong Yang"], 2024)
    assert accept(ref, hints_for(ref), qwen2) is None
    assert accept(ref, hints_for(ref), qwen25) is not None
    # the other direction: a record with a version the reference does not print
    gpt4 = _ref("OpenAI. GPT-4 technical report. arXiv:2303.08774, 2023.")
    assert accept(gpt4, hints_for(gpt4), _rec("GPT-4.5 Technical Report", [], 2025)) is None


def test_version_numbers_ignore_identifiers() -> None:
    assert raw.version_numbers(
        "Llama 3.1 and GPT-3.5, arXiv:2407.21783, doi 10.18653/v1/2024.findings-acl.924"
    ) == {
        "3.1",
        "3.5",
    }


def test_strip_author_prefix_from_grobid_title() -> None:
    swallowed = "Ivo Danihelka, Oriol Vinyals, and Koray Kavukcuoglu. Video pixel networks"
    assert raw.strip_author_prefix(swallowed) == "Video pixel networks"
    assert raw.strip_author_prefix("Attention is all you need") == "Attention is all you need"
    assert raw.strip_author_prefix("Deep learning. Nature review") == "Deep learning. Nature review"


VPN = (
    "Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex "
    "Graves, and Koray Kavukcuoglu. Video pixel networks. In International Conference on "
    "Machine Learning, pages 1771-1779, 2017."
)


def test_index_searches_title_first_then_raw_string() -> None:
    from reviewer.citations.resolve import Resolver

    class Index:
        available = True

        def __init__(self, hit_on: str) -> None:
            self.hit_on = hit_on
            self.queries: list[str] = []

        def search(self, q: str, k: int = 5) -> list[dict[str, object]]:
            self.queries.append(q)
            if self.hit_on not in q:
                return []
            return [
                {
                    "key": "arxiv:1610.00527",
                    "kind": "arxiv",
                    "year": 2016,
                    "title": "Video Pixel Networks",
                }
            ]

    class OpenAlex:
        def by_arxiv(self, arxiv_id: str) -> list[Record]:
            return []

    grobid = (
        "Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks"
    )
    ref = _ref(VPN, title=grobid)
    index = Index(hit_on="Video pixel networks")
    res = Resolution(ref=ref, hints=hints_for(ref))
    Resolver(index, OpenAlex(), None, None)._index(res)  # type: ignore[arg-type]
    assert index.queries == ["Video pixel networks"]
    assert res.records and res.records[0][0].arxiv_id == "1610.00527"
    # a title search that finds nothing falls back to the raw string (author list)
    index = Index(hit_on="Kalchbrenner")
    res = Resolution(ref=ref, hints=hints_for(ref))
    Resolver(index, OpenAlex(), None, None)._index(res)  # type: ignore[arg-type]
    assert len(index.queries) == 2 and "Kalchbrenner" in index.queries[1]
    assert res.records
