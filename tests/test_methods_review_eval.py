import json
import re
from datetime import date

import httpx
import pytest
import typer

from reviewer.config import Settings
from reviewer.llm.client import LLMClient, TransportRequest, TransportResponse
from reviewer.methods.checker import load_methods_prompt
from reviewer.methods.openreview_pdf import (
    API,
    API_V1,
    OpenReviewError,
    OpenReviewPdfClient,
    PdfChoice,
    Revision,
    api_for,
    choose_revision,
    pdf_creation_date,
    provenance,
    retry_after_s,
    revisions_v1,
    revisions_v2,
)
from reviewer.methods.review_commands import run_review_eval
from reviewer.methods.review_eval import (
    SplitLockedError,
    canonical_category,
    cases_path,
    group_submissions,
    load_rows,
)
from reviewer.models import (
    ConcernCheck,
    ConcernCheckOutput,
    CritiqueMatch,
    CritiqueMatchOutput,
    TokenUsage,
)
from tests.test_methods import FakeLLM as FakeChecker
from tests.test_methods import _paper


def _row(sid="S1", rid="R1", cat="baselines", excerpt="No strong baselines.", **kw) -> dict:
    return {
        "submission_id": sid,
        "venue": "ICLR.cc/2024/Conference",
        "submission_title": "T",
        "review_id": rid,
        "review_field": "weaknesses",
        "category": cat,
        "review_excerpt": excerpt,
        "cutoff_date": "2023-09-28",
        "some_future_field": 1,
        **kw,
    }


def _write(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --------------------------------------------------------------------------- loading


def test_test_split_is_locked(tmp_path):
    with pytest.raises(SplitLockedError):
        cases_path(tmp_path, "test")
    with pytest.raises(SplitLockedError):
        cases_path(tmp_path, "dev", tmp_path / "methods_test.jsonl")
    assert cases_path(tmp_path, "dev") == tmp_path / "dev.jsonl"
    f = tmp_path / "dev.jsonl"
    _write(f, [_row(), _row(sid="S2", split="test")])
    with pytest.raises(SplitLockedError):
        load_rows(f)


def test_canonical_category():
    assert canonical_category("Leakage/Contamination") == "leakage_contamination"
    assert canonical_category("seeds/variance") == "seeds_variance"
    assert canonical_category("claims-vs-results") == "claims_vs_results"
    assert canonical_category("Compute fairness") == "compute_fairness"
    assert canonical_category("dataset/split disclosure") == "dataset_split_disclosure"
    assert canonical_category("Hyperparameter disclosure") == "hyperparameter_disclosure"
    assert canonical_category("evaluation scope") == "evaluation_scope"
    assert canonical_category("weird thing") == "weird_thing"


def test_load_and_group(tmp_path):
    f = tmp_path / "dev.jsonl"
    _write(
        f,
        [
            _row(sid="S2"),
            _row(sid="S1", cat="Seeds/Variance"),
            _row(sid="S1", cat="Seeds/Variance"),  # duplicate
            _row(sid="S1", rid="R2", excerpt="Test set used for tuning.", cat="leakage"),
        ],
    )
    subs = group_submissions(load_rows(f))
    assert [s.submission_id for s in subs] == ["S1", "S2"]
    s1 = subs[0]
    assert [c.category for c in s1.concerns] == ["seeds_variance", "leakage_contamination"]
    assert s1.cutoff_date == date(2023, 9, 28) and s1.paper_id == "or-S1"
    assert [e.limitation for e in s1.critiques().entries][1] == "Test set used for tuning."


def test_index_format_row(tmp_path):
    """The field names reviewer-index actually writes (data/methodology_eval/dev.jsonl)."""
    row = {
        "concern_id": "X1__R9__0",
        "venue": "NeurIPS.cc/2023/Conference",
        "submission_id": "X1",
        "submission_title": "T",
        "final_status": "rejected",
        "review_id": "R9",
        "field": "weaknesses",
        "category": "compute_fairness",
        "excerpt": "Runtime vs baselines is not compared.",
        "offsets": [3, 40],
        "model": "claude-sonnet-5",
        "prompt_version": "methodology_v1",
    }
    f = tmp_path / "dev.jsonl"
    _write(f, [row])
    (r,) = load_rows(f)
    assert r.review_excerpt == row["excerpt"] and r.review_field == "weaknesses"
    assert r.case_id == "X1__R9__0" and r.excerpt_offsets == [3, 40]
    (sub,) = group_submissions([r])
    assert sub.cutoff_date == date(2023, 5, 17)  # from the deadline table
    assert sub.final_status == "rejected"


# --------------------------------------------------------------------------- PDFs


def _ms(y, m, d) -> int:
    from datetime import UTC, datetime

    return int(datetime(y, m, d, tzinfo=UTC).timestamp() * 1000)


def test_choose_revision():
    revs = [
        Revision("a", _ms(2023, 9, 20), True),
        Revision("meta", _ms(2023, 9, 29), False),
        Revision("b", _ms(2023, 10, 2), True),  # within the 7-day grace
        Revision("rebuttal", _ms(2023, 11, 20), True),
    ]
    assert choose_revision(revs, date(2023, 9, 28)).revision_id == "b"
    assert choose_revision(revs, date(2023, 1, 1)).revision_id == "a"  # none before: earliest
    assert choose_revision(revs, None).revision_id == "a"
    assert choose_revision([Revision("m", 1, False)], date(2023, 9, 28)).revision_id is None


def test_provenance():
    hist = PdfChoice("e1", "latest PDF revision before 2023-09-28 + 7 days", "deadline")
    assert provenance(hist, None, date(2023, 9, 28)) == "deadline_revision"
    early = PdfChoice("e1", "no PDF revision before 2023-09-28 + 7 days; earliest", "earliest")
    assert provenance(early, None, date(2023, 9, 28)) == "earliest_revision"
    cur = PdfChoice(None, "no PDF revisions visible in history; current PDF")
    assert provenance(cur, date(2023, 9, 29), date(2023, 9, 28)) == "current_deadline_dated"
    assert provenance(cur, date(2023, 11, 20), date(2023, 9, 28)) == "current_revised"
    assert provenance(cur, None, date(2023, 9, 28)) == "current_undated"


def test_pdf_creation_date(tmp_path, monkeypatch):
    import subprocess

    def fake_run(*a, **kw):
        out = "Pages: 3\nCreationDate:    Fri Sep 29 07:01:32 2023 CDT\n"
        return subprocess.CompletedProcess(a, 0, out, "")

    monkeypatch.setattr("reviewer.methods.openreview_pdf.subprocess.run", fake_run)
    assert pdf_creation_date(tmp_path / "x.pdf") == date(2023, 9, 29)
    out = "CreationDate:    Sun Jun  2 19:52:19 2024 JST\n"
    monkeypatch.setattr(
        "reviewer.methods.openreview_pdf.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, out, ""),
    )
    assert pdf_creation_date(tmp_path / "x.pdf") == date(2024, 6, 2)


def test_api_for_and_revision_parsing():
    assert api_for("ICLR.cc/2023/Conference") == API_V1
    assert api_for("ICLR.cc/2024/Conference") == API
    assert api_for("NeurIPS.cc/2023/Conference") == API
    v2 = revisions_v2({"edits": [{"id": "e1", "tcdate": 5, "note": {"content": {"pdf": {}}}}]})
    assert v2 == [Revision("e1", 5, True)]
    v1 = revisions_v1({"references": [{"id": "r1", "tcdate": 7, "content": {"title": "x"}}]})
    assert v1 == [Revision("r1", 7, False)]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        methods_eval_dir=tmp_path / "methods_eval",
        anthropic_api_key=None,
        openreview_username="u",
        openreview_password="p",
        openreview_delay_s=0,
        openreview_v2_history=True,
        llm_max_retries=1,
        methods_prompt="methods_v1",  # the fakes answer with v1's nine items
    )


def test_pdf_client_v2_downloads_deadline_revision_once(settings):
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/login":
            return httpx.Response(200, json={"token": "t"})
        assert req.headers["Authorization"] == "Bearer t"
        if req.url.path == "/notes/edits":
            return httpx.Response(
                200,
                json={
                    "edits": [
                        {"id": "e1", "tcdate": _ms(2023, 9, 27), "note": {"content": {"pdf": 1}}},
                        {"id": "e2", "tcdate": _ms(2023, 11, 20), "note": {"content": {"pdf": 1}}},
                    ]
                },
            )
        if req.url.path == "/notes/edits/attachment":
            assert req.url.params["id"] == "e1"
            return httpx.Response(200, content=b"%PDF-1.5 fake")
        return httpx.Response(404)

    client = OpenReviewPdfClient(
        settings, http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    got = client.download("S1", "ICLR.cc/2024/Conference", date(2023, 9, 28))
    assert got.path.read_bytes().startswith(b"%PDF") and "revision e1" in got.reason
    assert got.provenance == "deadline_revision"
    n = len(calls)
    again = client.download("S1", "ICLR.cc/2024/Conference", date(2023, 9, 28))
    assert again.path == got.path and again.provenance == "deadline_revision"
    assert len(calls) == n  # served from disk


def test_retry_after_and_rate_limit_wait(settings, monkeypatch):
    assert retry_after_s("Please try again in 16 minutes and 16 seconds (x)") == 16 * 60 + 31
    assert retry_after_s("Please try again in 16 minutes (x)") == 16 * 60 + 15
    assert retry_after_s("Please try again in 40 seconds") == 55
    assert retry_after_s("nope") == 20 * 60
    slept: list[float] = []
    monkeypatch.setattr("reviewer.methods.openreview_pdf.time.sleep", slept.append)
    n = {"pdf": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/login":
            return httpx.Response(200, json={"token": "t"})
        assert req.url.path == "/pdf"  # v2 history off: no /notes/edits request
        n["pdf"] += 1
        if n["pdf"] == 1:
            return httpx.Response(429, text="Please try again in 1 minutes and 5 seconds")
        return httpx.Response(200, content=b"%PDF-1.5 fake")

    off = settings.model_copy(update={"openreview_v2_history": False})
    client = OpenReviewPdfClient(off, http=httpx.Client(transport=httpx.MockTransport(handler)))
    got = client.download("S7", "ICLR.cc/2025/Conference", date(2024, 10, 1))
    assert got.path.exists() and n["pdf"] == 2 and 80 in slept
    assert got.provenance == "current_undated"  # the fake PDF has no CreationDate


def test_pdf_client_needs_credentials_and_a_pdf(settings):
    no_creds = settings.model_copy(update={"openreview_username": None})
    client = OpenReviewPdfClient(no_creds, http=httpx.Client(transport=httpx.MockTransport(None)))
    with pytest.raises(OpenReviewError, match="OPENREVIEW_USERNAME"):
        client.download("S1", "ICLR.cc/2024/Conference", None)

    def html(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/login":
            return httpx.Response(200, json={"token": "t"})
        if req.url.path == "/references":
            return httpx.Response(200, json={"references": []})
        return httpx.Response(200, content=b"<html>challenge</html>")

    client = OpenReviewPdfClient(settings, http=httpx.Client(transport=httpx.MockTransport(html)))
    with pytest.raises(OpenReviewError, match="not a PDF"):
        client.download("S9", "ICLR.cc/2023/Conference", None)


# --------------------------------------------------------------------------- end to end


class FakeAll(FakeChecker):
    """Checker/repair from test_methods; the matcher matches concern 0 to seeds_variance and
    the rest to nothing; the verifier says valid."""

    def __call__(self, req: TransportRequest) -> TransportResponse:
        if req.schema is CritiqueMatchOutput:
            n = len(re.findall(r"<critique index=", req.user))
            ms = [
                CritiqueMatch(
                    critique_index=i,
                    scope="methodology",
                    label="match" if i == 0 else "none",
                    finding_ids=["seeds_variance"] if i == 0 else [],
                    reason="r",
                )
                for i in range(n)
            ]
            return TransportResponse(
                CritiqueMatchOutput(matches=ms), TokenUsage(input_tokens=10), "end_turn"
            )
        if req.schema is ConcernCheckOutput:
            ids = re.findall(r'<concern id="([^"]+)">', req.user)
            checks = [ConcernCheck(concern_id=i, validity="valid", reason="ok") for i in ids]
            return TransportResponse(
                ConcernCheckOutput(checks=checks), TokenUsage(input_tokens=10), "end_turn"
            )
        return super().__call__(req)


def _setup(settings, sids=("S1", "S2")):
    rows = []
    for sid in sids:
        rows += [
            _row(sid=sid, cat="seeds/variance", excerpt=f"{sid}: single seed only."),
            _row(sid=sid, rid="R2", cat="baselines", excerpt=f"{sid}: weak baselines."),
        ]
        paper = _paper(f"or-{sid}").model_copy(update={"title": f"Paper {sid}"})
        settings.openreview_parsed_dir.mkdir(parents=True, exist_ok=True)
        (settings.openreview_parsed_dir / f"or-{sid}.json").write_text(paper.model_dump_json())
    _write(settings.methods_eval_dir / "dev.jsonl", rows)


def test_review_eval_end_to_end(settings, monkeypatch):
    monkeypatch.setattr("reviewer.methods.review_commands.git_state", lambda: ("abc" * 5, False))
    _setup(settings)
    fake = FakeAll()
    items = run_review_eval(
        settings=settings, client=LLMClient(settings, transport=fake), precision_n=3
    )
    assert len(items) == 4
    assert [o.strict.value for o in items] == ["match", "none", "match", "none"]
    assert items[0].own_item is True and items[1].own_item is None
    assert items[0].pdf_provenance == "existing_parse"
    report = (settings.private_reports_dir / "methods_review_eval_dev.md").read_text()
    assert "| seeds_variance, strict | 2 | 2 | 0 | 100% | 100% |" in report
    assert "| baselines, strict | 2 | 0 | 0 | 0% | 0% |" in report
    assert "| strict | 4 | 2 | 0 | 50% | 50% |" in report
    assert "valid 3" in report and "commit abcabcabca" in report
    assert "precision lower bound): 2/" in report
    runs = (settings.private_reports_dir / "methods_review_eval_runs.jsonl").read_text()
    assert json.loads(runs.splitlines()[-1])["split"] == "dev"
    assert (settings.private_reports_dir / "methods_review_eval_precision_dev.md").exists()
    # a rerun reuses the saved reviews and the LLM cache: no new calls
    before = dict(fake.calls)
    run_review_eval(settings=settings, client=LLMClient(settings, transport=fake), precision_n=3)
    assert fake.calls == before


def test_review_eval_guards(settings, monkeypatch, capsys):
    monkeypatch.setattr("reviewer.methods.review_commands.git_state", lambda: ("abc" * 5, False))
    with pytest.raises(typer.Exit):  # no dev file yet
        run_review_eval(settings=settings, client=LLMClient(settings, transport=FakeAll()))
    _setup(settings, sids=("S1", "S2", "S3"))
    (settings.openreview_parsed_dir / "or-S3.json").unlink()
    assert run_review_eval(settings=settings, dry_run=True) == []
    out = capsys.readouterr().out
    assert "3 dev submissions, 6 concerns; 2 already parsed, 1 to download+parse" in out
    # max_cost 0: nothing runs; --no-fetch: unparsed submissions are recorded as failures
    items = run_review_eval(
        settings=settings, client=LLMClient(settings, transport=FakeAll()), max_cost=0
    )
    assert items == []
    items = run_review_eval(
        settings=settings, client=LLMClient(settings, transport=FakeAll()), fetch=False
    )
    assert len(items) == 4
    assert "no parse and --no-fetch" in settings.failures_file.read_text()
    with pytest.raises(SplitLockedError):
        run_review_eval("test", settings=settings)


# --------------------------------------------------------------------------- v2 + compare


def test_v2_checklist_and_validation():
    from reviewer.methods.checker import items_for, output_problems
    from reviewer.models import ChecklistFinding, ChecklistItemId, MethodsOutput

    v1, v2 = items_for("methods_v1+repair_v1"), items_for("methods_v2")
    assert ChecklistItemId.abstract_support in v1 and ChecklistItemId.claims_support not in v1
    assert v2[-2:] == [ChecklistItemId.claims_support, ChecklistItemId.evaluation_scope]
    out = MethodsOutput(
        findings=[
            ChecklistFinding(item=i, verdict="yes", rationale="r", quotes=[], concern="")
            for i in v1
        ],
        other_concerns=[],
    )
    problems = output_problems(out, v2)
    assert any("missing checklist items: claims_support, evaluation_scope" in p for p in problems)
    assert any("not on this checklist: abstract_support" in p for p in problems)
    assert load_methods_prompt("methods_v2").text.count("evaluation_scope:") == 1


def test_second_sample_is_disjoint_and_seeded():
    from reviewer.methods.review_commands import sample_submissions
    from reviewer.methods.review_eval import Submission

    subs = [Submission(f"S{i:03d}", "v", "t", None, ()) for i in range(200)]
    a = {s.submission_id for s in sample_submissions(subs, 30)}
    b = sample_submissions(subs, 25, seed=20260927, exclude_sample=30)
    assert len(b) == 25 and not a & {s.submission_id for s in b}
    assert b == sample_submissions(subs, 25, seed=20260927, exclude_sample=30)


def test_own_item_respects_asked_items():
    from reviewer.methods.evaluate import CritiqueOutcome
    from reviewer.methods.review_eval import Submission, concern_outcomes
    from reviewer.models import MethodsConcernRow

    row = MethodsConcernRow(
        submission_id="S", review_id="R", category="evaluation_scope", review_excerpt="one dataset"
    )
    sub = Submission("S", "v", "t", None, (row,))
    o = CritiqueOutcome("or-S", 0, "x", "methodology", "match", "match", ("other:0",), "r")
    (v1,) = concern_outcomes(sub, [o], asked={"abstract_support"})
    assert v1.own_item is None  # v1 has no evaluation_scope item
    (v2,) = concern_outcomes(sub, [o], asked={"evaluation_scope"})
    assert v2.own_item is False and v2.concern_index == 0


def test_render_compare_is_paired():
    from reviewer.methods.review_eval import render_compare
    from reviewer.models import ConcernOutcome

    def oc(sid, i, cat, label, checker):
        return ConcernOutcome(
            submission_id=sid,
            venue="v",
            review_id="R",
            category=cat,
            excerpt="e",
            scope="methodology",
            strict=label,
            lenient=label,
            finding_ids=[],
            own_item=None,
            reason="r",
            checker=checker,
            concern_index=i,
        )

    base = [oc("S1", 0, "baselines", "none", "v1"), oc("S2", 0, "baselines", "match", "v1")]
    new = [oc("S1", 0, "baselines", "match", "v2")]  # S2 failed in the new run
    text = "\n".join(render_compare("A", base, new, (1, 4), (2, 5)))
    assert "1 submissions reviewed by both, 1 concerns" in text
    assert "| **all** | 1 | 0% / 0% | 100% / 100% | +100 / +100 |" in text
    assert "v2: 5 strict findings, 2 also raised by a reviewer (40%)" in text


def test_interrupted_files_are_redone(settings, monkeypatch):
    """A half-written parse or provenance note (machine died mid-write) is not trusted."""
    from reviewer.methods.review_commands import OpenReviewParser
    from reviewer.methods.review_eval import Submission

    settings.openreview_parsed_dir.mkdir(parents=True)
    (settings.openreview_parsed_dir / "or-S1.json").write_text('{"paper_id": "or-S1", "tit')
    parser = OpenReviewParser(settings, fetch=False)
    sub = Submission("S1", "ICLR.cc/2024/Conference", "t", None, ())
    assert parser.existing(sub) is None
    assert not (settings.openreview_parsed_dir / "or-S1.json").exists()

    pdf_dir = settings.openreview_pdf_dir
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "S2.pdf").write_bytes(b"%PDF-1.5 old")
    (pdf_dir / "S2.revision.json").write_text('{"reason": "x", "prov')

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/login":
            return httpx.Response(200, json={"token": "t"})
        return httpx.Response(200, content=b"%PDF-1.5 fresh")

    off = settings.model_copy(update={"openreview_v2_history": False})
    client = OpenReviewPdfClient(off, http=httpx.Client(transport=httpx.MockTransport(handler)))
    got = client.download("S2", "ICLR.cc/2024/Conference", None)
    assert got.path.read_bytes() == b"%PDF-1.5 fresh"
    assert json.loads((pdf_dir / "S2.revision.json").read_text())["provenance"]
    assert not list(pdf_dir.glob("*.tmp"))
