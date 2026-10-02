"""GrobidClient and `reviewer parse` with httpx.MockTransport; no network."""

import json
from pathlib import Path

import httpx
import pytest
import typer

from reviewer.config import Settings
from reviewer.failures import load_failures
from reviewer.parse.commands import run_parse
from reviewer.parse.grobid_client import (
    GROBID_PARAMS,
    TEI_COORDINATES,
    GrobidClient,
    GrobidError,
    canonical_params,
    grobid_cache_key,
)
from tests.fulltext import fulltext_tei

FIXTURES = Path(__file__).parent / "fixtures"


class FakeGrobid:
    """Records requests; answers /api/isalive and fulltext with scripted responses."""

    def __init__(self, statuses: list[int] | None = None, tei: str = "<TEI/>") -> None:
        self.statuses = list(statuses or [])
        self.tei = tei
        self.fulltext_calls = 0
        self.last_body = b""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/isalive":
            return httpx.Response(200, text="true")
        assert request.url.path == "/api/processFulltextDocument"
        self.fulltext_calls += 1
        self.last_body = request.read()
        status = self.statuses.pop(0) if self.statuses else 200
        if status == 200:
            return httpx.Response(200, text=self.tei)
        return httpx.Response(status, text="x" * 500)


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", grobid_url="http://grobid.test")


def _client(tmp_path: Path, fake: FakeGrobid) -> GrobidClient:
    return GrobidClient(
        _settings(tmp_path), transport=httpx.MockTransport(fake), sleep=lambda _: None
    )


def _pdf(path: Path, content: bytes = b"%PDF-1.5 fake") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_second_call_hits_cache(tmp_path: Path) -> None:
    fake = FakeGrobid(tei="<TEI>one</TEI>")
    client = _client(tmp_path, fake)
    pdf = _pdf(tmp_path / "a.pdf")
    assert client.process(pdf) == "<TEI>one</TEI>"
    assert client.process(pdf) == "<TEI>one</TEI>"
    assert fake.fulltext_calls == 1


def test_request_is_multipart_with_params(tmp_path: Path) -> None:
    fake = FakeGrobid()
    _client(tmp_path, fake).process(_pdf(tmp_path / "a.pdf"))
    body = fake.last_body.decode("latin-1")
    assert 'name="input"; filename="a.pdf"' in body
    for key in GROBID_PARAMS:
        assert f'name="{key}"' in body


def test_key_depends_on_content_not_filename(tmp_path: Path) -> None:
    fake = FakeGrobid()
    client = _client(tmp_path, fake)
    client.process(_pdf(tmp_path / "a.pdf", b"same"))
    client.process(_pdf(tmp_path / "renamed.pdf", b"same"))
    assert fake.fulltext_calls == 1
    client.process(_pdf(tmp_path / "a.pdf", b"different"))
    assert fake.fulltext_calls == 2


def test_key_depends_on_params() -> None:
    other = {**GROBID_PARAMS, "consolidateCitations": "0"}
    assert grobid_cache_key("abc") != grobid_cache_key("abc", other)
    assert canonical_params({"b": "1", "a": "2"}) == "a=2&b=1"


def test_tei_coordinates_requested_and_in_cache_key(tmp_path: Path) -> None:
    coords = GROBID_PARAMS["teiCoordinates"]
    assert isinstance(coords, list)
    assert {"s", "head", "p", "formula", "figure", "ref", "biblStruct", "note"} <= set(coords)
    assert tuple(coords) == TEI_COORDINATES
    canonical = canonical_params(GROBID_PARAMS)
    for element in coords:
        assert f"teiCoordinates={element}" in canonical
    without = {k: v for k, v in GROBID_PARAMS.items() if k != "teiCoordinates"}
    assert grobid_cache_key("abc") != grobid_cache_key("abc", without)
    # repeated values are order-independent in the key
    shuffled = {**GROBID_PARAMS, "teiCoordinates": list(reversed(coords))}
    assert grobid_cache_key("abc") == grobid_cache_key("abc", shuffled)
    # and sent as one repeated multipart field per element
    fake = FakeGrobid()
    _client(tmp_path, fake).process(_pdf(tmp_path / "a.pdf"))
    body = fake.last_body.decode("latin-1")
    assert body.count('name="teiCoordinates"') == len(coords)


def test_503_is_retried(tmp_path: Path) -> None:
    fake = FakeGrobid(statuses=[503, 503, 200], tei="<TEI>ok</TEI>")
    assert _client(tmp_path, fake).process(_pdf(tmp_path / "a.pdf")) == "<TEI>ok</TEI>"
    assert fake.fulltext_calls == 3


def test_503_gives_up_after_max_retries(tmp_path: Path) -> None:
    fake = FakeGrobid(statuses=[503] * 10)
    with pytest.raises(GrobidError, match="HTTP 503"):
        _client(tmp_path, fake).process(_pdf(tmp_path / "a.pdf"))
    assert fake.fulltext_calls == 4


def test_other_error_raises_with_status_and_body_prefix(tmp_path: Path) -> None:
    fake = FakeGrobid(statuses=[500])
    with pytest.raises(GrobidError) as info:
        _client(tmp_path, fake).process(_pdf(tmp_path / "a.pdf"))
    msg = str(info.value)
    assert "HTTP 500" in msg
    assert "x" * 200 in msg
    assert "x" * 201 not in msg
    assert fake.fulltext_calls == 1


def test_is_alive(tmp_path: Path) -> None:
    assert _client(tmp_path, FakeGrobid()).is_alive()

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = GrobidClient(_settings(tmp_path), transport=httpx.MockTransport(down))
    assert not client.is_alive()


# --------------------------------------------------------------------------- run_parse


def test_run_parse_writes_json_skips_and_records_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake = FakeGrobid(statuses=[200, 500], tei=fulltext_tei("1706.03762"))
    real_init = GrobidClient.__init__

    def init(self: GrobidClient, s: Settings, transport: httpx.BaseTransport | None = None) -> None:
        real_init(self, s, transport=httpx.MockTransport(fake), sleep=lambda _: None)

    monkeypatch.setattr(GrobidClient, "__init__", init)
    # No network or models in unit tests: stub the arXiv abstract lookup and Docling.
    from reviewer.corpus.arxiv_abstracts import ArxivAbstractClient
    from reviewer.parse.docling_tables import DoclingTableExtractor

    monkeypatch.setattr(
        ArxivAbstractClient, "abstracts", lambda self, ids: dict.fromkeys(ids, "abs")
    )
    monkeypatch.setattr(DoclingTableExtractor, "extract", lambda self, pdf: [])
    papers = tmp_path / "papers" / "arxiv"
    _pdf(papers / "1706.03762.pdf", b"attention")
    _pdf(papers / "2501.00001v2.pdf", b"broken")

    run_parse(papers, settings=settings)

    out = settings.parsed_dir / "1706.03762.json"
    data = json.loads(out.read_text())
    assert data["title"] == "Attention Is All You Need"
    assert data["parse_report"]["has_contributions"] is False
    assert data["contributions"] is None
    failures = load_failures(settings.failures_file)
    assert [(f.stage, f.paper_id) for f in failures] == [("parse", "2501.00001")]
    assert "HTTP 500" in failures[0].error

    calls = fake.fulltext_calls
    run_parse(papers / "1706.03762.pdf", settings=settings)  # already parsed → skipped
    assert fake.fulltext_calls == calls
    run_parse(papers / "1706.03762.pdf", force=True, settings=settings)  # cache, not network
    assert fake.fulltext_calls == calls


def test_run_parse_fails_fast_when_grobid_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GrobidClient, "is_alive", lambda self: False)
    _pdf(tmp_path / "p.pdf")
    with pytest.raises(typer.Exit):
        run_parse(tmp_path / "p.pdf", settings=_settings(tmp_path))


def test_dropped_connection_waits_for_restart_and_retries_once(tmp_path):
    import httpx

    from reviewer.config import Settings
    from reviewer.parse.grobid_client import GrobidClient

    calls = {"post": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("isalive"):
            return httpx.Response(200, text="true")
        calls["post"] += 1
        if calls["post"] == 1:
            raise httpx.ReadError("Connection reset by peer", request=request)
        return httpx.Response(200, text="<TEI/>")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 crash")
    client = GrobidClient(
        Settings(data_dir=tmp_path), transport=httpx.MockTransport(handler), sleep=lambda s: None
    )
    assert client.process(pdf) == "<TEI/>"
    assert calls["post"] == 2


def test_repeated_dropped_connection_raises(tmp_path):
    import httpx
    import pytest

    from reviewer.config import Settings
    from reviewer.parse.grobid_client import GrobidClient, GrobidError

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("isalive"):
            return httpx.Response(200, text="true")
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    pdf = tmp_path / "y.pdf"
    pdf.write_bytes(b"%PDF-1.4 crash twice")
    client = GrobidClient(
        Settings(data_dir=tmp_path), transport=httpx.MockTransport(handler), sleep=lambda s: None
    )
    with pytest.raises(GrobidError, match="Server disconnected"):
        client.process(pdf)
