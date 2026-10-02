"""Corpus workstream tests. Network-free: arXiv is faked with httpx.MockTransport."""

from pathlib import Path

import httpx
import pytest

from reviewer.config import PROJECT_ROOT, Settings
from reviewer.corpus.arxiv_client import ArxivClient, ArxivError, parse_atom_feed
from reviewer.corpus.commands import fetch_ids
from reviewer.corpus.corpus import CorpusError, discover_own, load_corpus, load_corpus_yaml
from reviewer.corpus.ids import is_arxiv_id, own_paper_id, strip_version
from reviewer.failures import load_failures
from reviewer.models import PaperTier

REAL_CORPUS = PROJECT_ROOT / "data" / "corpus.yaml"
needs_corpus = pytest.mark.skipif(
    not REAL_CORPUS.exists(), reason="corpus.yaml is not in the public repo"
)
FAKE_PDF = b"%PDF-1.5\n%fake pdf body\n%%EOF\n"


def make_settings(tmp_path: Path, delay: float = 3.0) -> Settings:
    return Settings(data_dir=tmp_path / "data", arxiv_delay_s=delay, _env_file=None)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def pdf_handler(request: httpx.Request) -> httpx.Response:
    arxiv_id = request.url.path.removeprefix("/pdf/")
    if arxiv_id == "9999.99999":
        return httpx.Response(404, text="not found")
    if arxiv_id == "2401.00002":
        return httpx.Response(200, html="<html>captcha</html>")
    return httpx.Response(200, content=FAKE_PDF, headers={"content-type": "application/pdf"})


class CountingTransport(httpx.MockTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return pdf_handler(request)


# ------------------------------------------------------------------------------------ ids


@pytest.mark.parametrize(
    ("s", "ok"),
    [
        ("1706.03762", True),
        ("2106.09685v2", True),
        ("2601.19871", True),
        ("hep-th/9901001", True),
        ("math.GT/0309136v1", True),
        ("own-foo-1234abcd", False),
        ("1706.037", False),
        ("", False),
    ],
)
def test_is_arxiv_id(s: str, ok: bool) -> None:
    assert is_arxiv_id(s) is ok


def test_strip_version() -> None:
    assert strip_version("2106.09685v2") == "2106.09685"
    assert strip_version("1706.03762") == "1706.03762"
    assert strip_version("own-paper-v2") == "own-paper-v2"


def test_own_paper_id_format_and_determinism() -> None:
    sha = "ABCDEF0123456789" + "0" * 48
    pid = own_paper_id(Path("/x/My Great  Paper (v3).PDF"), sha)
    assert pid == "own-my-great-paper-v3-abcdef01"
    assert own_paper_id(Path("/other/dir/My Great  Paper (v3).PDF"), sha) == pid


def test_own_paper_id_slug_truncated() -> None:
    pid = own_paper_id(Path("a" * 100 + ".pdf"), "f" * 64)
    assert pid == "own-" + "a" * 40 + "-ffffffff"


def test_own_paper_id_depends_on_content(tmp_path: Path) -> None:
    own = tmp_path / "own"
    own.mkdir()
    (own / "paper.pdf").write_bytes(b"%PDF-1 version one")
    first = discover_own(own)[0].paper_id
    (own / "paper.pdf").write_bytes(b"%PDF-1 version two")
    second = discover_own(own)[0].paper_id
    assert first != second
    assert first.startswith("own-paper-") and second.startswith("own-paper-")


# --------------------------------------------------------------------------------- corpus


@needs_corpus
def test_real_corpus_yaml_validates() -> None:
    entries = load_corpus_yaml(REAL_CORPUS)
    assert len(entries) == 30
    tiers = [e.tier for e in entries]
    assert tiers.count(PaperTier.famous) == 12
    assert tiers.count(PaperTier.ordinary) == 8
    assert tiers.count(PaperTier.lab) == 10
    ids = [e.paper_id for e in entries]
    assert len(set(ids)) == len(ids)
    for e in entries:
        assert is_arxiv_id(e.paper_id) and strip_version(e.paper_id) == e.paper_id
        assert e.reason.strip() and e.title and e.pdf_path is None


@needs_corpus
def test_load_corpus_discovers_own(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    settings.corpus_file.write_text(REAL_CORPUS.read_text(encoding="utf-8"), encoding="utf-8")
    settings.own_dir.mkdir(parents=True)
    (settings.own_dir / "B paper.pdf").write_bytes(b"%PDF-1 b")
    (settings.own_dir / "a_paper.PDF").write_bytes(b"%PDF-1 a")
    (settings.own_dir / "notes.txt").write_text("ignored")

    entries = load_corpus(settings)
    own = [e for e in entries if e.tier == PaperTier.own]
    assert len(entries) == 32
    assert [e.paper_id.rsplit("-", 1)[0] for e in own] == ["own-a-paper", "own-b-paper"]
    assert all(e.pdf_path and Path(e.pdf_path).exists() for e in own)


@needs_corpus
def test_load_corpus_without_own_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    settings.corpus_file.write_text(REAL_CORPUS.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(load_corpus(settings)) == 30


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True)
    settings.corpus_file.write_text(
        "papers:\n"
        "  - {paper_id: '1706.03762', tier: famous, reason: a, title: t}\n"
        "  - {paper_id: '1706.03762', tier: ordinary, reason: b, title: t}\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="duplicate"):
        load_corpus(settings)


@pytest.mark.parametrize(
    "entry",
    [
        "{paper_id: 'not-an-id', tier: famous, reason: a}",
        "{paper_id: '1706.03762', tier: own, reason: a}",
        "{paper_id: '1706.03762', tier: legendary, reason: a}",
    ],
)
def test_invalid_entries_rejected(tmp_path: Path, entry: str) -> None:
    path = tmp_path / "corpus.yaml"
    path.write_text(f"papers:\n  - {entry}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus_yaml(path)


# ---------------------------------------------------------------------------- arXiv client


def test_download_writes_pdf_and_sends_user_agent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    transport = CountingTransport()
    dest = tmp_path / "out" / "2401.00001.pdf"
    with ArxivClient(settings, transport=transport, sleep=SleepRecorder()) as client:
        assert client.download_pdf("2401.00001", dest) == dest
    assert dest.read_bytes() == FAKE_PDF
    assert not dest.with_suffix(".pdf.part").exists()
    assert transport.requests[0].headers["user-agent"] == settings.user_agent
    assert str(transport.requests[0].url) == "https://arxiv.org/pdf/2401.00001"


def test_download_skips_existing_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    transport = CountingTransport()
    dest = tmp_path / "2401.00001.pdf"
    dest.write_bytes(b"%PDF-existing")
    with ArxivClient(settings, transport=transport, sleep=SleepRecorder()) as client:
        client.download_pdf("2401.00001", dest)
    assert transport.requests == []
    assert dest.read_bytes() == b"%PDF-existing"


def test_download_404_raises(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    dest = tmp_path / "9999.99999.pdf"
    with (
        ArxivClient(settings, transport=CountingTransport(), sleep=SleepRecorder()) as client,
        pytest.raises(ArxivError, match="404"),
    ):
        client.download_pdf("9999.99999", dest)
    assert not dest.exists()


def test_download_non_pdf_raises(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    dest = tmp_path / "2401.00002.pdf"
    with (
        ArxivClient(settings, transport=CountingTransport(), sleep=SleepRecorder()) as client,
        pytest.raises(ArxivError, match="not a PDF"),
    ):
        client.download_pdf("2401.00002", dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_delay_between_requests_but_not_after_skip(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, delay=3.0)
    sleep = SleepRecorder()
    transport = CountingTransport()
    (tmp_path / "2401.00003.pdf").write_bytes(FAKE_PDF)
    with ArxivClient(settings, transport=transport, sleep=sleep) as client:
        client.download_pdf("2401.00001", tmp_path / "2401.00001.pdf")  # network, no prior
        assert sleep.calls == []
        client.download_pdf("2401.00003", tmp_path / "2401.00003.pdf")  # skip
        assert sleep.calls == []
        client.download_pdf("2401.00004", tmp_path / "2401.00004.pdf")  # network
    assert len(sleep.calls) == 1
    assert 0 < sleep.calls[0] <= 3.0
    assert len(transport.requests) == 2


def test_fetch_ids_records_failures_and_continues(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.arxiv_dir.mkdir(parents=True)
    (settings.arxiv_dir / "2401.00003.pdf").write_bytes(FAKE_PDF)
    ids = ["2401.00001", "9999.99999", "2401.00002", "2401.00003"]
    with ArxivClient(settings, transport=CountingTransport(), sleep=SleepRecorder()) as client:
        fetched, skipped, failed = fetch_ids(ids, settings, client)
    assert (fetched, skipped, failed) == (1, 1, 2)
    failures = load_failures(settings.failures_file)
    assert [(f.stage, f.paper_id) for f in failures] == [
        ("fetch", "9999.99999"),
        ("fetch", "2401.00002"),
    ]


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <arxiv:comment>15 pages, 5 figures</arxiv:comment>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


def test_parse_atom_feed() -> None:
    meta = parse_atom_feed(ATOM)["1706.03762"]
    assert meta.title == "Attention Is All You Need"
    assert meta.primary_category == "cs.CL"
    assert meta.categories == ["cs.CL", "cs.LG"]
    assert meta.comment == "15 pages, 5 figures"
    assert meta.published.year == 2017


def test_fetch_metadata_batches_ids(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=ATOM)

    settings = make_settings(tmp_path)
    transport = httpx.MockTransport(handler)
    with ArxivClient(settings, transport=transport, sleep=SleepRecorder()) as client:
        meta = client.fetch_metadata(["1706.03762v7", "1810.04805"])
    assert list(meta) == ["1706.03762"]
    assert httpx.URL(seen[0]).params["id_list"] == "1706.03762,1810.04805"
