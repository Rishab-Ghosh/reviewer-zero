"""ArxivAbstractClient with a fake curl runner; no network."""

import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from reviewer.config import Settings
from reviewer.corpus.arxiv_abstracts import ArxivAbstractClient, parse_abstracts


def _entry(arxiv_id: str, summary: str, title: str = "A  Title") -> str:
    return (
        f"<entry><id>http://arxiv.org/abs/{arxiv_id}</id><title>{title}</title>"
        f"<summary>{summary}</summary></entry>"
    )


def _feed(*entries: str) -> bytes:
    return ('<feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + "</feed>").encode()


class FakeCurl:
    """Answers each curl call with every requested id except `missing`, as version-2 entries."""

    def __init__(
        self, missing: set[str] | None = None, statuses: list[str] | None = None, exit_code: int = 0
    ) -> None:
        self.missing = missing or set()
        self.statuses = list(statuses or [])
        self.exit_code = exit_code
        self.batches: list[list[str]] = []
        self.args: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.args.append(args)
        query = parse_qs(urlparse(args[-1]).query)
        ids = query["id_list"][0].split(",")
        assert query["max_results"] == [str(len(ids))]
        self.batches.append(ids)
        status = self.statuses.pop(0) if self.statuses else "200"
        entries = [
            _entry(f"{i}v2", f"  Abstract of\n  {i} with $\\alpha$. ")
            for i in ids
            if i not in self.missing
        ]
        if self.missing:
            entries.append(
                "<entry><id>http://arxiv.org/api/errors#bad</id><summary>bad id</summary></entry>"
            )
        out = _feed(*entries) + f"\n__HTTP_CODE__:{status}".encode()
        return subprocess.CompletedProcess(args, self.exit_code, stdout=out, stderr=b"")


def _client(tmp_path: Path, fake: FakeCurl, sleeps: list[float]) -> ArxivAbstractClient:
    settings = Settings(data_dir=tmp_path / "data", arxiv_delay_s=3.0)
    return ArxivAbstractClient(settings, runner=fake, sleep=sleeps.append)


def test_parse_abstracts_strips_version_and_whitespace() -> None:
    xml = _feed(
        _entry("2201.11903v6", "  We explore\n  chain of thought $x^2$ . "),
        "<entry><id>http://arxiv.org/api/errors#x</id><title>Error</title>"
        "<summary>incorrect id format</summary></entry>",
    )
    out = parse_abstracts(xml)
    assert list(out) == ["2201.11903"]
    assert out["2201.11903"].abstract == "We explore chain of thought $x^2$ ."
    assert out["2201.11903"].title == "A Title"


def test_batching_delay_and_curl_args(tmp_path: Path) -> None:
    fake = FakeCurl()
    sleeps: list[float] = []
    ids = [f"2001.{i:05d}" for i in range(45)]
    out = _client(tmp_path, fake, sleeps).abstracts(ids)
    assert [len(b) for b in fake.batches] == [20, 20, 5]
    assert sleeps == [3.0, 3.0]
    assert len(out) == 45
    assert out["2001.00007"] == "Abstract of 2001.00007 with $\\alpha$."
    args = fake.args[0]
    assert args[0] == "curl" and "-s" in args
    assert args[args.index("-A") + 1] == Settings().user_agent
    assert args[args.index("--max-time") + 1] == "60"
    assert args[-1].startswith("https://export.arxiv.org/api/query?")


def test_cached_ids_not_requeried_and_versions_stripped(tmp_path: Path) -> None:
    fake = FakeCurl()
    sleeps: list[float] = []
    _client(tmp_path, fake, sleeps).abstracts(["1706.03762v5", "2305.18290"])
    assert fake.batches == [["1706.03762", "2305.18290"]]
    out = _client(tmp_path, fake, sleeps).abstracts(["1706.03762", "2305.18290v1", "2203.15556"])
    assert fake.batches[1:] == [["2203.15556"]]
    assert set(out) == {"1706.03762", "2305.18290", "2203.15556"}
    _client(tmp_path, fake, sleeps).abstracts(["1706.03762"])
    assert len(fake.batches) == 2
    assert sleeps == []  # one request per client instance: no delay needed


def test_partial_results_missing_ids(tmp_path: Path) -> None:
    fake = FakeCurl(missing={"2001.00002"})
    out = _client(tmp_path, fake, []).abstracts(["2001.00001", "2001.00002"])
    assert set(out) == {"2001.00001"}
    # The missing id was not cached, so it is asked for again next time.
    again = FakeCurl()
    _client(tmp_path, again, []).abstracts(["2001.00001", "2001.00002"])
    assert again.batches == [["2001.00002"]]


def test_http_error_and_curl_failure_give_partial_results(tmp_path: Path) -> None:
    fake = FakeCurl(statuses=["406", "200"])
    ids = [f"2001.{i:05d}" for i in range(25)]
    out = _client(tmp_path, fake, []).abstracts(ids)
    assert set(out) == set(ids[20:])

    fake = FakeCurl(exit_code=28)  # curl timeout
    assert _client(tmp_path, fake, []).abstracts(["2002.00001"]) == {}

    def raising(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError("curl")

    client = ArxivAbstractClient(Settings(data_dir=tmp_path / "data"), runner=raising)
    assert client.abstracts(["2002.00002"]) == {}


def test_malformed_xml_is_partial_failure(tmp_path: Path) -> None:
    def junk(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 0, stdout=b"<oops\n__HTTP_CODE__:200")

    client = ArxivAbstractClient(Settings(data_dir=tmp_path / "data"), runner=junk)
    assert client.abstracts(["2003.00001"]) == {}


def test_429_is_retried_with_backoff(tmp_path):
    import subprocess

    from reviewer.config import Settings
    from reviewer.corpus.arxiv_abstracts import ArxivAbstractClient

    feed = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1706.03762v7</id>'
        b"<title>T</title><summary> An  abstract. </summary></entry></feed>"
    )
    responses = [b"Rate exceeded.\n__HTTP_CODE__:429", feed + b"\n__HTTP_CODE__:200"]
    sleeps: list[float] = []

    def runner(args, **kw):
        out = responses.pop(0)
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr=b"")

    client = ArxivAbstractClient(Settings(data_dir=tmp_path), runner=runner, sleep=sleeps.append)
    assert client.abstracts(["1706.03762"]) == {"1706.03762": "An abstract."}
    assert any(s >= 15 for s in sleeps)
