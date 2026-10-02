import json

import httpx2
import pytest
from pydantic import BaseModel

from reviewer.config import Settings
from reviewer.llm.client import (
    LLMAPIError,
    LLMClient,
    LLMMalformedOutputError,
    LLMRefusalError,
    LLMTruncatedError,
    MissingAPIKeyError,
    TransportRequest,
    TransportResponse,
    anthropic_transport,
    compute_cost,
)
from reviewer.models import TokenUsage


class Answer(BaseModel):
    value: str


class FakeTransport:
    def __init__(self, responses: list[TransportResponse] | None = None) -> None:
        self.responses = responses
        self.calls: list[TransportRequest] = []

    def __call__(self, req: TransportRequest) -> TransportResponse:
        self.calls.append(req)
        if self.responses:
            return self.responses.pop(0)
        return TransportResponse(
            parsed=Answer(value=f"{req.model}:{req.user}"),
            usage=TokenUsage(input_tokens=1000, output_tokens=200),
            stop_reason="end_turn",
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, anthropic_api_key=None)


def _call(client: LLMClient, **kw):
    args = {"system": "sys", "user": "hi", "model": "claude-sonnet-5", "cache_tag": "extract_v1"}
    args.update(kw)
    return client.structured(schema=Answer, **args)


def test_cache_hit_avoids_second_call_and_keeps_cost(settings):
    fake = FakeTransport()
    client = LLMClient(settings, transport=fake)
    first, u1 = _call(client)
    second, u2 = _call(client)
    assert len(fake.calls) == 1
    assert first == second
    assert not u1.from_cache and u2.from_cache
    assert u2.cost_usd == pytest.approx(u1.cost_usd) and u1.cost_usd > 0
    assert client.totals.n_fresh == 1 and client.totals.n_cached == 1
    assert client.totals.cached.cost_usd == pytest.approx(u1.cost_usd)

    # a fresh client over the same cache dir also hits
    fake2 = FakeTransport()
    _call(LLMClient(settings, transport=fake2))
    assert fake2.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-opus-5-5"},
        {"cache_tag": "extract_v2"},
        {"user": "other"},
        {"system": "s2"},
    ],
)
def test_changed_inputs_make_new_call(settings, change):
    fake = FakeTransport()
    client = LLMClient(settings, transport=fake)
    _call(client)
    _call(client, **change)
    assert len(fake.calls) == 2


def test_request_carries_settings(settings):
    fake = FakeTransport()
    _call(LLMClient(settings, transport=fake))
    req = fake.calls[0]
    assert req.effort == settings.llm_effort and req.max_tokens == settings.llm_max_tokens
    assert req.schema is Answer


def test_compute_cost():
    usage = TokenUsage(
        input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=500_000
    )
    assert compute_cost("claude-sonnet-5", usage) == pytest.approx(1.5 * 2.0 + 0.1 * 10.0)
    assert compute_cost("claude-opus-5-5", usage) == pytest.approx(1.5 * 4.0 + 0.1 * 20.0)
    assert compute_cost("unknown-model", usage) == 0.0


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (TransportResponse(None, TokenUsage(), "refusal"), LLMRefusalError),
        (TransportResponse(None, TokenUsage(output_tokens=16000), "max_tokens"), LLMTruncatedError),
        (TransportResponse(None, TokenUsage(), "end_turn", "bad json"), LLMMalformedOutputError),
    ],
)
def test_failed_responses_raise_and_are_not_cached(settings, response, error):
    fake = FakeTransport([response])
    client = LLMClient(settings, transport=fake)
    with pytest.raises(error):
        _call(client)
    _call(client)  # next call goes to the transport again (default fake answer)
    assert len(fake.calls) == 2
    assert client.totals.n_fresh == 2


def test_missing_api_key_only_for_uncached_calls(settings):
    LLMClient(settings, transport=FakeTransport()).structured(
        system="sys", user="hi", model="claude-sonnet-5", schema=Answer, cache_tag="t"
    )
    no_key = LLMClient(settings)
    _, usage = no_key.structured(
        system="sys", user="hi", model="claude-sonnet-5", schema=Answer, cache_tag="t"
    )
    assert usage.from_cache
    with pytest.raises(MissingAPIKeyError):
        no_key.structured(
            system="sys", user="new", model="claude-sonnet-5", schema=Answer, cache_tag="t"
        )


def _mock_sdk(monkeypatch, body: dict) -> list[dict]:
    """Route the real SDK through an in-process mock HTTP transport (no network)."""
    import anthropic

    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200, content=_sse(body).encode(), headers={"content-type": "text/event-stream"}
        )

    real_init = anthropic.Anthropic.__init__

    def init(self, **kw):
        kw["http_client"] = httpx2.Client(transport=httpx2.MockTransport(handler))
        real_init(self, **kw)

    monkeypatch.setattr(anthropic.Anthropic, "__init__", init)
    return sent


def _sse(message: dict) -> str:
    """Server-sent events for `message`, as the streaming Messages API sends them."""
    start = {**message, "content": [], "stop_reason": None}
    start["usage"] = {**message["usage"], "output_tokens": 0}
    events: list[tuple[str, dict]] = [
        ("message_start", {"type": "message_start", "message": start})
    ]
    for i, block in enumerate(message["content"]):
        events += [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block["text"]},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": i}),
        ]
    events += [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        },
    }


def _request(**kw) -> TransportRequest:
    args = {
        "model": "claude-sonnet-5",
        "system": "sys",
        "user": "hi",
        "schema": Answer,
        "max_tokens": 100,
        "effort": "medium",
    }
    args.update(kw)
    return TransportRequest(**args)


def test_default_transport_parses_and_sends_expected_body(monkeypatch):
    sent = _mock_sdk(monkeypatch, _message('{"value": "ok"}'))
    resp = anthropic_transport("sk-test")(_request())
    assert resp.parsed == Answer(value="ok")
    assert resp.usage.input_tokens == 10 and resp.usage.cache_read_input_tokens == 0
    body = sent[0]
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in body and "temperature" not in body
    assert body["system"] == "sys"
    assert body["stream"] is True


def test_default_transport_keeps_stop_reason_on_truncated_json(monkeypatch):
    _mock_sdk(monkeypatch, _message('{"value": "o', stop_reason="max_tokens"))
    resp = anthropic_transport("sk-test")(_request())
    assert resp.parsed is None and resp.stop_reason == "max_tokens"
    assert resp.error is not None


def test_default_transport_wraps_a_mid_stream_network_failure(monkeypatch):
    """A connection that dies mid-stream (e.g. the machine losing power and coming back, after
    the SDK's own request-level wrapping has already succeeded and events are being iterated)
    must not escape as a raw httpx2 error and crash the caller: it becomes an LLMAPIError like
    any other API failure, so one failed item is recorded instead of crashing the whole run."""
    from anthropic.resources.messages.messages import Messages

    class _DeadStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            raise httpx2.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )

    monkeypatch.setattr(Messages, "stream", lambda self, **kw: _DeadStream())
    with pytest.raises(LLMAPIError, match="RemoteProtocolError"):
        anthropic_transport("sk-test")(_request())


def test_anthropic_transport_sets_an_explicit_read_timeout(monkeypatch):
    """No timeout means a stalled connection (no error, no more bytes) can hang forever, since
    streaming has no fixed overall deadline; this is what let a dead laptop's stuck connection
    sit open for hours instead of failing fast."""
    import anthropic

    seen: dict = {}
    real_init = anthropic.Anthropic.__init__

    def init(self, **kw):
        seen["timeout"] = kw.get("timeout")
        kw["http_client"] = httpx2.Client(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(200))
        )
        real_init(self, **kw)

    monkeypatch.setattr(anthropic.Anthropic, "__init__", init)
    anthropic_transport("sk-test", timeout_s=42.0)
    assert isinstance(seen["timeout"], httpx2.Timeout)
    assert seen["timeout"].read == 42.0
