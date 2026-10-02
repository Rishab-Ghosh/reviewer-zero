"""Cached Anthropic client for structured (Pydantic) outputs, with token and cost accounting.

The network call sits behind a small `Transport` callable so tests can inject fakes:
`TransportRequest` in, `TransportResponse` (parsed model or None, raw usage, stop_reason) out.
Responses are cached on disk under `<settings.paper_cache>/llm/`, keyed by a hash of
(cache_tag, model, schema name, system prompt, user message).
"""

import logging
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from reviewer.cache import DiskCache, content_key
from reviewer.config import MODEL_PRICES_PER_MTOK, Settings
from reviewer.models import TokenUsage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- errors


class LLMError(RuntimeError):
    """A structured call that produced no usable output. `usage` is what the attempt cost."""

    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or TokenUsage()


class LLMRefusalError(LLMError):
    """The model refused (stop_reason 'refusal')."""


class LLMTruncatedError(LLMError):
    """The model hit max_tokens before finishing (stop_reason 'max_tokens')."""


class LLMMalformedOutputError(LLMError):
    """The response did not parse into the requested schema."""


class LLMAPIError(LLMError):
    """The API call itself failed (after the SDK's own retries for 429/5xx)."""


class MissingAPIKeyError(LLMError):
    """A non-cached call was needed but ANTHROPIC_API_KEY is not set."""


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True)
class TransportRequest:
    model: str
    system: str
    user: str
    schema: type[BaseModel]
    max_tokens: int
    effort: str


@dataclass(frozen=True)
class TransportResponse:
    parsed: BaseModel | None
    """The parsed output, or None if the response did not validate against the schema."""
    usage: TokenUsage
    """Raw token counts; cost is filled in by the client."""
    stop_reason: str | None
    error: str | None = None
    """Why `parsed` is None, when known (e.g. the validation error)."""


Transport = Callable[[TransportRequest], TransportResponse]


def _usage_from_sdk(usage: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _parse_text(text: str, schema: type[BaseModel]) -> tuple[BaseModel | None, str | None]:
    try:
        return schema.model_validate_json(text), None
    except ValidationError as e:
        return None, f"output did not validate against {schema.__name__}: {e}"


def anthropic_transport(api_key: str, max_retries: int = 4, timeout_s: float = 180.0) -> Transport:
    """Default transport. Sends exactly the body `messages.parse(output_format=...)` sends, via
    `messages.stream`, and parses the text itself so stop_reason and usage survive a response
    that fails validation. The SDK retries 429/5xx up to `max_retries` times.

    `timeout_s` bounds every phase (connect, read, write, pool) of the underlying HTTP
    request: without it a stalled connection (peer stops sending mid-stream, no error, no
    more bytes) can hang forever, since streaming has no fixed overall deadline. On timeout
    the SDK's own retry applies; if that too is exhausted, or the connection fails outright
    (e.g. the peer closes it mid-message), the failure is raised as LLMAPIError so callers
    record it as one failed item and move on, instead of crashing the whole run."""
    import anthropic
    import httpx2

    sdk = anthropic.Anthropic(
        api_key=api_key,
        max_retries=max_retries,
        timeout=httpx2.Timeout(timeout_s, connect=10.0),
    )

    def send(req: TransportRequest) -> TransportResponse:
        output_config: Any = {
            "effort": req.effort,
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(req.schema.model_json_schema()),
            },
        }
        try:
            # Streaming, so long outputs (large max_tokens, adaptive thinking) don't hit HTTP
            # timeouts; the final message is identical to a non-streaming response.
            with sdk.messages.stream(
                model=req.model,
                max_tokens=req.max_tokens,
                system=req.system,
                messages=[{"role": "user", "content": req.user}],
                output_config=output_config,
            ) as stream:
                message = stream.get_final_message()
        except (anthropic.APIError, httpx2.HTTPError) as e:
            # httpx2.HTTPError: a network-level failure (e.g. the peer closes the
            # connection mid-stream) that the SDK does not wrap as an APIError and would
            # otherwise propagate raw, crashing the whole run instead of failing one item.
            raise LLMAPIError(f"{req.model} API error: {type(e).__name__}: {e}") from e
        usage = _usage_from_sdk(message.usage)
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text.strip():
            return TransportResponse(None, usage, message.stop_reason, "empty text output")
        parsed, error = _parse_text(text, req.schema)
        return TransportResponse(parsed, usage, message.stop_reason, error)

    return send


# --------------------------------------------------------------------------- cost


def compute_cost(model: str, usage: TokenUsage) -> float:
    """USD cost; cache reads/writes billed at the input rate (upper bound, see config)."""
    prices = MODEL_PRICES_PER_MTOK.get(model)
    if prices is None:
        logger.warning("no price for model %r; recording cost as 0", model)
        return 0.0
    input_rate, output_rate = prices
    n_input = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    return (n_input * input_rate + usage.output_tokens * output_rate) / 1_000_000


def with_cost(model: str, usage: TokenUsage) -> TokenUsage:
    return usage.model_copy(update={"cost_usd": compute_cost(model, usage)})


# --------------------------------------------------------------------------- client


@dataclass
class UsageTotals:
    fresh: TokenUsage = field(default_factory=TokenUsage)
    """Spend on real API calls made by this client (including failed attempts)."""
    cached: TokenUsage = field(default_factory=TokenUsage)
    """Original cost of responses served from the disk cache (no new spend)."""
    n_fresh: int = 0
    n_cached: int = 0


def cache_key(
    cache_tag: str, model: str, schema: type[BaseModel], system: str, user: str, effort: str
) -> str:
    """Hash of (prompt version, model, effort, schema, input text)."""
    return content_key(cache_tag, model, f"effort={effort}", schema.__name__, system, user)


def check_stop_reason(stop_reason: str | None, model: str, usage: TokenUsage) -> None:
    if stop_reason == "refusal":
        raise LLMRefusalError(f"{model} refused to answer (stop_reason=refusal)", usage)
    if stop_reason == "max_tokens":
        raise LLMTruncatedError(
            f"{model} hit max_tokens ({usage.output_tokens} output tokens) before finishing",
            usage,
        )


class LLMClient:
    def __init__(self, settings: Settings, transport: Transport | None = None) -> None:
        self.settings = settings
        self._transport = transport
        self.cache = DiskCache(settings.paper_cache, "llm", ".json")
        self.totals = UsageTotals()
        self._totals_lock = threading.Lock()

    def _get_transport(self) -> Transport:
        if self._transport is None:
            if not self.settings.anthropic_api_key:
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set (add it to .env); it is needed for an "
                    "uncached LLM call"
                )
            self._transport = anthropic_transport(
                self.settings.anthropic_api_key, timeout_s=self.settings.llm_timeout_s
            )
        return self._transport

    def _from_cache[T: BaseModel](self, key: str, schema: type[T]) -> tuple[T, TokenUsage] | None:
        payload = self.cache.get_json(key)
        if payload is None:
            return None
        try:
            parsed = schema.model_validate(payload["parsed"])
            usage = TokenUsage.model_validate(payload["usage"])
        except (KeyError, ValidationError) as e:
            logger.warning("ignoring unreadable cache entry %s: %s", key[:12], e)
            return None
        usage = usage.model_copy(update={"from_cache": True})
        with self._totals_lock:
            self.totals.cached += usage
            self.totals.n_cached += 1
        logger.info(
            "LLM cache hit %s (%s, %s): %d in / %d out tokens, original cost $%.4f",
            key[:12],
            payload.get("model"),
            schema.__name__,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
        )
        return parsed, usage

    def _send(
        self, request: TransportRequest, cache_tag: str
    ) -> tuple[TransportResponse, TokenUsage]:
        """One uncached call, inside the settings' call guard (budget + step deadline) if any.
        A call that raises without usage is charged its whole hold by the guard."""
        guard = self.settings.call_guard
        logger.info(
            "LLM call %s (%s, %s, %d chars, max_tokens %d)",
            request.model,
            cache_tag,
            request.schema.__name__,
            len(request.user),
            request.max_tokens,
        )
        with guard(request) if guard is not None else nullcontext() as slot:
            try:
                response = self._get_transport()(request)
            except MissingAPIKeyError:
                if slot is not None:
                    slot.actual_usd = 0.0  # never sent
                raise
            except LLMError as e:
                # Charge what the API reported; with no token counts the guard charges the
                # whole hold (overcount, never undercount).
                if slot is not None and e.usage is not None and e.usage.input_tokens > 0:
                    slot.actual_usd = compute_cost(request.model, e.usage)
                raise
            usage = with_cost(request.model, response.usage)
            if slot is not None:
                slot.actual_usd = usage.cost_usd
        with self._totals_lock:
            self.totals.fresh += usage
            self.totals.n_fresh += 1
        logger.info(
            "LLM done %s: %d in / %d out tokens, $%.4f, stop_reason=%s",
            request.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
            response.stop_reason,
        )
        return response, usage

    def structured[T: BaseModel](
        self, *, system: str, user: str, model: str, schema: type[T], cache_tag: str
    ) -> tuple[T, TokenUsage]:
        """One structured call, served from the disk cache when possible.

        Raises LLMRefusalError, LLMTruncatedError, LLMMalformedOutputError, LLMAPIError,
        MissingAPIKeyError.
        Failed attempts are not cached, but their spend is counted in `totals.fresh`."""
        key = cache_key(cache_tag, model, schema, system, user, self.settings.llm_effort)
        cached = self._from_cache(key, schema)
        if cached is not None:
            return cached

        request = TransportRequest(
            model=model,
            system=system,
            user=user,
            schema=schema,
            max_tokens=self.settings.llm_max_tokens,
            effort=self.settings.llm_effort,
        )
        response, usage = self._send(request, cache_tag)
        if response.stop_reason == "max_tokens" and self.settings.llm_retry_truncated:
            logger.warning(
                "LLM %s hit max_tokens %d; retrying once at 2x", model, request.max_tokens
            )
            request = replace(request, max_tokens=2 * request.max_tokens)
            response, usage = self._send(request, cache_tag)

        check_stop_reason(response.stop_reason, model, usage)
        parsed = response.parsed
        if parsed is None or not isinstance(parsed, schema):
            detail = response.error or "no parsed output"
            raise LLMMalformedOutputError(f"{model} returned malformed output: {detail}", usage)

        self.cache.put_json(
            key,
            {
                "parsed": parsed.model_dump(mode="json"),
                "usage": usage.model_dump(mode="json"),
                "model": model,
                "schema": schema.__name__,
                "cache_tag": cache_tag,
                "stop_reason": response.stop_reason,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return parsed, usage
