"""reviewer-zero configuration: environment variables only (no files, no prompts).

REVIEWER_ZERO_INDEX_URL, REVIEWER_ZERO_INDEX_KEY, REVIEWER_ZERO_CACHE; GROBID_URL and
ANTHROPIC_API_KEY are read by the pipeline's own Settings from the same environment.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from reviewer.config import Settings

DEFAULT_INDEX_URL: str | None = "https://api.tryreviewerzero.com"
"""The hosted index (public since 2026-10-02, DECISIONS P-13). REVIEWER_ZERO_INDEX_URL overrides
it, e.g. for a local index."""


class McpConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REVIEWER_ZERO_", extra="ignore")

    index_url: str | None = DEFAULT_INDEX_URL
    index_key: str | None = Field(default=None, repr=False)
    """Sent only as the X-API-Key header; never logged or put in a message."""
    cache: Path = Path.home() / ".cache" / "reviewer-zero"
    index_timeout_s: float = 30.0
    index_retries: int = 2
    """Retries for connection errors and 502/503/504 (and a 429 whose Retry-After is short)."""
    index_max_retry_after_s: float = 5.0
    """Longest Retry-After waited out inside a call; longer → IndexRateLimitedError. Chat tools
    keep it short; review_paper (minutes long, ~100+ index requests) waits up to 30 s."""

    def pipeline_settings(self) -> Settings:
        """The pipeline's Settings for MCP tools: no .env file is read (the user's environment
        only), public lookups cached under `cache/public`, outputs under `cache/work`."""
        return Settings(
            _env_file=None,
            index_url=self.index_url or "",
            index_api_key=self.index_key,
            index_timeout_s=self.index_timeout_s,
            data_dir=self.cache / "work",
            reports_dir=self.cache / "work" / "reports",
            cache_root=self.cache / "public",
        )
