"""`reviewer-zero`: an MCP server over stdio for Claude Code and Claude Desktop."""

import logging
import queue
from collections.abc import Callable
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import anyio

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from reviewer.logging_setup import configure_logging
from reviewer.mcp import descriptions
from reviewer.mcp.citations_tool import run_check_citations
from reviewer.mcp.config import McpConfig
from reviewer.mcp.format_tool import StageId, VenueId, run_check_format
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.lookups import run_citations, run_paper
from reviewer.mcp.prior_work import QueryError, run_find_prior_work
from reviewer.mcp.quotes_tool import run_verify_quotes
from reviewer.mcp.review_tool import run_review_paper
from reviewer.models import (
    CitationCheckResult,
    CitationsResult,
    FindPriorWorkResult,
    FormatCheckResult,
    PaperRecord,
    ReviewPaperResult,
    StepResult,
    VerifyQuotesResult,
)
from reviewer.novelty.index_client import IndexClientError

logger = logging.getLogger(__name__)

NAME = "reviewer-zero"
INSTRUCTIONS = (
    "Reviewer Zero checks an ML paper before submission, on the user's machine. "
    "Never give the paper a score or an accept/reject verdict, and never rewrite the user's text."
)


def package_version() -> str:
    try:
        return version("reviewer-zero")
    except PackageNotFoundError:
        return "0+unknown"


def default_index() -> HostedIndex:
    return HostedIndex(McpConfig())


def build_server(index_factory: Callable[[], HostedIndex] = default_index) -> MCPServer:
    """`index_factory` is called per tool call, so a missing URL or key is reported to the
    model as a tool error instead of preventing the server from starting."""
    server = MCPServer(name=NAME, instructions=INSTRUCTIONS, version=package_version())

    async def run[T](fn: Callable[[], T]) -> T:
        """Blocking work off the event loop. Deliberate, user-facing errors (input fixes,
        key/quota/URL problems) reach the model as ToolError; anything else stays a generic
        error and its text stays on the server."""
        try:
            return await anyio.to_thread.run_sync(fn)
        except (QueryError, IndexClientError) as e:
            raise ToolError(str(e)) from e

    @server.tool(name="find_prior_work", description=descriptions.FIND_PRIOR_WORK)
    async def find_prior_work(
        queries: list[str], before: date | None = None, k: int = 40
    ) -> FindPriorWorkResult:
        return await run(lambda: run_find_prior_work(index_factory(), queries, before, k))

    @server.tool(name="paper", description=descriptions.PAPER)
    async def paper(key: str) -> PaperRecord:
        return await run(lambda: run_paper(index_factory(), key))

    @server.tool(name="citations", description=descriptions.CITATIONS)
    async def citations(
        key: str, direction: Literal["cites", "cited_by"] = "cites", limit: int = 50
    ) -> CitationsResult:
        return await run(lambda: run_citations(index_factory(), key, direction, limit))

    @server.tool(name="check_format", description=descriptions.CHECK_FORMAT)
    async def check_format(pdf_path: str, venue: VenueId, stage: StageId) -> FormatCheckResult:
        return await run(lambda: run_check_format(pdf_path, venue, stage))

    @server.tool(name="check_citations", description=descriptions.CHECK_CITATIONS)
    async def check_citations(pdf_path: str) -> CitationCheckResult:
        return await run(lambda: run_check_citations(pdf_path))

    @server.tool(name="verify_quotes", description=descriptions.VERIFY_QUOTES)
    async def verify_quotes(pdf_path: str, quotes: list[str]) -> VerifyQuotesResult:
        return await run(lambda: run_verify_quotes(pdf_path, quotes))

    @server.tool(name="review_paper", description=descriptions.REVIEW_PAPER)
    async def review_paper(
        pdf_path: str,
        venue: VenueId,
        ctx: Context,
        confirm: bool = False,
        novelty: Literal["lite", "full"] = "full",
    ) -> ReviewPaperResult:
        """Steps finish in worker threads; their results go through a queue that this task
        drains into MCP progress notifications (one per finished step)."""
        finished: queue.SimpleQueue[StepResult] = queue.SimpleQueue()
        stop = anyio.Event()

        async def pump() -> None:
            n = 0
            while not (stop.is_set() and finished.empty()):
                while not finished.empty():
                    r = finished.get()
                    n += 1
                    await ctx.report_progress(n, 8, f"{r.step.value}: {r.status.value}")
                await anyio.sleep(0.5)

        # A ToolError raised inside the task group would leave it wrapped in an exception
        # group, which the SDK reports as a generic error; re-raise it after the group exits.
        outcome: list[ReviewPaperResult | ToolError] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(pump)
            try:
                outcome.append(
                    await run(
                        lambda: run_review_paper(
                            pdf_path, venue, confirm, on_step=finished.put, novelty=novelty
                        )
                    )
                )
            except ToolError as e:
                outcome.append(e)
            finally:
                stop.set()
        if isinstance(outcome[0], ToolError):
            raise outcome[0]
        return outcome[0]

    return server


def main() -> None:
    configure_logging()
    build_server().run()
