"""Logging configuration, applied once by each entry point (CLI, MCP server)."""

import logging

MAX_RECORD_CHARS = 2000
"""Records longer than this are dropped: a long message is the likeliest way paper text could
reach a log (BP §7; MASTER_PLAN task 5.5)."""


class DropLongRecords(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return len(record.getMessage()) <= MAX_RECORD_CHARS
        except Exception:  # a record whose args do not format is dropped too
            return False


def configure_logging(verbose: bool = False) -> None:
    """Log to stderr (the MCP stdio transport owns stdout) with the long-record filter on every
    root handler."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, DropLongRecords) for f in handler.filters):
            handler.addFilter(DropLongRecords())
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
