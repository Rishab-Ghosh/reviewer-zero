import logging

import anyio
from mcp.client import Client

from reviewer.logging_setup import MAX_RECORD_CHARS, DropLongRecords
from reviewer.mcp.server import NAME, build_server


def test_server_answers_initialize_in_memory():
    async def go() -> tuple[str, list]:
        async with Client(build_server()) as client:
            tools = await client.list_tools()
            return client.server_info.name, tools.tools

    name, tools = anyio.run(go)
    assert name == NAME
    assert isinstance(tools, list)


def test_long_log_records_are_dropped():
    f = DropLongRecords()

    def rec(msg: str, *args: object) -> logging.LogRecord:
        return logging.LogRecord("t", logging.INFO, __file__, 1, msg, args, None)

    assert f.filter(rec("short"))
    assert not f.filter(rec("x" * (MAX_RECORD_CHARS + 1)))
    assert not f.filter(rec("%s", "y" * (MAX_RECORD_CHARS + 1)))  # long via args
    assert not f.filter(rec("%d", "not a number"))  # unformattable
