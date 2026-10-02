import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from reviewer.index_gate import MAX_CONCURRENT_INDEX_REQUESTS
from reviewer.mcp.config import McpConfig
from reviewer.mcp.index import HostedIndex
from reviewer.mcp.prior_work import run_find_prior_work
from tests.test_mcp_prior_work import BEFORE, QUERIES, FakeIndexServer


def test_index_requests_in_flight_never_exceed_the_per_key_limit(tmp_path):
    """Three find_prior_work calls at once, each on its own HostedIndex (as parallel tool calls
    and review_paper's concurrent claims are): together they keep ≤ 4 requests in flight."""
    lock = threading.Lock()
    state = {"now": 0, "peak": 0, "n": 0}

    def slow_server():
        inner = FakeIndexServer()

        def handle(req: httpx.Request) -> httpx.Response:
            with lock:
                state["now"] += 1
                state["n"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.03)
            with lock:
                state["now"] -= 1
            return inner(req)

        return handle

    def one(i: int) -> int:
        idx = HostedIndex(
            McpConfig(index_url="https://index.test", index_key="k", cache=tmp_path / str(i)),
            transport=httpx.MockTransport(slow_server()),
            sleep=lambda s: None,
        )
        return len(run_find_prior_work(idx, QUERIES, BEFORE, k=40).candidates)

    with ThreadPoolExecutor(3) as pool:
        assert list(pool.map(one, range(3))) == [40, 40, 40]
    assert state["n"] > 3 * len(QUERIES)
    assert 1 < state["peak"] <= MAX_CONCURRENT_INDEX_REQUESTS == 4
