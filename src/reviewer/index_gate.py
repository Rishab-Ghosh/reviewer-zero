"""Process-wide cap on concurrent requests to the hosted index.

The public index allows each API key at most MAX_CONCURRENT_INDEX_REQUESTS requests in flight.
Every keyed index client in this process (the MCP tools' `HostedIndex`, `review_paper`'s novelty
step, `check_citations`' resolver) sends through one shared semaphore, so tools running in
parallel in one MCP server still stay under the limit. A slot is held until the response body
has been read, so the server never sees more requests in flight than slots.
"""

import threading

import httpx

MAX_CONCURRENT_INDEX_REQUESTS = 4
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INDEX_REQUESTS)


class IndexGateTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with _slots:
            response = self.inner.handle_request(request)
            try:
                response.read()
            except BaseException:
                response.close()
                raise
            return response

    def close(self) -> None:
        self.inner.close()
