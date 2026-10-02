"""Content-addressed on-disk cache for external calls (GROBID, LLM)."""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_key(*parts: str) -> str:
    """Stable key over several strings; parts are length-prefixed so boundaries matter."""
    h = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        h.update(f"{len(encoded)}:".encode())
        h.update(encoded)
    return h.hexdigest()


class DiskCache:
    """One directory per namespace; one file per key. Writes are atomic (tmp + rename)."""

    def __init__(self, root: Path, namespace: str, suffix: str) -> None:
        self.dir = root / namespace
        self.suffix = suffix
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}{self.suffix}"

    def get_text(self, key: str) -> str | None:
        p = self.path(key)
        if p.exists():
            logger.info("cache hit %s/%s", self.dir.name, key[:12])
            return p.read_text(encoding="utf-8")
        return None

    def put_text(self, key: str, value: str) -> Path:
        p = self.path(key)
        # Per-writer temp name: concurrent writers of the same key must not share it.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(p)
        return p

    def get_json(self, key: str) -> dict | None:
        text = self.get_text(key)
        return None if text is None else json.loads(text)

    def put_json(self, key: str, value: dict) -> Path:
        return self.put_text(key, json.dumps(value, indent=2, sort_keys=True, default=str))
