"""Append-only failure log (data/failures.jsonl). Failures are recorded, never hidden."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from reviewer.models import Failure, Stage

logger = logging.getLogger(__name__)


def record_failure(path: Path, stage: Stage, paper_id: str, error: str) -> Failure:
    failure = Failure(stage=stage, paper_id=paper_id, error=error, timestamp=datetime.now(UTC))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(failure.model_dump_json() + "\n")
    logger.error("[%s] %s failed: %s", stage, paper_id, error)
    return failure


def load_failures(path: Path) -> list[Failure]:
    if not path.exists():
        return []
    return [
        Failure.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
