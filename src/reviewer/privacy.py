"""Pre-commit guard: nothing derived from data/private/ or data/papers/own/ may be committed.

Blocks (1) staged paths under private locations or own-paper outputs, and (2) staged report/data
files whose text contains an own-paper id or any sentence from private gold or critiques.
Run by pre-commit with the staged file names as arguments.
"""

import logging
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from reviewer.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

PRIVATE_PATH_RE = re.compile(
    r"^(data/private/|data/papers/own/.+\.pdf$|reports/private/|data/failures\.jsonl$"
    r"|data/(parsed|claims|claims_strong|claims_v\d+)/own-)"
)
OWN_ID_RE = re.compile(r"\bown-[a-z0-9-]+-[0-9a-f]{8}\b")
# Only generated artifacts are content-checked; tests use synthetic own ids on purpose.
CONTENT_CHECKED_RE = re.compile(r"^(reports/|data/|STATUS\.md$|README\.md$)")
MIN_SNIPPET_CHARS = 50
MIN_SNIPPET_WORDS = 8
"""Short fragments (e.g. a list of public benchmark names) are not distinctive enough to be
private; a real private sentence clears both limits."""


def ensure_private(path: Path) -> None:
    """Refuse to write a private report anywhere that could be committed."""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return  # outside the repo (e.g. tests' tmp dirs)
    if not PRIVATE_PATH_RE.search(rel):
        raise RuntimeError(f"refusing to write private output to non-private path {rel}")


def write_private(path: Path, text: str) -> None:
    ensure_private(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote %s", path)


def append_private(path: Path, line: str) -> None:
    ensure_private(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def private_path_violations(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if PRIVATE_PATH_RE.search(p)]


def _strings(node: object) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def private_snippets(private_dir: Path) -> list[str]:
    """Normalized sentences (>= MIN_SNIPPET_CHARS) from private gold and critique files."""
    snippets: set[str] = set()
    for sub in ("gold", "critiques"):
        for f in sorted((private_dir / sub).glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            for text in _strings(data):
                for piece in re.split(r"(?<=[.;:])\s+", " ".join(text.split())):
                    if len(piece) >= MIN_SNIPPET_CHARS and len(piece.split()) >= MIN_SNIPPET_WORDS:
                        snippets.add(piece)
    return sorted(snippets)


def public_corpus_text(parsed_dir: Path) -> str:
    """Whitespace-normalized text of every public (non-own) parsed paper. A private sentence
    that appears verbatim here is a quotation of a public paper, not private content."""
    parts: list[str] = []
    for f in sorted(parsed_dir.glob("*.json")):
        if not f.name.startswith("own-"):
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    return " ".join(" ".join(parts).split())


def drop_public(snippets: Sequence[str], public_text: str) -> list[str]:
    return [s for s in snippets if s not in public_text]


def content_violations(path: str, text: str, snippets: Sequence[str]) -> list[str]:
    if not CONTENT_CHECKED_RE.search(path):
        return []
    found = [f"{path}: own-paper id {m.group(0)}" for m in OWN_ID_RE.finditer(text)]
    flat = " ".join(text.split())
    found += [f"{path}: private text '{s[:50]}…'" for s in snippets if s in flat]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    problems = [f"{p}: private path" for p in private_path_violations(paths)]
    settings = get_settings()
    snippets = drop_public(
        private_snippets(settings.private_dir), public_corpus_text(settings.parsed_dir)
    )
    for p in paths:
        path = Path(p)
        if not path.is_file() or path.suffix.lower() not in {".md", ".yaml", ".yml", ".json"}:
            continue
        problems += content_violations(
            p, path.read_text(encoding="utf-8", errors="ignore"), snippets
        )
    for problem in problems:
        logger.error("privacy guard: %s", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
