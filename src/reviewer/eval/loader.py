"""Read the files other stages produce. The eval stage never imports the other workstreams.

Inputs: data/corpus.yaml, data/parsed/<id>.json, data/claims/<id>.json (``*.pass1.json``
ignored), data/gold/<id>.yaml, data/private/gold/<id>.yaml, data/failures.jsonl. Own papers
(``own-*``) are not listed in corpus.yaml; they are discovered from the parsed/claims/failure
files.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from reviewer.models import (
    CorpusEntry,
    Failure,
    GoldClaims,
    GoldSource,
    PaperClaims,
    PaperTier,
    ParsedPaper,
)

from .schemas import PaperRecord

logger = logging.getLogger(__name__)

OWN_PREFIX = "own-"
_CORPUS_FIELDS = frozenset(CorpusEntry.model_fields)


def load_corpus(path: Path) -> list[CorpusEntry]:
    """Entries of data/corpus.yaml in file order (unknown keys ignored)."""
    if not path.exists():
        logger.warning("corpus file %s not found; only own papers will be reported", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("papers", []) if isinstance(data, dict) else data
    entries: list[CorpusEntry] = []
    for item in items or []:
        if not isinstance(item, dict):
            logger.warning("skipping malformed corpus entry: %r", item)
            continue
        try:
            entries.append(
                CorpusEntry.model_validate({k: v for k, v in item.items() if k in _CORPUS_FIELDS})
            )
        except ValidationError as e:
            logger.warning("skipping invalid corpus entry %r: %s", item.get("paper_id"), e)
    return entries


def _load_json_dir[T: BaseModel](
    directory: Path, model: type[T], skip_suffixes: tuple[str, ...] = ()
) -> tuple[dict[str, T], dict[str, str]]:
    """Map stem -> model for every <id>.json in `directory`, plus stem -> error for bad files."""
    loaded: dict[str, T] = {}
    errors: dict[str, str] = {}
    if not directory.exists():
        return loaded, errors
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(skip_suffixes):
            continue
        paper_id = path.name.removesuffix(".json")
        try:
            loaded[paper_id] = model.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError) as e:
            msg = f"{directory.name}/{path.name} could not be loaded: {_first_line(str(e))}"
            logger.warning(msg)
            errors[paper_id] = msg
    return loaded, errors


def load_parsed(directory: Path) -> tuple[dict[str, ParsedPaper], dict[str, str]]:
    return _load_json_dir(directory, ParsedPaper)


def load_claims(directory: Path) -> tuple[dict[str, PaperClaims], dict[str, str]]:
    return _load_json_dir(directory, PaperClaims, skip_suffixes=(".pass1.json",))


def load_gold(directory: Path, known_ids: Iterable[str] | None = None) -> dict[str, GoldClaims]:
    """Load data/gold/*.yaml (other extensions, e.g. the .sample file, are ignored)."""
    known = set(known_ids) if known_ids is not None else None
    gold: dict[str, GoldClaims] = {}
    if not directory.exists():
        return gold
    for path in sorted(directory.glob("*.yaml")):
        stem = path.stem
        if known is not None and stem not in known:
            logger.warning("gold file %s does not match any known paper_id; ignored", path.name)
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and not isinstance(data.get("paper_id"), str):
                # An unquoted arXiv id is read by YAML as a float; the file name is authoritative.
                data["paper_id"] = stem
            g = GoldClaims.model_validate(data)
        except (yaml.YAMLError, ValidationError, OSError) as e:
            logger.warning("gold file %s is invalid: %s", path.name, _first_line(str(e)))
            continue
        if g.paper_id != stem:
            logger.warning(
                "gold file %s has paper_id %r; using the file name", path.name, g.paper_id
            )
        gold[stem] = g
    return gold


PRIVATE_GOLD_SOURCES = frozenset({GoldSource.author, GoldSource.lab_summary})


def load_all_gold(
    public_dir: Path, private_dir: Path, known_ids: Iterable[str] | None = None
) -> tuple[dict[str, GoldClaims], set[str]]:
    """Gold from data/gold/ and data/private/gold/, plus the ids whose gold is private.

    Gold is private when it was loaded from `private_dir` or its source is author or
    lab_summary. If both directories have a file for a paper, the private one wins.
    """
    known = list(known_ids) if known_ids is not None else None
    public = load_gold(public_dir, known)
    private = load_gold(private_dir, known)
    for pid in sorted(public.keys() & private.keys()):
        logger.warning(
            "gold for %s exists in both %s and %s; using the private one",
            pid,
            public_dir,
            private_dir,
        )
    gold = {**public, **private}
    private_ids = set(private) | {
        pid for pid, g in public.items() if g.source in PRIVATE_GOLD_SOURCES
    }
    for pid in sorted(set(public) - set(private)):
        if public[pid].source in PRIVATE_GOLD_SOURCES:
            logger.warning(
                "gold %s in %s has source %s; treating it as private (it belongs in %s)",
                pid,
                public_dir,
                public[pid].source.value,
                private_dir,
            )
    return gold, private_ids


def load_failures_tolerant(path: Path) -> list[Failure]:
    """Every parseable line of data/failures.jsonl; malformed lines are logged and skipped."""
    if not path.exists():
        return []
    failures: list[Failure] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            failures.append(Failure.model_validate_json(line))
        except (ValidationError, ValueError) as e:
            logger.warning("failures.jsonl line %d unreadable: %s", n, _first_line(str(e)))
    return failures


def latest_failures(failures: Iterable[Failure]) -> list[Failure]:
    """The most recent failure per (paper_id, stage), in chronological order."""
    latest: dict[tuple[str, str], Failure] = {}
    for f in failures:
        key = (f.paper_id, f.stage)
        if key not in latest or f.timestamp >= latest[key].timestamp:
            latest[key] = f
    return sorted(latest.values(), key=lambda f: f.timestamp)


TIER_ORDER: tuple[PaperTier, ...] = (
    PaperTier.own,
    PaperTier.lab,
    PaperTier.ordinary,
    PaperTier.famous,
)
_TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}


def order_entries(entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Own (sorted by id), then lab, then ordinary, then famous; corpus order within a tier."""
    indexed = list(enumerate(entries))
    indexed.sort(
        key=lambda ie: (
            _TIER_RANK[ie[1].tier],
            ie[1].paper_id if ie[1].tier == PaperTier.own else "",
            ie[0],
        )
    )
    return [e for _, e in indexed]


def build_records(
    corpus: list[CorpusEntry],
    parsed: dict[str, ParsedPaper],
    claims: dict[str, PaperClaims],
    gold: dict[str, GoldClaims],
    failures: list[Failure],
    load_errors: dict[str, list[str]] | None = None,
    private_gold_ids: Iterable[str] = (),
) -> list[PaperRecord]:
    """One record per paper in report order. Pure.

    Papers are: every corpus entry, every own paper seen in any output or failure, and any
    other paper id that has outputs but is missing from corpus.yaml (reported as ordinary,
    with a note in its reason).
    """
    load_errors = load_errors or {}
    private_ids = set(private_gold_ids)
    latest = latest_failures(failures)
    entries = list(corpus)
    known = {e.paper_id for e in entries}
    seen_ids = [*parsed, *claims, *(f.paper_id for f in latest), *load_errors]
    for pid in dict.fromkeys(seen_ids):
        if pid in known:
            continue
        known.add(pid)
        if pid.startswith(OWN_PREFIX):
            entries.append(CorpusEntry(paper_id=pid, tier=PaperTier.own))
        else:
            logger.warning("paper %s has outputs but is not in corpus.yaml", pid)
            entries.append(
                CorpusEntry(
                    paper_id=pid, tier=PaperTier.ordinary, reason="(not listed in corpus.yaml)"
                )
            )
    return [
        PaperRecord(
            entry=e,
            parsed=parsed.get(e.paper_id),
            claims=claims.get(e.paper_id),
            gold=gold.get(e.paper_id),
            gold_private=e.paper_id in gold and e.paper_id in private_ids,
            failures=[f for f in latest if f.paper_id == e.paper_id],
            load_errors=load_errors.get(e.paper_id, []),
        )
        for e in order_entries(entries)
    ]


def _first_line(s: str, limit: int = 300) -> str:
    """Collapse a (possibly multi-line) error message to one line of at most `limit` chars."""
    one = " | ".join(line.strip() for line in s.strip().splitlines() if line.strip())
    return one if len(one) <= limit else one[: limit - 1] + "…"
