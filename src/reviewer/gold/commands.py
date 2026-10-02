"""`reviewer gold build`: write gold claims and critiques. No LLM is involved anywhere.

- Author gold  -> data/private/gold/<own id>.yaml     (source: author)
- Lab gold     -> data/private/gold/<arxiv id>.yaml   (source: lab_summary)
- Critiques    -> data/private/critiques/<id>.yaml    (never fed to extraction)
- Famous gold  -> data/gold/<arxiv id>.yaml           (source: paper_contributions, public)
"""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from reviewer.cache import sha256_file
from reviewer.config import Settings, get_settings
from reviewer.corpus.corpus import load_corpus
from reviewer.corpus.ids import own_paper_id
from reviewer.gold.author import author_gold
from reviewer.gold.famous import famous_gold
from reviewer.gold.review import ReviewSpec, build_from_review
from reviewer.models import PaperTier, ParsedPaper

logger = logging.getLogger(__name__)

SPEC_FILE = "gold_src/spec.yaml"


def write_yaml(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        model.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
        width=100_000,
    )
    path.write_text(text, encoding="utf-8")
    logger.info("wrote %s", path)


def build_private(settings: Settings) -> None:
    spec_path = settings.private_dir / SPEC_FILE
    if not spec_path.exists():
        logger.warning("no private spec at %s; skipping author/lab gold and critiques", spec_path)
        return
    spec = ReviewSpec.model_validate(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    for a in spec.author_gold:
        own_pdf = settings.own_dir / a.own_pdf
        paper_id = own_paper_id(own_pdf, sha256_file(own_pdf))
        gold = author_gold(
            settings.private_dir / a.gold_pdf,
            paper_id,
            a.author,
            setting_overrides=a.setting_overrides,
            override_note=a.override_note,
        )
        write_yaml(settings.private_gold_dir / f"{paper_id}.yaml", gold)
        unresolved = [c for c in gold.claims if c.notes and "UNRESOLVED" in c.notes]
        logger.info(
            "author gold %s: %d claims, %d with unresolved 'As above'",
            paper_id,
            len(gold.claims),
            len(unresolved),
        )
    golds, critiques = build_from_review(settings.private_dir / spec.review_pdf, spec)
    for g in golds:
        write_yaml(settings.private_gold_dir / f"{g.paper_id}.yaml", g)
    for c in critiques:
        write_yaml(settings.critiques_dir / f"{c.paper_id}.yaml", c)
    for pid, why in spec.skipped_gold.items():
        logger.info("lab gold skipped %s: %s", pid, why)
    for pid, why in spec.no_critiques.items():
        logger.info("no critiques for %s: %s", pid, why)


def build_famous(settings: Settings) -> list[str]:
    """Returns famous paper ids with no explicit contributions statement."""
    without: list[str] = []
    for entry in load_corpus(settings):
        if entry.tier != PaperTier.famous:
            continue
        parsed = settings.parsed_dir / f"{entry.paper_id}.json"
        if not parsed.exists():
            logger.warning("famous gold: %s not parsed; skipped", entry.paper_id)
            continue
        paper = ParsedPaper.model_validate_json(parsed.read_text(encoding="utf-8"))
        if not paper.contributions:
            without.append(entry.paper_id)
        write_yaml(settings.gold_dir / f"{entry.paper_id}.yaml", famous_gold(paper))
    logger.info("famous papers without a contributions statement: %s", ", ".join(without))
    return without


def run_gold_build(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    build_private(settings)
    build_famous(settings)
