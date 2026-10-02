"""Opt-in live test of the critic pass (critic_v3) on the DeepSpeed GPU-overhead truism.

Skipped unless REVIEWER_LIVE_TESTS=1 and an API key is configured (env or .env). It calls
the real API (model from settings.llm_model) once or a few times (retries), with the cache
in a temp dir.
Run with: REVIEWER_LIVE_TESTS=1 uv run pytest tests/test_critic_live.py
"""

import logging
import os

import pytest

from reviewer.claims.extractor import run_critic
from reviewer.claims.prompt_builder import build_context, render
from reviewer.claims.prompts import load_prompt
from reviewer.config import Settings
from reviewer.llm.client import LLMClient
from reviewer.models import (
    Claim,
    ClaimsDraft,
    ClaimVerdict,
    Defect,
    ParsedPaper,
    Section,
    Table,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    os.environ.get("REVIEWER_LIVE_TESTS") != "1" or not Settings().anthropic_api_key,
    reason="live API test: set REVIEWER_LIVE_TESTS=1 and ANTHROPIC_API_KEY (env or .env)",
)

DEEPSPEED_TRUISM = (
    "Increasing the number of GPUs in DeepSpeed-based distributed ViT training increases "
    "communication overhead, as observed on the Nebula cluster, which must be balanced against "
    "speedup gains."
)
SPECIFIC = (
    "DeepSpeed ZeRO stage-2 data-parallel training of ViT-B/16 on ImageNet-1k reaches a 5.1x "
    "speedup on 8 GPUs over a single GPU while keeping top-1 accuracy within 0.3 points "
    "(81.2% vs 81.5%)."
)


def _paper() -> ParsedPaper:
    """A small synthetic DeepSpeed ViT scaling paper."""
    return ParsedPaper(
        paper_id="synthetic-deepspeed-vit",
        source_path="synthetic.pdf",
        title="Scaling Vision Transformer Training with DeepSpeed on the Nebula Cluster",
        abstract=(
            "We study data-parallel training of Vision Transformers (ViT-B/16) on ImageNet-1k "
            "with DeepSpeed ZeRO stage 2 on 1 to 8 NVIDIA A100 GPUs of the Nebula cluster. "
            "Training on 8 GPUs is 5.1x faster than on a single GPU, and top-1 accuracy stays "
            "within 0.3 points (81.2% vs 81.5%). As more GPUs are added, communication "
            "overhead grows and scaling efficiency falls from 94% at 2 GPUs to 64% at 8 GPUs."
        ),
        contributions=(
            "Our contributions are: (1) a benchmark of DeepSpeed ZeRO stage-2 ViT training on "
            "1 to 8 GPUs; (2) the observation that communication overhead increases with the "
            "number of GPUs and must be balanced against speedup."
        ),
        sections=[
            Section(
                heading="Introduction",
                number="1",
                paragraphs=[
                    "Vision Transformers are expensive to train. We measure how far DeepSpeed "
                    "data parallelism reduces ViT-B/16 training time on the Nebula cluster."
                ],
            ),
            Section(
                heading="Experimental Setup",
                number="3",
                paragraphs=[
                    "We train ViT-B/16 for 90 epochs on ImageNet-1k with DeepSpeed ZeRO stage "
                    "2, global batch size 1024, on 1, 2, 4, and 8 A100 GPUs."
                ],
            ),
            Section(
                heading="Results",
                number="4",
                paragraphs=[
                    "Table 2 reports wall-clock time and accuracy. Increasing the number of "
                    "GPUs increases communication overhead, which must be balanced against "
                    "speedup gains."
                ],
            ),
            Section(
                heading="Conclusion",
                number="5",
                paragraphs=[
                    "DeepSpeed gives a 5.1x speedup on 8 GPUs at nearly unchanged accuracy; "
                    "communication overhead grows with GPU count."
                ],
            ),
        ],
        tables=[
            Table(
                label="Table 2",
                caption="ViT-B/16 on ImageNet-1k with DeepSpeed ZeRO-2.",
                rows=[
                    ["GPUs", "Hours", "Speedup", "Scaling eff.", "Top-1 (%)"],
                    ["1", "61.2", "1.0x", "100%", "81.5"],
                    ["2", "32.6", "1.9x", "94%", "81.4"],
                    ["4", "18.0", "3.4x", "85%", "81.3"],
                    ["8", "12.0", "5.1x", "64%", "81.2"],
                ],
            )
        ],
    )


def _draft() -> ClaimsDraft:
    return ClaimsDraft(
        one_line_summary=(
            "DeepSpeed ZeRO-2 trains ViT-B/16 on ImageNet-1k 5.1x faster on 8 GPUs than on one, "
            "with top-1 accuracy within 0.3 points."
        ),
        claims=[
            Claim(
                statement=SPECIFIC,
                contribution_type="empirical_finding",
                method="DeepSpeed ZeRO stage-2 data parallelism for Vision Transformer training",
                problem="Reducing wall-clock training time of Vision Transformers",
                setting="ViT-B/16, ImageNet-1k, 90 epochs, 1 to 8 A100 GPUs on Nebula",
                result="5.1x speedup on 8 GPUs vs 1 GPU; top-1 81.2% vs 81.5%",
                novelty_assertion="A benchmark of DeepSpeed ZeRO-2 ViT training on 1 to 8 GPUs",
                evidence=["Table 2", "4 Results"],
                confidence=0.9,
            ),
            Claim(
                statement=DEEPSPEED_TRUISM,
                contribution_type="empirical_finding",
                method="DeepSpeed data-parallel training of Vision Transformers",
                problem="Scaling distributed ViT training across GPUs",
                setting="ViT-B/16 on the Nebula cluster, 1 to 8 GPUs",
                result="Communication overhead grows with the number of GPUs",
                novelty_assertion="The observation that overhead must be balanced against speedup",
                evidence=["4 Results"],
                confidence=0.6,
            ),
        ],
        keywords=["DeepSpeed", "ZeRO", "Vision Transformer", "data parallelism", "ImageNet-1k"],
    )


def test_critic_v3_rejects_deepspeed_truism(tmp_path):
    settings = Settings(data_dir=tmp_path)
    critic_prompt = load_prompt("critic_v3")
    draft = _draft()
    result = run_critic(
        LLMClient(settings),
        context_text=render(build_context(_paper())),
        draft=draft,
        model=settings.llm_model,
        critic_prompt=critic_prompt,
        cache_tag=f"live+{critic_prompt.version}",
        max_retries=settings.llm_max_retries,
    )
    critic = result.output
    for note in critic.review:
        logger.info("note %s: %s %s %s", note.draft_index, note.verdict, note.defects, note.reason)

    truism_notes = [n for n in critic.review if n.draft_index == 1]
    assert len(truism_notes) == 1
    note = truism_notes[0]
    assert note.verdict == ClaimVerdict.rejected
    assert Defect.truism in note.defects
    assert note.final_index is None
    assert DEEPSPEED_TRUISM not in [c.statement for c in critic.final.claims]
    # no final claim came from the truism (kept, rewritten, or re-added under another index)
    assert all(n.draft_index != 1 or n.final_index is None for n in critic.review)
