"""Multi-query retrieval: an LLM turns a claim (or a whole paper) into one search query per
component, phrased like a prior paper's abstract would describe that component."""

import logging

from reviewer.claims.prompts import Prompt
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.models import NoveltyQuery, QueryPlan, SearchQuery, TokenUsage
from reviewer.novelty.rerank import render_claim

logger = logging.getLogger(__name__)

MIN_QUERIES = 6
MAX_QUERIES = 10


class QueryGenerationError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def render_paper(title: str, abstract: str) -> str:
    return f"<paper>\nTitle: {title}\nAbstract: {abstract or '[no abstract]'}\n</paper>"


def plan_problems(plan: QueryPlan) -> list[str]:
    problems = []
    n = len(plan.queries)
    if not MIN_QUERIES <= n <= MAX_QUERIES:
        problems.append(f"expected {MIN_QUERIES} to {MAX_QUERIES} queries, got {n}")
    texts = [" ".join(q.text.lower().split()) for q in plan.queries]
    if any(not t for t in texts):
        problems.append("every query needs non-empty text")
    if len(set(texts)) != len(texts):
        problems.append("queries must be distinct")
    return problems


def generate_queries(
    client: LLMClient,
    user: str,
    *,
    origin: str,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> tuple[list[SearchQuery], TokenUsage]:
    """Generated queries with ids '<c|p><i>'. Raises QueryGenerationError after retries."""
    usage = TokenUsage()
    problem: str | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            logger.warning("query generation retry %d/%d: %s", attempt, max_retries, problem)
        text = (
            user
            if problem is None
            else (
                f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: your previous answer "
                f"was rejected: {problem}. Answer again.\n</previous_attempt_error>"
            )
        )
        try:
            plan, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=QueryPlan,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            problem = str(e)
            continue
        usage += call_usage
        problem = "; ".join(plan_problems(plan)) or None
        if problem is None:
            prefix = origin[0]
            queries = [
                SearchQuery(
                    id=f"{prefix}{i}",
                    origin=origin,
                    kind=q.kind.value,
                    component=q.component,
                    text=q.text,
                )
                for i, q in enumerate(plan.queries)
            ]
            return queries, usage
    raise QueryGenerationError(
        f"query generation still invalid after {max_retries} retries: {problem}", usage
    )


def claim_queries(
    client: LLMClient, query: NoveltyQuery, *, prompt: Prompt, model: str, max_retries: int
) -> tuple[list[SearchQuery], TokenUsage]:
    return generate_queries(
        client,
        render_claim(query),
        origin="claim",
        prompt=prompt,
        model=model,
        max_retries=max_retries,
    )


def paper_queries(
    client: LLMClient, title: str, abstract: str, *, prompt: Prompt, model: str, max_retries: int
) -> tuple[list[SearchQuery], TokenUsage]:
    return generate_queries(
        client,
        render_paper(title, abstract),
        origin="paper",
        prompt=prompt,
        model=model,
        max_retries=max_retries,
    )
