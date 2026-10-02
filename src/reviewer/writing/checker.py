"""One LLM call per paper on the full text, then code verification of every quote."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from reviewer.claims.prompts import Prompt
from reviewer.eval.judge import judge_context_full
from reviewer.llm.client import LLMClient, LLMMalformedOutputError
from reviewer.models import ParsedPaper, TokenUsage, WritingOutput, WritingReport
from reviewer.writing.verify import REFERENCES_RE, verify_output

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class WritingError(RuntimeError):
    def __init__(self, message: str, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def load_writing_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name.removesuffix('.md')}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file {path}")
    return Prompt(version=path.stem, text=path.read_text(encoding="utf-8").strip())


def writing_context(paper: ParsedPaper) -> str:
    """The judge_v2 context with related work kept (only references left out) and every
    figure caption added."""
    return judge_context_full(paper, skip=REFERENCES_RE, figures=True)


def writing_user_message(paper: ParsedPaper) -> str:
    return f"<paper>\n{writing_context(paper)}\n</paper>"


def check_paper(
    client: LLMClient,
    paper: ParsedPaper,
    *,
    private: bool,
    prompt: Prompt,
    model: str,
    max_retries: int,
) -> WritingReport:
    """Retries only malformed output; quotes that fail verification are dropped, not
    retried."""
    user = writing_user_message(paper)
    usage = TokenUsage()
    error: str | None = None
    for attempt in range(max_retries + 1):
        text = (
            user
            if error is None
            else f"{user}\n\n<previous_attempt_error>\nAttempt {attempt + 1}: {error}. Answer "
            "again.\n</previous_attempt_error>"
        )
        try:
            out, call_usage = client.structured(
                system=prompt.text,
                user=text,
                model=model,
                schema=WritingOutput,
                cache_tag=prompt.version,
            )
        except LLMMalformedOutputError as e:
            usage += e.usage or TokenUsage()
            error = str(e)
            logger.warning("writing %s retry %d: %s", paper.paper_id, attempt + 1, error)
            continue
        usage += call_usage
        v = verify_output(paper, out)
        logger.info(
            "writing %s: %d findings kept, %d dropped",
            paper.paper_id,
            len(v.findings),
            len(v.dropped),
        )
        return WritingReport(
            paper_id=paper.paper_id,
            title=paper.title,
            private=private,
            model=model,
            prompt_version=prompt.version,
            generated=datetime.now(UTC),
            abstract_trace=v.trace,
            contribution=v.contribution,
            related_work=v.related_work,
            findings=v.findings,
            dropped=v.dropped,
            token_usage=usage,
        )
    raise WritingError(f"malformed output after {max_retries} retries: {error}", usage)
