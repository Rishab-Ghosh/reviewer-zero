"""Models local to the eval workstream (not shared across stages)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from reviewer.models import (
    ClaimGrade,
    CorpusEntry,
    Failure,
    Flag,
    GoldClaims,
    PaperClaims,
    ParsedPaper,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimilarityHit(_Model):
    """A claim's highest TF-IDF cosine to another claim of the same paper."""

    claim_index: int
    other_index: int | None
    """0-based index of the most similar other claim; None when the paper has one claim."""
    score: float


class NumberCheck(_Model):
    """One number from a gold `result` or `setting` field, looked up in the extracted claims."""

    number: str
    """The token as printed in the gold result, e.g. '−0.016' or '48.6%'."""
    cores: list[str]
    """Numeric cores that were searched for, e.g. ['0.016'] or ['48.6']."""
    found: bool
    field: Literal["result", "setting"] = "result"
    """Which gold field the number came from."""


class GoldAlignment(_Model):
    """One gold claim next to its closest extracted claim."""

    gold_index: int
    closest_index: int | None
    """0-based index of the extracted claim with the highest cosine; None if no claims."""
    score: float
    numbers: list[NumberCheck] = Field(default_factory=list)


class GoldComparison(_Model):
    alignments: list[GoldAlignment]

    def _checks(self, fields: tuple[str, ...]) -> list[NumberCheck]:
        return [n for a in self.alignments for n in a.numbers if n.field in fields]

    @property
    def n_numbers(self) -> int:
        """Numbers in gold `result` fields (the primary recall metric)."""
        return len(self._checks(("result",)))

    @property
    def n_found(self) -> int:
        return sum(n.found for n in self._checks(("result",)))

    @property
    def recall(self) -> float | None:
        """Result numbers found / result numbers; None when gold results have no numbers."""
        return self.n_found / self.n_numbers if self.n_numbers else None

    @property
    def n_numbers_all(self) -> int:
        """Numbers in gold `result` and `setting` fields."""
        return len(self._checks(("result", "setting")))

    @property
    def n_found_all(self) -> int:
        return sum(n.found for n in self._checks(("result", "setting")))

    @property
    def recall_all(self) -> float | None:
        return self.n_found_all / self.n_numbers_all if self.n_numbers_all else None


class PaperRecord(_Model):
    """Everything the report knows about one paper."""

    entry: CorpusEntry
    parsed: ParsedPaper | None = None
    claims: PaperClaims | None = None
    gold: GoldClaims | None = None
    failures: list[Failure] = Field(default_factory=list)
    """Latest failure per stage for this paper."""
    load_errors: list[str] = Field(default_factory=list)
    """Output files that exist but could not be read or validated."""
    flags: list[Flag] = Field(default_factory=list)
    gold_private: bool = False
    """True when the gold came from data/private/gold/ or has source author/lab_summary.
    Private gold never appears in the public report."""
    similarities: list[SimilarityHit] = Field(default_factory=list)
    gold_comparison: GoldComparison | None = None

    @property
    def is_own(self) -> bool:
        return self.entry.tier.value == "own" or self.entry.paper_id.startswith("own-")

    @property
    def title(self) -> str:
        if self.parsed is not None and self.parsed.title:
            return self.parsed.title
        if self.entry.title:
            return self.entry.title
        if self.claims is not None and self.claims.title:
            return self.claims.title
        return self.entry.paper_id


class GradesFile(_Model):
    """Contents of reports/private/grades.yaml."""

    grades: list[ClaimGrade] = Field(default_factory=list)
    stale: list[ClaimGrade] = Field(default_factory=list)
    """Filled-in grades whose claim no longer exists in the current extraction output."""


class ScoreRow(_Model):
    group: str
    n_claims: int
    n_graded: int
    mean_specificity: float | None = None
    mean_correctness: float | None = None


class SummaryTable(_Model):
    by_type: list[ScoreRow]
    by_tier: list[ScoreRow]
    overall: ScoreRow
    n_papers: int
    n_total: int
    n_graded: int
    """Claims with a specificity or correctness score."""
    gold_graded: int
    """Graded claims in papers that have a gold file (the gold-match denominator)."""
    gold_matched: int
    """Of those, claims with matches_gold set."""
    gold_match_rate: float | None = None


class ParseQualityRow(_Model):
    """One line of `reviewer eval parse-quality`."""

    paper_id: str
    tier: str
    n_tables: int
    lost_tables: list[str] = Field(default_factory=list)
    suspicious_authors: list[str] = Field(default_factory=list)
