"""Data models shared by every stage.

Attribute docstrings are exported as JSON-schema descriptions (`use_attribute_docstrings`),
so for the claim models the docstrings *are* the schema the LLM sees. Edit them with care.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


# --------------------------------------------------------------------------- parsing


class Author(_Model):
    name: str
    """Full name as printed on the paper."""
    affiliation: str | None = None
    """Primary affiliation, if GROBID found one."""


class Section(_Model):
    heading: str
    """Section heading text without its number."""
    number: str | None = None
    """Section number as printed (e.g. '3.2'), if any."""
    level: int = 1
    """Nesting depth: 1 for top-level sections, 2 for subsections, and so on."""
    paragraphs: list[str] = Field(default_factory=list)
    """Paragraph texts in reading order."""
    fallback: bool = False
    """True when this section is not from GROBID but a page's pdftotext text spliced in because
    the parse recovered under 50% of that page."""
    fallback_page: int | None = None
    """For fallback sections: the 1-based PDF page the text comes from."""


class Figure(_Model):
    label: str
    """Figure label, e.g. 'Figure 3' (or GROBID's xml id if no label was printed)."""
    caption: str
    """Caption text."""


class Table(_Model):
    label: str
    """Table label, e.g. 'Table 2'."""
    caption: str
    """Caption text."""
    rows: list[list[str]] | None = None
    """Cell text row by row, when GROBID recovered cell structure."""
    raw_text: str | None = None
    """Flattened table text, used when no cell structure is available."""
    source: Literal["grobid", "docling"] = "grobid"
    """Which parser produced this table."""
    page: int | None = None


class Reference(_Model):
    ref_id: str
    """GROBID's bibliography id (e.g. 'b12'), the target of in-text citations."""
    raw_string: str
    """The reference exactly as printed in the bibliography."""
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None


class InTextCitation(_Model):
    section_heading: str
    """Heading of the section the citing sentence is in."""
    sentence: str
    """The full citing sentence."""
    ref_ids: list[str]
    """Bibliography ids cited in this sentence; ids may be unresolved (see parse_report)."""


class ParseReport(_Model):
    n_sections: int = 0
    n_paragraphs: int = 0
    n_figures: int = 0
    n_tables: int = 0
    n_references: int = 0
    n_in_text_citations: int = 0
    n_resolved_citations: int = 0
    """Citation ref_ids that point at an existing Reference."""
    n_unresolved_citations: int = 0
    has_contributions: bool = False
    warnings: list[str] = Field(default_factory=list)


class PageCoverage(_Model):
    page: int
    """1-based PDF page."""
    parse_chars: int
    """Non-whitespace characters the parse attributed to this page (via GROBID coordinates)."""
    parse_words: int
    pdftotext_chars: int
    """Non-whitespace characters in the PDF's text layer for this page (pdftotext)."""
    pdftotext_words: int
    coverage: float | None
    """parse_chars / pdftotext_chars; null when the text layer is empty."""
    flagged: bool
    """coverage < 0.5."""


class ParsedPaper(_Model):
    paper_id: str
    """arXiv id (e.g. '1706.03762') or 'own-<slug>-<sha8>' for user-supplied PDFs."""
    source_path: str
    pdf_sha256: str | None = None
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    in_text_citations: list[InTextCitation] = Field(default_factory=list)
    contributions: str | None = None
    """The paper's own contributions statement from the introduction, if it has one."""
    parse_report: ParseReport = Field(default_factory=ParseReport)
    abstract_source: Literal["arxiv", "grobid"] = "grobid"
    """arxiv: the abstract is arXiv's metadata abstract (source of truth for arXiv papers);
    grobid: GROBID's abstract (papers with no arXiv id, or arXiv lookup failed)."""
    grobid_abstract: str | None = None
    """GROBID's abstract, kept for reference when abstract_source is arxiv."""
    table_source: Literal["grobid", "docling", "none"] = "grobid"
    """Which parser supplied `tables` (docling when GROBID had none or fewer)."""
    pages: list[PageCoverage] = Field(default_factory=list)
    """Per-page text coverage of the parse against the PDF text layer (pdftotext)."""
    fallback_pages: list[int] = Field(default_factory=list)
    """Pages whose pdftotext text was spliced in as fallback sections."""


# --------------------------------------------------------------------------- corpus


class PaperTier(StrEnum):
    famous = "famous"
    ordinary = "ordinary"
    own = "own"
    lab = "lab"


class CorpusEntry(_Model):
    paper_id: str
    """arXiv id, or 'own-...' for PDFs in data/papers/own/."""
    tier: PaperTier
    reason: str = ""
    """One line explaining why this paper is in the corpus."""
    title: str | None = None
    pdf_path: str | None = None
    """Set for own papers; arXiv papers live at data/papers/arxiv/<id>.pdf."""


# --------------------------------------------------------------------------- claims


class ContributionType(StrEnum):
    new_method = "new_method"
    new_task_or_benchmark = "new_task_or_benchmark"
    new_dataset = "new_dataset"
    theoretical_result = "theoretical_result"
    empirical_finding = "empirical_finding"
    analysis_or_study = "analysis_or_study"
    system_or_tool = "system_or_tool"
    negative_result = "negative_result"


class Claim(_Model):
    statement: str
    """One sentence stating the claim. It must be specific enough that a reader could search
    for prior work with it and could in principle find it false. It names the method and the
    setting, and includes numbers when the paper reports them. A statement that could describe
    a hundred different papers is a failure."""
    contribution_type: ContributionType
    """The kind of contribution this claim represents."""
    method: str
    """What was built or done, named concretely (the paper's name for it plus what it is),
    e.g. 'Transformer: an encoder-decoder built only from multi-head self-attention'."""
    problem: str
    """The task or question the claim addresses."""
    setting: str
    """Where the claim holds: datasets, models, scale, domain, compute, and any other
    conditions that pin it down."""
    result: str
    """The outcome, with numbers and the baselines compared against whenever the paper gives
    them; otherwise the qualitative outcome."""
    novelty_assertion: str
    """What the authors say is new about this, in their own terms. This is what they assert,
    which may differ from what they did."""
    evidence: list[str]
    """Section headings, table labels, or figure labels in the paper that support the claim."""
    confidence: float = Field(ge=0.0, le=1.0)
    """Probability, 0 to 1, that the paper actually makes this claim as stated."""


class ClaimsDraft(_Model):
    """What the LLM returns in each pass; validated into PaperClaims afterwards."""

    one_line_summary: str
    """One sentence: what the paper does and what it finds, with its headline number if any."""
    claims: list[Claim]
    """Between 1 and 4 distinct claims the paper makes, most central first. Only as many as
    the paper actually supports: fewer strong claims beat padding."""
    keywords: list[str]
    """Between 5 and 10 search keywords: method names, task names, datasets, and technical
    terms a researcher would use to find prior work on this paper's claims."""


class Defect(StrEnum):
    truism = "truism"
    unfalsifiable = "unfalsifiable"
    studied_not_found = "studied_not_found"
    vague = "vague"
    unsupported = "unsupported"
    duplicate = "duplicate"
    missing_numbers = "missing_numbers"
    missing_method_or_setting = "missing_method_or_setting"
    overstated = "overstated"
    conflates_novelty = "conflates_novelty"


class ClaimVerdict(StrEnum):
    kept = "kept"
    rewritten = "rewritten"
    rejected = "rejected"
    added = "added"


class CriticNote(_Model):
    draft_index: int | None
    """0-based index of the pass-one claim this note is about; null only for an added claim."""
    final_index: int | None = None
    """0-based index of the claim in `final` that this note produced: the kept, rewritten, or
    added claim. Null only for a rejected claim."""
    verdict: ClaimVerdict
    """kept: the claim appears in `final` unchanged. rewritten: it appears in `final` changed.
    rejected: it does not appear in `final`. added: a claim in `final` that did not come from
    any pass-one claim. Never write an added note unless you actually added a claim."""
    defects: list[Defect] = Field(default_factory=list)
    """Every defect found in the draft claim (empty when kept). A claim with the truism defect
    must be rejected, never rewritten."""
    reason: str
    """Why: point to the specific text behind each defect, or say why it was already fine."""


class CriticOutput(_Model):
    """What the critic pass returns."""

    review: list[CriticNote]
    """Exactly one note per pass-one claim, in draft order, plus one note per added claim.
    The notes must describe exactly what changed between the draft and `final`."""
    final: ClaimsDraft
    """The corrected claim set that replaces the pass-one draft. It may hold fewer claims than
    the draft (at least 1)."""


class TokenUsage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    from_cache: bool = False
    """True when this response was served from the on-disk cache (no new spend)."""

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            from_cache=self.from_cache and other.from_cache,
        )


class SupportItem(_Model):
    field: Literal["statement", "setting", "result"]
    """Which field of the claim the element is in."""
    element: str
    """The number (with its unit, e.g. '41.9%') or the conclusion (e.g. 'does not scale with
    model size') exactly as the claim writes it."""
    quote: str
    """Verbatim text from the paper that supports it: a table row exactly as shown (cells
    separated by ' | ') or a sentence. Copy it character for character; use '...' only to skip
    words inside one sentence or row. A number element must appear in its quote."""
    source: str
    """Where the quote is: a table label (e.g. 'Table 2') or a section heading."""


class ClaimVerification(_Model):
    """What the per-claim verification call returns."""

    items: list[SupportItem]
    """One item for every number and every conclusion in the claim you return (the fixed
    claim when verdict is fixed). Empty when verdict is drop."""
    verdict: Literal["keep", "fixed", "drop"]
    """keep: every element is supported as written. fixed: you corrected elements (or removed
    unsupported ones) so every remaining element is supported. drop: the claim cannot be
    supported; prefer drop over keeping anything you cannot quote support for."""
    claim: Claim | None
    """The claim to keep: unchanged for keep, corrected for fixed, null for drop."""
    reason: str
    """One or two sentences: what was unsupported and what you changed, or why you dropped it."""


class VerificationRecord(_Model):
    """Outcome of the per-claim verification pass for one critic claim."""

    input_index: int
    """0-based index of the claim in the critic's final set."""
    outcome: Literal["kept", "fixed", "dropped"]
    reason: str
    items: list[SupportItem] = Field(default_factory=list)
    """The verified quotes (empty when dropped)."""
    attempts: int = 1
    problems: list[str] = Field(default_factory=list)
    """Code-check problems of the last attempt (why an unverifiable claim was dropped)."""


class ClaimGrounding(_Model):
    claim_index: int
    missing: list[str]
    """Numbers (as written in the claim) that appear nowhere in the paper's full text."""


class GroundingReport(_Model):
    """Code-level check that every number in each claim appears in the paper."""

    initial: list[ClaimGrounding]
    """Claims with ungrounded numbers after the first critic pass."""
    final: list[ClaimGrounding]
    """Claims still ungrounded after the critic was sent the missing numbers."""
    rounds: int = 0
    """Extra critic passes made to fix ungrounded numbers."""


class ExtractionMetadata(_Model):
    model: str
    prompt_version: str
    """Versions of both prompt files, e.g. 'extract_v1+critic_v1'."""
    timestamp: datetime
    token_usage: TokenUsage
    """Total over both passes and all retries."""
    pass_usage: dict[str, TokenUsage] = Field(default_factory=dict)
    """Usage per pass ('extract', 'critic')."""
    retries: int = 0
    critic_review: list[CriticNote] = Field(default_factory=list)
    grounding: GroundingReport | None = None
    """Number grounding result (None: check not run)."""
    verification: list[VerificationRecord] | None = None
    """Per-claim verification pass (critic_v5); None when not run."""
    critic_reconciled: list[int] = Field(default_factory=list)
    """Indices into `critic_review` of notes the critic marked 'kept' although it edited the
    claim; code relabelled them 'rewritten'."""
    critic_fallback: str | None = None
    """Set when the critic pass failed after all retries: the claims are the pass-one draft,
    not reviewed by the critic, and this is the critic's last error."""


class PaperClaims(_Model):
    paper_id: str
    title: str
    one_line_summary: str
    claims: list[Claim]
    keywords: list[str]
    extraction_metadata: ExtractionMetadata

    @field_validator("claims")
    @classmethod
    def _zero_to_four_claims(cls, v: list[Claim]) -> list[Claim]:
        if not 0 <= len(v) <= 4:
            raise ValueError(f"expected 0 to 4 claims, got {len(v)}")
        return v

    @model_validator(mode="after")
    def _empty_only_when_verification_dropped_all(self) -> "PaperClaims":
        """Zero claims only when the verification pass dropped every claim (an empty set is
        recorded rather than an unverified claim kept)."""
        if not self.claims:
            ver = self.extraction_metadata.verification
            if not ver or any(r.outcome != "dropped" for r in ver):
                raise ValueError("expected 1 to 4 claims (0 only when verification dropped all)")
        return self

    @field_validator("keywords")
    @classmethod
    def _five_to_ten_keywords(cls, v: list[str]) -> list[str]:
        if not 5 <= len(v) <= 10:
            raise ValueError(f"expected 5 to 10 keywords, got {len(v)}")
        return v


# --------------------------------------------------------------------------- gold + eval


class GoldSource(StrEnum):
    author = "author"
    """Written by the paper's author (the user's own paper)."""
    lab_summary = "lab_summary"
    """Copied verbatim from the lab's private literature review."""
    paper_contributions = "paper_contributions"
    """Copied verbatim from the paper's own contributions statement and abstract."""


class GoldClaim(_Model):
    statement: str
    """A reference claim, copied verbatim from the gold source."""
    contribution_type: ContributionType | None = None
    method: str | None = None
    problem: str | None = None
    setting: str | None = None
    result: str | None = None
    """Numbers in this field drive the number-recall check."""
    novelty_assertion: str | None = None
    notes: str | None = None
    """Provenance notes, e.g. where 'As above' was resolved or could not be."""


class GoldClaims(_Model):
    """Contents of data/gold/<paper_id>.yaml or data/private/gold/<paper_id>.yaml."""

    paper_id: str
    source: GoldSource
    source_document: str | None = None
    """File the claims were copied from."""
    author: str | None = None
    """Who wrote the gold claims."""
    claims: list[GoldClaim]


class CritiqueEntry(_Model):
    limitation: str
    """The limitation as written (left column of a limitations table, or one list item)."""
    explanation: str | None = None
    """The table's explanation column, when the source is a two-column table."""
    source_block: str
    """Which block of the source it came from, e.g. 'limitations table', 'Limitations list'."""


class Critiques(_Model):
    """Contents of data/private/critiques/<paper_id>.yaml. Never fed to extraction."""

    paper_id: str
    source_document: str
    entries: list[CritiqueEntry]


class FlagKind(StrEnum):
    vague_phrase = "vague_phrase"
    missing_numbers = "missing_numbers"
    too_long = "too_long"
    near_duplicate = "near_duplicate"


class Flag(_Model):
    claim_index: int
    kind: FlagKind
    detail: str


class ClaimGrade(_Model):
    """One entry in reports/grades.yaml. Score fields are filled in by hand."""

    paper_id: str
    tier: PaperTier
    claim_index: int
    contribution_type: ContributionType
    statement: str
    specificity: int | None = Field(default=None, ge=1, le=5)
    correctness: int | None = Field(default=None, ge=1, le=5)
    matches_gold: int | None = Field(default=None, ge=-1)
    """0-based index of the gold claim this extracted claim matches; -1 means graded and
    matches no gold claim; null means not graded yet. Only meaningful for papers with a gold
    file."""
    notes: str = ""


# --------------------------------------------------------------------------- novelty


class _ApiModel(BaseModel):
    """Responses from reviewer-index: tolerate fields we do not use."""

    model_config = ConfigDict(extra="ignore")


class IndexPaper(_ApiModel):
    """One paper returned by reviewer-index /search or /neighbors."""

    key: str
    """Bare arXiv id for arXiv papers; 'openalex:W...' / 's2:...' otherwise."""
    kind: str = "arxiv"
    title: str | None = None
    abstract: str | None = None
    published_date: date | None = None
    year: int | None = None
    similarity: float | None = None
    """Cosine similarity to the query text, when the endpoint was given one."""
    relation: Literal["cites", "cited_by"] | None = None
    """For /neighbors: 'cites' = the queried paper cites this one; 'cited_by' = this one cites
    the queried paper."""


class IndexSearchResponse(_ApiModel):
    results: list[IndexPaper]


class IndexNeighborsResponse(_ApiModel):
    results: list[IndexPaper]


class IndexPaperDetail(_ApiModel):
    """Bibliographic metadata of one paper: GET /paper/{key} or one entry of POST /papers."""

    key: str
    kind: str = "arxiv"
    title: str | None = None
    authors: list[str] | None = None
    """In order; null for out-of-corpus works whose authors the index does not store."""
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None


class ExpandSeedLink(_ApiModel):
    seed: str
    relation: Literal["cites", "cited_by"]
    """'cites': the seed cites the candidate; 'cited_by': the candidate cites the seed."""


class ExpandCandidate(_ApiModel):
    """One candidate from POST /expand (one-hop citation neighbor of any seed)."""

    key: str
    kind: str = "arxiv"
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    published_date: date | None = None
    seed_count: int
    """Distinct seeds linked to this candidate (either direction)."""
    seeds: list[ExpandSeedLink] = Field(default_factory=list)
    max_cosine: float | None = None
    """Max cosine similarity to any query text sent; null without an embedding."""


class ExpandResponse(_ApiModel):
    candidates: list[ExpandCandidate]
    total: int = 0
    """Eligible candidates before max_candidates truncation."""


class IndexHealth(_ApiModel):
    index_version: str
    """Changes whenever the index content changes; part of every index cache key."""


class CandidateSource(StrEnum):
    search_statement = "search_statement"
    search_method_problem = "search_method_problem"
    generated = "generated"
    """A generated component or whole-paper query (see Candidate.hits)."""
    expansion = "expansion"


class QueryKind(StrEnum):
    technique = "technique"
    sub_method = "sub_method"
    training_trick = "training_trick"
    objective = "objective"
    dataset_construction = "dataset_construction"
    problem_framing = "problem_framing"


class GeneratedQuery(_Model):
    kind: QueryKind
    """Which kind of component this query targets. technique: a core technique of the paper.
    sub_method: one sub-module or step of the method. training_trick: a training or inference
    trick (schedule, loss weighting, sampling, prompting detail). objective: the loss, reward or
    optimisation target. dataset_construction: how data, labels or a benchmark were built.
    problem_framing: the underlying problem as a prior paper would pose it."""
    component: str
    """A few words naming the component in generic terms (never the paper's own coined names)."""
    text: str
    """The search query: one or two sentences phrased the way a prior paper's abstract would
    describe this component, in generic technical vocabulary, without this paper's coined names,
    headline numbers or benchmark results."""


class QueryPlan(_Model):
    """What the query-generation call returns."""

    queries: list[GeneratedQuery]
    """Between 6 and 10 queries, one per distinct component, covering different kinds where the
    paper has them. No two queries target the same component."""


class SearchQuery(_Model):
    """One query actually run against the index."""

    id: str
    """'b0', 'b1' for the baseline queries; 'c<i>' claim-level and 'p<i>' paper-level
    generated queries."""
    origin: Literal["baseline", "claim", "paper"]
    kind: str
    """QueryKind value, or 'statement' / 'method_problem' for baseline queries."""
    component: str = ""
    text: str


class QueryHit(_Model):
    query_id: str
    rank: int
    """1-based rank in that query's results (after self/metadata filtering)."""
    similarity: float | None = None
    """Cosine similarity to that query."""


class Candidate(_Model):
    """A prior-work candidate for one claim, before LLM judgment."""

    paper: IndexPaper
    similarity: float
    """Best cosine similarity to any query that returned it (or to the claim, for expansion)."""
    sources: list[CandidateSource]
    via: list[str] = Field(default_factory=list)
    """For expansion candidates: keys of the search hits they were reached from."""
    hits: list[QueryHit] = Field(default_factory=list)
    """Every query that returned this candidate, with its rank there."""


class NoveltyLabel(StrEnum):
    same = "same"
    subsumes = "subsumes"
    close = "close"
    builds_on = "builds_on"
    """rerank_v3+: work the claim uses or extends as a foundation."""
    different = "different"


LABEL_ORDER: tuple[NoveltyLabel, ...] = (
    NoveltyLabel.same,
    NoveltyLabel.subsumes,
    NoveltyLabel.close,
    NoveltyLabel.builds_on,
    NoveltyLabel.different,
)


class CandidateJudgment(_Model):
    candidate_id: int
    """The number shown in the candidate's <candidate id="..."> tag."""
    label: NoveltyLabel
    """same: the prior work already makes essentially this claim (same method idea applied to
    the same problem, with the same kind of finding). subsumes: the prior work's contribution
    is more general and contains this claim as a special case or a direct instance. close: same
    problem and a similar method or finding, but it differs in a way that matters to the claim.
    different: none of the above."""
    reason: str
    """One sentence naming the specific overlap or the specific difference."""


class RerankOutput(_Model):
    """What the rerank call returns: one judgment per candidate shown."""

    judgments: list[CandidateJudgment]
    """Exactly one judgment per candidate, in the order shown."""


class CandidateJudgmentV2(CandidateJudgment):
    """rerank_v2: component-level overlap counts as close."""

    label: NoveltyLabel
    """same: the prior work already makes essentially this claim (same method idea applied to
    the same problem, with the same kind of finding). subsumes: the prior work's contribution
    is more general and contains this claim as a special case or a direct instance. close: the
    same problem with a similar method or finding that differs in a way that matters, OR the
    prior work already introduced or used one key component the claim depends on (a technique,
    sub-method, objective, training or inference method, or data construction), even if its
    overall problem or domain differs. different: none of the above."""


class RerankOutputV2(_Model):
    """What the rerank_v2 call returns: one judgment per candidate shown."""

    judgments: list[CandidateJudgmentV2]
    """Exactly one judgment per candidate, in the order shown."""


class ListwiseItem(_Model):
    candidate_id: int
    """The number shown in the candidate's <candidate id="..."> tag."""
    reason: str
    """One sentence: what a reviewer would cite this paper for, relative to the claim."""


class ListwiseRanking(_Model):
    """What the listwise ranking call returns."""

    ranking: list[ListwiseItem]
    """Every candidate shown, exactly once, ordered from the one a careful reviewer would most
    strongly cite as overlapping prior work to the one they would least."""


class CandidateJudgmentV3(CandidateJudgment):
    """rerank_v3: tighter same/subsumes, and builds_on for foundations."""

    label: NoveltyLabel
    """same: the claim's core contribution already exists in this work (the same central idea
    applied to the same problem, with the same kind of result); sharing a component is not
    enough. subsumes: the claim is a special case or a direct instance of this work's own
    contribution (the work already covers the claim's setting). close: overlaps the claim on the
    problem with a similar method or finding that differs in a way that matters, or already
    introduced one key component the claim depends on (single-component overlap is close, never
    same or subsumes). builds_on: the claim uses or extends this work as its foundation (a cited
    method it applies, a framework it critiques, reanalyzes, or improves); the claim itself
    presents it as prior work it starts from. different: none of the above."""


class RerankOutputV3(_Model):
    """What the rerank_v3 call returns: one judgment per candidate shown."""

    judgments: list[CandidateJudgmentV3]
    """Exactly one judgment per candidate, in the order shown."""


class SeverityOutput(_Model):
    """What the severity call returns for one claim."""

    score: int
    """0: no meaningful overlap. 1: prior work overlaps on components (techniques, objectives,
    data) but not on the core idea. 2: substantial overlap on the core idea, with a real
    difference that keeps the claim a contribution. 3: the core contribution already exists in
    prior work."""
    drivers: list[str]
    """Keys (as shown in the candidate tags) of the one or two works that drive the score; empty
    only for score 0."""
    reason: str
    """One line: which work overlaps with what, and why that is this severity."""


class SeverityResult(_Model):
    score: int
    drivers: list[str]
    reason: str
    prompt_version: str
    model: str


class NoveltyQuery(_Model):
    """The claim being checked (built from a Claim, or from an eval case)."""

    statement: str
    method: str = ""
    problem: str = ""
    setting: str = ""
    result: str = ""


class JudgedCandidate(_Model):
    candidate: Candidate
    label: NoveltyLabel
    reason: str
    listwise_rank: int | None = None
    """1-based position in the listwise ranking (only for candidates it ranked)."""
    listwise_reason: str | None = None


class NoveltyVerdict(StrEnum):
    novel = "novel"
    incremental = "incremental"
    already_done = "already_done"


class CandidateCounts(_Model):
    search_statement: int = 0
    """Unique hits of the statement query after filters."""
    search_method_problem: int = 0
    """Unique hits of the method + problem query after filters."""
    search_generated: int = 0
    """Unique hits of the generated queries."""
    n_queries: int = 0
    """Queries run (baseline + generated)."""
    search_union: int = 0
    expansion_new: int = 0
    """Candidates added by /neighbors expansion that no search query returned."""
    expansion_nodes: int = 0
    """Search hits expanded (up to the top 10)."""
    expansion_nodes_empty: int = 0
    """Expanded hits whose /neighbors returned nothing (no citation data)."""
    filtered_self: int = 0
    filtered_own_preprint: int = 0
    """Dropped because it is very likely the target's own earlier preprint under a different
    title (`is_own_preprint`): high title or abstract overlap, not newer than the target."""
    filtered_future: int = 0
    """Returned by the index despite `before` and dropped by the local guard (should be 0)."""
    filtered_undated: int = 0
    """Dropped because they have neither a published date nor a year to compare with the
    cutoff (only when there is a cutoff)."""
    filtered_no_metadata: int = 0
    """Dropped because the index withholds their title or abstract (known-bad metadata)."""
    pool: int = 0
    """Candidates sent to the LLM."""
    pool_from_expansion: int = 0
    """Of `pool`, how many only expansion found."""
    pool_from_generated_only: int = 0
    """Of `pool`, how many only generated queries found (no baseline query, no expansion)."""


class ClaimNovelty(_Model):
    claim_index: int
    query: NoveltyQuery
    verdict: NoveltyVerdict
    top: list[JudgedCandidate]
    """The 5 closest prior works."""
    ranked: list[JudgedCandidate]
    """Every judged candidate, closest first (label, then similarity)."""
    counts: CandidateCounts
    token_usage: TokenUsage
    queries: list[SearchQuery] = Field(default_factory=list)
    unpooled: list[Candidate] = Field(default_factory=list)
    """Retrieved but cut by the first-pass pool filter (not judged)."""
    query_usage: TokenUsage = Field(default_factory=TokenUsage)
    """Query-generation spend attributed to this claim (paper-level calls on the first claim)."""
    severity: SeverityResult | None = None
    """LLM severity of the overlap (0-3); when set, the verdict is derived from it."""


class PaperNovelty(_Model):
    paper_id: str
    title: str
    tier: str
    cutoff: str | None
    """Candidates from this date on were excluded (sent to the index as `before`); null = none."""
    cutoff_source: str | None = None
    """Where the cutoff came from, e.g. 'arXiv v1 date (index)'."""
    index_version: str
    model: str
    prompt_version: str
    timestamp: datetime
    claims: list[ClaimNovelty]
    token_usage: TokenUsage


class ReviewNamedWork(_ApiModel):
    title: str | None = None
    year: int | None = None


class CaseResolution(_ApiModel):
    status: str
    """'resolved' when the named work was found in the index, else 'unresolved'."""
    index_key: str | None = None
    matched_title: str | None = None
    matched_year: int | None = None
    after_cutoff: bool | None = None
    """The resolved work appeared after the submission's cutoff (the leakage filter drops it)."""


class IndexCaseRow(_ApiModel):
    """One line of reviewer-index data/novelty_eval/cases.jsonl: one prior work named in one
    review excerpt of one OpenReview submission."""

    case_id: str
    submission_id: str
    venue: str
    submission_title: str
    submission_abstract: str = ""
    named_prior_work: ReviewNamedWork
    resolution: CaseResolution
    cutoff_date: date | None = None
    """Leakage cutoff for this submission; the venue deadline is used when absent."""


class NegativeCaseRow(_ApiModel):
    """One line of reviewer-index negatives_dev.jsonl: an accepted submission whose reviews
    raise no overlap or novelty concern and praise its novelty."""

    submission_id: str
    venue: str
    submission_title: str
    submission_abstract: str = ""
    cutoff_date: date | None = None
    decision: str | None = None


class IclrReviewScore(_ApiModel):
    """One review's scores from reviewer-index data/openreview/decisions.jsonl."""

    rating: int
    confidence: int | None = None
    soundness: int | None = None
    presentation: int | None = None
    contribution: int | None = None


class IclrDecisionRow(_ApiModel):
    """One line of reviewer-index data/openreview/decisions.jsonl: ground truth (decision and
    reviewer ratings) for the meta-reviewer dev run (plan §10). `*_raw` string fields, `number`,
    `tier`, and `venue_label` are ignored."""

    submission_id: str
    venue: str
    year: int
    title: str
    final_status: str
    """accepted / rejected / withdrawn / desk_rejected."""
    decision: str | None = None
    n_reviews: int = 0
    ratings: list[int] = Field(default_factory=list)
    confidences: list[int] = Field(default_factory=list)
    reviews: list[IclrReviewScore] = Field(default_factory=list)


class PriorWorkRef(_Model):
    key: str | None = None
    """Index key (bare arXiv id or 'openalex:W...')."""
    title: str | None = None


class NoveltyEvalCase(_Model):
    """One submission with the prior work its reviewers named (grouped from IndexCaseRow)."""

    case_id: str
    """The OpenReview submission id."""
    venue: str = ""
    title: str
    abstract: str = ""
    cutoff_date: date | None = None
    """The venue's submission deadline: newer work is not prior work."""
    prior_work: list[PriorWorkRef]
    """Distinct named works resolved to an index key."""
    n_unresolved: int = 0
    """Named works not found in the index (cannot be retrieved; not in the recall denominator)."""
    n_after_cutoff: int = 0
    """Named works resolved but newer than the cutoff (dropped by the leakage filter by design;
    not in the recall denominator)."""


class NamedJudgment(_Model):
    """How the reranker judged one reviewer-named work that reached the pool."""

    key: str
    title: str | None = None
    abstract: str | None = None
    label: NoveltyLabel
    reason: str
    rank: int
    """1-based rank in the final ranked list."""
    listwise_rank: int | None = None


class NoveltyEvalCaseResult(_Model):
    case_id: str
    venue: str = ""
    n_named: int
    """Named works resolved in the index (the recall denominator)."""
    n_unresolved: int = 0
    n_after_cutoff: int = 0
    found_at: list[int | None]
    """1-based rank of each named work in the ranked list (null = not ranked)."""
    in_pool: list[bool]
    """Whether each named work reached the LLM pool at all."""
    in_union: list[bool] = Field(default_factory=list)
    """Whether any query (or expansion) retrieved each named work, before the pool filter."""
    found_by: list[list[str]] = Field(default_factory=list)
    """Per named work: ids of the queries that retrieved it ('x' = expansion)."""
    queries: list[SearchQuery] = Field(default_factory=list)
    named_judgments: list[NamedJudgment | None] = Field(default_factory=list)
    severity: SeverityResult | None = None
    """Paper-level severity (eval cases are one query per submission)."""
    """Per named work: its judgment if it reached the pool, else null."""
    recall_at_5: float
    recall_at_10: float = 0.0
    recall_at_20: float
    pool_recall: float
    union_recall: float = 0.0
    verdict: NoveltyVerdict | None = None
    error: str | None = None


# --------------------------------------------------------------------------- claim judge


class ClaimJudgment(_Model):
    claim_index: int
    """The number shown in the claim's <claim index="..."> tag."""
    specificity: int
    """1 to 5. 5: names the method, the setting, and a number, so it could be checked and
    searched for. 4: names method and setting but no number (or the number is vague). 3: names
    the method or the setting but not both, or leans on generic wording. 2: mostly generic, with
    one specific detail. 1: could describe a hundred papers."""
    specificity_reason: str
    """One line naming what is missing or generic. Required when specificity is 3 or below;
    empty otherwise."""
    correctness: int
    """1 to 5. 5: the paper says exactly this (method, setting, and numbers match the text or
    tables). 4: says this with a small imprecision (a rounded or slightly misattributed number,
    a setting detail off). 3: partly supported: the gist is there but a key element is
    overstated, mixed up, or not in the text shown. 2: mostly unsupported or distorted. 1: the
    paper does not say this."""
    correctness_reason: str
    """One line pointing to the mismatch or missing support. Required when correctness is 3 or
    below; empty otherwise."""


class JudgeOutput(_Model):
    """What the claim judge returns for one paper."""

    judgments: list[ClaimJudgment]
    """Exactly one judgment per claim shown, in the order shown."""


class PaperJudgment(_Model):
    paper_id: str
    tier: str
    model: str
    prompt_version: str
    claims_prompt_version: str
    """Extraction prompt versions of the claims judged (from their metadata)."""
    used_gold: bool
    judgments: list[ClaimJudgment]
    token_usage: TokenUsage


# --------------------------------------------------------------------------- citations

CitationSource = Literal["index", "openalex", "crossref", "arxiv"]
"""Where a reference was resolved, in lookup order (arXiv last: printed arXiv ids, then arXiv
title search, for preprints the other sources miss or mis-title)."""
AbstractSource = Literal["arxiv", "index", "openalex", "crossref"]


class WorkVersion(_Model):
    """One published form of a cited work (an arXiv preprint or a venue publication)."""

    source: Literal["index", "openalex", "crossref", "arxiv"]
    work_id: str
    """OpenAlex W id, DOI, or arXiv id, depending on `source`."""
    title: str
    year: int | None = None
    publication_date: str | None = None
    """ISO date (or year-month) when the source has one."""
    venue: str | None = None
    is_preprint: bool
    authors: list[str] = Field(default_factory=list)


class ResolvedWork(_Model):
    """A reference matched to a real paper. Built only from source records, never invented."""

    source: CitationSource
    title: str
    authors: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
    doi: str | None = None
    """Venue DOI when one is known (never the 10.48550 arXiv DOI)."""
    openalex_ids: list[str] = Field(default_factory=list)
    versions: list[WorkVersion] = Field(default_factory=list)
    title_similarity: float
    """Best fuzzy match of the resolved title inside the raw reference string, in [0, 1]."""
    abstract: str | None = None
    abstract_source: AbstractSource | None = None

    @property
    def years(self) -> list[int]:
        return sorted({v.year for v in self.versions if v.year is not None})

    @property
    def venue_versions(self) -> list[WorkVersion]:
        return [v for v in self.versions if not v.is_preprint]


class FlagType(StrEnum):
    unresolved = "unresolved"
    wrong_year = "wrong_year"
    wrong_authors = "wrong_authors"
    arxiv_now_published = "arxiv_now_published"
    duplicate = "duplicate"


class CitationFlag(_Model):
    type: FlagType
    ref_id: str
    detail: str
    """One-line description of what is wrong."""
    evidence: list[str] = Field(default_factory=list)
    """Lines a human can check: the raw string fragment, the resolved record and its id."""


class ReferenceCheck(_Model):
    ref_id: str
    raw_string: str
    """What the paper actually prints. All checks compare against this, not against the
    GROBID-consolidated fields."""
    cited_years: list[int] = Field(default_factory=list)
    cited_as_arxiv_only: bool = False
    resolved: ResolvedWork | None = None
    attempts: list[str] = Field(default_factory=list)
    """Resolution trace, one line per lookup ("openalex doi:… -> no match")."""
    flags: list[CitationFlag] = Field(default_factory=list)


class SupportVerdict(StrEnum):
    supported = "supported"
    contradicted = "contradicted"
    not_in_abstract = "not_in_abstract"
    cant_tell_from_abstract = "cant_tell_from_abstract"


class PropertyType(StrEnum):
    method_type = "method_type"
    actor_role = "actor_role"
    result_direction = "result_direction"
    numeric_value = "numeric_value"
    none = "none"


class SupportJudgment(_Model):
    """The judge's output for one cited reference in one citing sentence. Fields are filled in
    order; the verdict comes last and must follow from the fields before it."""

    ref_id: str
    """The reference id exactly as given in the input (e.g. "b12")."""
    attributed: str
    """What the citing sentence attributes to this reference, in one short clause."""
    abstract_usable: bool
    """True if the abstract is a readable abstract of the listed paper (whatever its topic);
    false only if it is empty, truncated before any content, garbled, not an abstract, or the
    abstract of a different paper than the listed title."""
    property_type: PropertyType
    """If the abstract states a property of the cited work that is incompatible with the
    attribution, which kind: method_type (the type of method), actor_role (who does what),
    result_direction (the direction of a result), numeric_value (a number for the same
    quantity). Otherwise none. Naming differences, subsets, versions, derived resources and
    different topics are always none."""
    sentence_span: str
    """The span of the citing sentence that makes the attribution, copied word for word."""
    abstract_span: str
    """The span of the abstract the verdict rests on (at most about 40 words), copied word for
    word. Empty when the abstract does not address the attribution."""
    explanation: str
    """One or two sentences. If property_type is not none, say how the two spans conflict."""
    verdict: SupportVerdict
    """Derived from the fields above. supported: abstract_span states or clearly implies the
    attribution. contradicted: only if property_type is not none and the two spans conflict on
    that property. not_in_abstract: a usable abstract that does not address the attribution.
    cant_tell_from_abstract: only if abstract_usable is false."""


class SupportBatch(_Model):
    judgments: list[SupportJudgment]
    """Exactly one judgment per reference given in the input."""


class CitationSupport(_Model):
    """A stored support judgment for one (sentence, reference) pair."""

    section_heading: str
    sentence: str
    ref_id: str
    verdict: SupportVerdict
    attributed: str = ""
    quote: str = ""
    """The judge's abstract_span (verbatim from the abstract)."""
    rationale: str = ""
    """The judge's explanation."""
    property_type: PropertyType | None = None
    sentence_span: str = ""
    judged_by_llm: bool
    """False when the verdict was assigned without an LLM (e.g. no abstract available)."""
    overridden_from: SupportVerdict | None = None
    """The LLM's verdict when code changed it: a supported/contradicted quote not found in the
    abstract (-> cant_tell), cant_tell on an abstract the judge called usable
    (-> not_in_abstract), or any verdict on an abstract it called unusable (-> cant_tell)."""
    override_reason: str | None = None
    verifier_verdict: SupportVerdict | None = None
    """For verdicts the first judge called contradicted: the verdict of an independent
    re-judgment of the same inputs (after the same code checks). Contradicted stands only if
    both agree."""


class CitationReport(_Model):
    paper_id: str
    title: str
    private: bool
    generated: datetime
    index_available: bool
    model: str
    prompt_version: str
    references: list[ReferenceCheck] = Field(default_factory=list)
    support: list[CitationSupport] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[str] = Field(default_factory=list)
    """Lookups or LLM calls that failed (also in data/failures.jsonl)."""

    @property
    def flags(self) -> list[CitationFlag]:
        return [f for r in self.references for f in r.flags]


# --------------------------------------------------------------------------- format


class Venue(StrEnum):
    iclr = "ICLR"
    neurips = "NeurIPS"
    icml = "ICML"


class SubmissionStage(StrEnum):
    submission = "submission"
    """Anonymous or 'under review': anonymization and submission page limits apply."""
    camera_ready = "camera_ready"
    """The PDF declares acceptance at a venue: camera-ready page limits apply."""
    preprint = "preprint"
    """No venue status found (or an arXiv preprint): anonymization findings are informational."""


class FormatSeverity(StrEnum):
    violation = "violation"
    warning = "warning"
    info = "info"


class FormatCheck(StrEnum):
    page_limit = "page_limit"
    author_block = "author_block"
    email = "email"
    self_citation = "self_citation"
    identifying_url = "identifying_url"
    acknowledgments = "acknowledgments"
    pdf_metadata = "pdf_metadata"
    required_section = "required_section"
    recommended_section = "recommended_section"
    unresolved_reference = "unresolved_reference"
    unreferenced_figure = "unreferenced_figure"
    unreferenced_table = "unreferenced_table"


class FormatFinding(_Model):
    check: FormatCheck
    severity: FormatSeverity
    detail: str
    venue: Venue | None = None
    page: int | None = None
    evidence: list[str] = Field(default_factory=list)
    source: str | None = None
    """URL of the official rule this finding applies, when it applies a venue rule."""


class VenuePages(_Model):
    venue: Venue
    submission_limit: int
    camera_ready_limit: int
    main_text_pages: float
    """Pages up to where the main text ends (fractional on the last page)."""
    pages_used: int
    """Pages the main text occupies (a partly filled last page counts as a page)."""
    ends_at: str | None
    """The heading that ends the main text for this venue, or None if none was found."""


class DeclaredVenue(_Model):
    venue: Venue | None
    """None for a venue outside ICLR/NeurIPS/ICML (e.g. a workshop) or when not found."""
    stage: SubmissionStage
    year: int | None = None
    text: str | None = None
    """The page-1 line that declares it."""


class FormatReport(_Model):
    paper_id: str
    title: str
    private: bool
    generated: datetime
    n_pages: int
    declared: DeclaredVenue
    pages: list[VenuePages] = Field(default_factory=list)
    findings: list[FormatFinding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- writing


class WritingCheck(StrEnum):
    abstract_overclaim = "abstract_overclaim"
    unclear_contribution = "unclear_contribution"
    undefined_term = "undefined_term"
    caption_not_standalone = "caption_not_standalone"
    related_work_list = "related_work_list"


class AbstractClaimVerdict(StrEnum):
    supported = "supported"
    overstated = "overstated"
    not_found = "not_found"
    unverifiable = "unverifiable"


class AbstractClaimTrace(_Model):
    abstract_span: str
    """One claim from the abstract, copied verbatim (a sentence or clause, no ellipses)."""
    result_span: str
    """Verbatim text from the paper body, a table, or a figure caption that is the closest
    evidence for this claim. Never from the abstract or title. Empty only if nothing in the
    paper shown bears on the claim."""
    gap: str
    """What the abstract claims that result_span does not show (a wider scope, a stronger
    word, a bigger number, a missing condition). Empty if result_span shows all of it."""
    verdict: AbstractClaimVerdict
    """supported: result_span shows the whole claim (gap empty). overstated: result_span shows
    a weaker or narrower result (gap says how). not_found: no result in the text shown bears on
    the claim. unverifiable: the evidence is plausibly only in a figure image or a part of the
    paper that is missing from the text shown."""


class ContributionVerdict(StrEnum):
    clear = "clear"
    vague = "vague"
    missing = "missing"


class ContributionCheck(_Model):
    intro_span: str
    """The introduction sentence that states the paper's contribution most directly, copied
    verbatim. If none does, the sentence that comes closest."""
    problem: str
    """What a reader still cannot tell about what is new after reading the introduction. Empty
    if the contribution is clear."""
    verdict: ContributionVerdict
    """clear: the introduction says what is new and what it achieves. vague: it says only
    something generic ("we study", "we propose a framework") without what is new or what it
    achieves. missing: no sentence states a contribution."""


class TermVerdict(StrEnum):
    defined_later = "defined_later"
    never_defined = "never_defined"


class TermUse(_Model):
    term: str
    """The term, symbol name, or acronym exactly as printed."""
    first_use_span: str
    """Verbatim text around the term's first use in the paper body (not the title or abstract),
    containing the term."""
    definition_span: str
    """Verbatim text where the paper later defines or expands the term. Empty if it never
    does."""
    verdict: TermVerdict


class CaptionCheck(_Model):
    label: str
    """The figure or table label as shown, e.g. 'Figure 3' or 'Table 2'."""
    caption_span: str
    """Verbatim text from that caption: the unclear part, or the whole caption if short."""
    missing: str
    """What a reader needs to understand the figure or table from its caption (and, for a
    table, its cells) but is not given: an undefined abbreviation, the metric or units, what
    bold or colours mean, which setting or dataset."""


class RelatedWorkVerdict(StrEnum):
    engages = "engages"
    mostly_lists = "mostly_lists"
    no_related_work = "no_related_work"


class RelatedWorkCheck(_Model):
    listing_spans: list[str]
    """Up to three sentences from the related-work discussion, copied verbatim, that only list
    or summarise prior work without relating it to this paper. Empty if there are none."""
    engaging_span: str
    """One verbatim sentence that contrasts prior work with this paper (what it lacks, how
    this paper differs or builds on it). Empty if there is none."""
    verdict: RelatedWorkVerdict
    """engages: most prior work discussed is related to this paper. mostly_lists: most of it
    is described without saying how this paper relates. no_related_work: the paper has no
    discussion of prior work."""


class WritingOutput(_Model):
    """What the writing checker returns for one paper."""

    abstract_claims: list[AbstractClaimTrace]
    """Every distinct claim the abstract makes about this paper's own method or results, in
    abstract order (not background sentences about the field)."""
    contribution: ContributionCheck
    undefined_terms: list[TermUse]
    """Paper-specific terms, acronyms, and notation used before they are defined, or never
    defined. Not standard terms a reader in the field knows. Empty if there are none."""
    captions: list[CaptionCheck]
    """Only the figures and tables whose caption is not readable on its own. Empty if every
    caption is."""
    related_work: RelatedWorkCheck


class WritingFinding(_Model):
    check: WritingCheck
    quote: str
    """Verbatim span from the paper, verified in code."""
    evidence: list[str] = Field(default_factory=list)
    """Further verified spans (the result an abstract claim was traced to, a later
    definition, other listing sentences)."""
    location: str
    """Where the quote was found (abstract, section heading, figure or table label)."""
    detail: str
    """The model's explanation."""


class DroppedFinding(_Model):
    check: WritingCheck
    reason: str
    """Why code rejected it (a quote not found verbatim, a term defined at first use, ...)."""
    item: str
    """The model's output for it, as JSON."""


class WritingReport(_Model):
    paper_id: str
    title: str
    private: bool
    model: str
    prompt_version: str
    generated: datetime
    abstract_trace: list[AbstractClaimTrace]
    """Abstract claims whose spans verified, with their verdicts."""
    contribution: ContributionCheck | None = None
    related_work: RelatedWorkVerdict | None = None
    findings: list[WritingFinding]
    dropped: list[DroppedFinding]
    token_usage: TokenUsage


# --------------------------------------------------------------------------- methodology


class ChecklistItemId(StrEnum):
    """Methodology checklist items (the prompt version decides which are asked). For every
    item, 'yes' means good practice."""

    baseline_tuning = "baseline_tuning"
    seeds_variance = "seeds_variance"
    significance = "significance"
    leakage = "leakage"
    compute_matched = "compute_matched"
    ablation_isolation = "ablation_isolation"
    data_disclosure = "data_disclosure"
    hparam_disclosure = "hparam_disclosure"
    abstract_support = "abstract_support"
    claims_support = "claims_support"
    evaluation_scope = "evaluation_scope"


class MethodsVerdict(StrEnum):
    yes = "yes"
    no = "no"
    unclear = "unclear"
    na = "na"


class ChecklistFinding(_Model):
    item: ChecklistItemId
    """Which checklist item this answers."""
    verdict: MethodsVerdict
    """yes: the paper meets the standard. no: it does not (a methodology problem). unclear: the
    text shown does not let you decide. na: the item does not apply to this paper (e.g. no
    pretrained-LLM or public-benchmark evaluation for leakage; no baselines at all)."""
    rationale: str
    """Two to four sentences: what the paper does on this point and why that earns the verdict.
    Name tables, sections, numbers."""
    quotes: list[str]
    """One to three passages copied character for character from the paper text shown (at least
    six words each; no paraphrase, no added words). Code checks every quote; a verdict with no
    quote found in the paper is downgraded to unclear. For an absence (e.g. no variance
    reported), quote the passage or table where it should be: the main result table caption or
    row, or the sentence stating the gain."""
    concern: str
    """For no or unclear: one sentence a strict PI would write in a review, specific to this
    paper (what is missing or wrong, and which claim it undermines). Empty for yes and na."""


class OtherConcern(_Model):
    concern: str
    """A methodology problem outside the checklist (e.g. evaluation on a single model, domain,
    or benchmark; a confound in the setup; a metric that does not measure the claim), as one
    specific sentence."""
    quotes: list[str]
    """One to three verbatim passages from the paper supporting the concern (same rules as for
    checklist quotes)."""


class MethodsOutput(_Model):
    """What the methodology checker returns for one paper."""

    findings: list[ChecklistFinding]
    """Exactly one finding per checklist item listed in the instructions, in that order."""
    other_concerns: list[OtherConcern]
    """Zero to five further methodology concerns, most serious first. Leave empty rather than
    padding with weak ones."""


class QuoteRepair(_Model):
    key: str
    """The key shown for the finding being repaired (an item id such as 'baseline_tuning', or
    'other:N')."""
    quotes: list[str]
    """One to three corrected passages copied character for character from the paper text.
    Empty if no passage in the text supports the finding."""


class QuoteRepairOutput(_Model):
    """Corrected quotes for findings whose quotes were not found in the paper."""

    repairs: list[QuoteRepair]
    """Exactly one entry per finding listed for repair."""


class QuoteCheck(_Model):
    quote: str
    verified: bool
    """True when code found the quote in the paper text (after normalisation)."""


class CheckedFinding(_Model):
    item: ChecklistItemId
    verdict: MethodsVerdict
    """Final verdict: the model's verdict, or unclear when no quote verified."""
    model_verdict: MethodsVerdict
    """What the model said before quote verification."""
    rationale: str
    concern: str
    quotes: list[QuoteCheck]
    downgraded: bool = False
    """True when the model's verdict was replaced by unclear for lack of a verified quote."""
    repaired: bool = False
    """True when the quotes went through the repair round."""
    violation_codes: list[str] = Field(default_factory=list)
    """Structured checker only: the verified violation codes the verdict was derived from."""
    agreement: str = ""
    """Two-run reviews, for `no` findings: 'confirmed' (both runs raised it) or 'possible'
    (one run did). Empty for single-run reviews and for items not raised."""


class CheckedConcern(_Model):
    concern: str
    quotes: list[QuoteCheck]
    verified: bool
    """At least one quote verified; unverified concerns count as unclear."""
    repaired: bool = False
    agreement: str = ""
    """Two-run reviews: 'confirmed' (a matching concern in both runs) or 'possible'."""


class PaperMethodsReview(_Model):
    paper_id: str
    tier: str
    model: str
    prompt_version: str
    effort: str
    findings: list[CheckedFinding]
    other_concerns: list[CheckedConcern]
    token_usage: TokenUsage
    runs: int = 1
    """Independent checker runs combined into this review (2 = confirmed/possible buckets)."""


class ViolationCode(StrEnum):
    """Fixed failure conditions, prefixed by the checklist item they belong to."""

    baseline_tuning__unequal_search = "baseline_tuning__unequal_search"
    baseline_tuning__defaults_or_copied_numbers = "baseline_tuning__defaults_or_copied_numbers"
    baseline_tuning__weak_or_dated_baselines = "baseline_tuning__weak_or_dated_baselines"
    seeds_variance__run_count_missing = "seeds_variance__run_count_missing"
    seeds_variance__variance_missing = "seeds_variance__variance_missing"
    significance__no_test_for_small_margin = "significance__no_test_for_small_margin"
    significance__small_test_set_no_test = "significance__small_test_set_no_test"
    leakage__tuned_on_test = "leakage__tuned_on_test"
    leakage__train_test_overlap = "leakage__train_test_overlap"
    leakage__pretrained_benchmark_no_check = "leakage__pretrained_benchmark_no_check"
    leakage__data_from_test_aware_model = "leakage__data_from_test_aware_model"
    compute_matched__more_samples_or_search = "compute_matched__more_samples_or_search"
    compute_matched__more_parameters = "compute_matched__more_parameters"
    compute_matched__more_training_or_data = "compute_matched__more_training_or_data"
    compute_matched__cost_not_reported = "compute_matched__cost_not_reported"
    ablation_isolation__no_ablations = "ablation_isolation__no_ablations"
    ablation_isolation__removed_together = "ablation_isolation__removed_together"
    ablation_isolation__credited_part_not_isolated = (
        "ablation_isolation__credited_part_not_isolated"
    )
    data_disclosure__table_dataset_unstated = "data_disclosure__table_dataset_unstated"
    data_disclosure__sizes_missing = "data_disclosure__sizes_missing"
    data_disclosure__subset_selection_unstated = "data_disclosure__subset_selection_unstated"
    data_disclosure__new_data_construction_missing = (
        "data_disclosure__new_data_construction_missing"
    )
    data_disclosure__baseline_inputs_unstated = "data_disclosure__baseline_inputs_unstated"
    hparam_disclosure__search_space_missing = "hparam_disclosure__search_space_missing"
    hparam_disclosure__selection_split_missing = "hparam_disclosure__selection_split_missing"
    hparam_disclosure__only_final_values = "hparam_disclosure__only_final_values"
    claims_support__broader_than_tested = "claims_support__broader_than_tested"
    claims_support__number_mismatch = "claims_support__number_mismatch"
    claims_support__marginal_or_inconsistent_gain = "claims_support__marginal_or_inconsistent_gain"
    claims_support__tradeoff_as_win = "claims_support__tradeoff_as_win"
    claims_support__qualitative_not_shown = "claims_support__qualitative_not_shown"
    claims_support__mechanism_untested = "claims_support__mechanism_untested"
    claims_support__no_experiment = "claims_support__no_experiment"
    evaluation_scope__few_datasets = "evaluation_scope__few_datasets"
    evaluation_scope__one_model_family_or_size = "evaluation_scope__one_model_family_or_size"
    evaluation_scope__synthetic_or_toy_only = "evaluation_scope__synthetic_or_toy_only"
    evaluation_scope__narrow_task = "evaluation_scope__narrow_task"
    evaluation_scope__scaling_unmeasured = "evaluation_scope__scaling_unmeasured"
    evaluation_scope__standard_benchmarks_missing = "evaluation_scope__standard_benchmarks_missing"

    @property
    def item(self) -> str:
        return self.value.split("__")[0]


class Violation(_Model):
    code: ViolationCode
    """The failure condition, from the list for this item only."""
    quote: str
    """One passage copied character for character from the paper text shown (at least six
    words) showing the failure; for an absence, the passage where it should appear."""
    explanation: str
    """One sentence, specific to this paper: what fails and which claim it undermines."""


class StructuredFinding(_Model):
    item: ChecklistItemId
    """Which checklist item this answers."""
    applies: bool
    """False only when the item cannot apply to this paper (e.g. no comparison to other methods
    at all for baseline_tuning; nothing evaluated could have seen test data for leakage)."""
    applies_quote: str
    """When applies is false: a verbatim passage showing why. Empty otherwise."""
    violations: list[Violation]
    """Every listed failure condition of this item that the text shows, each with its own quote.
    Empty when none holds. List only conditions the text establishes, not ones you suspect."""
    compliance_quotes: list[str]
    """Verbatim passages showing the paper meets the standard for this item (e.g. the sentence
    reporting variance over 5 seeds). Empty if the text shows none."""


class StructuredMethodsOutput(_Model):
    """What the structured methodology checker returns for one paper."""

    findings: list[StructuredFinding]
    """Exactly one finding per checklist item listed in the instructions, in that order."""
    other_concerns: list[OtherConcern]
    """Zero to five further methodology concerns, most serious first. Leave empty rather than
    padding with weak ones."""


class CritiqueScope(StrEnum):
    methodology = "methodology"
    """About how the experiments were designed, run, or reported (baselines, seeds, breadth of
    models/benchmarks/domains evaluated, ablations, confounds, leakage, disclosure, support for
    claims)."""
    design = "design"
    """About the method itself or its applicability (cost, assumptions, dependencies, what it
    does not do), not about how it was evaluated."""


class CritiqueMatchLabel(StrEnum):
    match = "match"
    partial = "partial"
    none = "none"


class CritiqueMatch(_Model):
    critique_index: int
    """The number in the critique's <critique index="..."> tag."""
    scope: CritiqueScope
    """Whether the human critique is about experimental methodology or about the method's
    design and applicability."""
    label: CritiqueMatchLabel
    """match: a checker finding raises essentially the same concern. partial: a finding touches
    the same issue but misses its main point, or is much broader or narrower. none: no finding
    raises it."""
    finding_ids: list[str]
    """Ids of the checker findings that match (empty for none)."""
    reason: str
    """One sentence justifying the label."""


class CritiqueMatchOutput(_Model):
    """One entry per human critique, in the order shown."""

    matches: list[CritiqueMatch]


class ConcernValidity(StrEnum):
    valid = "valid"
    partly_valid = "partly_valid"
    invalid = "invalid"


class ConcernCheck(_Model):
    concern_id: str
    """The id shown in the concern's <concern id="..."> tag."""
    validity: ConcernValidity
    """valid: the paper really has this problem as stated. partly_valid: a real issue but
    overstated, or the paper partly addresses it. invalid: the paper does address it, or the
    concern misreads the paper."""
    reason: str
    """One or two sentences, citing where the paper does or does not address it."""


class ConcernCheckOutput(_Model):
    """One check per concern shown."""

    checks: list[ConcernCheck]


class MethodsConcernRow(_ApiModel):
    """One methodology concern from an OpenReview review (reviewer-index
    data/methodology_eval/, one JSON line per concern). Unused fields are ignored; both the
    index's field names (excerpt, field, concern_id, offsets) and the longer ones are accepted."""

    submission_id: str
    venue: str = ""
    submission_title: str = ""
    submission_abstract: str = ""
    final_status: str | None = None
    """accepted / rejected / withdrawn / desk_rejected, when the file carries it."""
    review_id: str
    review_field: str = Field(default="", validation_alias=AliasChoices("review_field", "field"))
    category: str
    """baselines, seeds/variance, significance, leakage/contamination, compute fairness,
    ablations, dataset/split disclosure, hyperparameter disclosure, claims-vs-results,
    evaluation scope, other (any spelling; canonicalised on load)."""
    review_excerpt: str = Field(validation_alias=AliasChoices("review_excerpt", "excerpt"))
    """The concern, verbatim from the review."""
    case_id: str | None = Field(
        default=None, validation_alias=AliasChoices("case_id", "concern_id")
    )
    excerpt_offsets: list[int] | None = Field(
        default=None, validation_alias=AliasChoices("excerpt_offsets", "offsets")
    )
    cutoff_date: date | None = None
    """The venue's submission deadline; picks the PDF revision reviewers saw."""
    split: str | None = None
    """'dev' or 'test' when the file carries it; a test row is refused on load."""


class ConcernOutcome(_Model):
    """How the checker did on one review concern (dev eval output, one JSON line each)."""

    submission_id: str
    venue: str
    review_id: str
    category: str
    excerpt: str
    scope: CritiqueScope
    strict: CritiqueMatchLabel
    lenient: CritiqueMatchLabel
    finding_ids: list[str]
    own_item: bool | None
    """Whether a matched finding is the checklist item for this category (None when the category
    has no item, or nothing matched)."""
    reason: str
    final_status: str | None = None
    checker: str = ""
    """Checker prompt version that produced the findings matched."""
    concern_index: int = -1
    """Position of the concern within its submission (pairs concerns across runs)."""

    @property
    def concern_key(self) -> tuple[str, int]:
        return (self.submission_id, self.concern_index)

    pdf_provenance: str = ""
    """How the PDF relates to the version reviewers saw (see openreview_pdf.provenance);
    'existing_parse' when a parse was supplied."""


# --------------------------------------------------------------------------- failures

Stage = Literal[
    "fetch",
    "parse",
    "extract",
    "eval",
    "compare",
    "novelty",
    "judge",
    "citations",
    "format",
    "writing",
    "methods",
    "meta",
]


class Failure(_Model):
    stage: Stage
    paper_id: str
    error: str
    timestamp: datetime


# --------------------------------------------------------------------------- meta-reviewer


class CheckerName(StrEnum):
    claims = "claims"
    novelty = "novelty"
    citations = "citations"
    methods = "methods"
    format = "format"
    writing = "writing"


class CheckerRunStatus(StrEnum):
    ok = "ok"
    failed = "failed"
    not_run = "not_run"


class CheckerStatus(_Model):
    checker: CheckerName
    status: CheckerRunStatus
    detail: str = ""
    """The recorded failure, when status is failed."""


class EvidenceItem(_Model):
    """One citable fact from one checker, shown to the meta-reviewer LLM and checked against
    when it cites `evidence_ids`."""

    id: str
    """Stable id, e.g. 'claim:1', 'novelty:2:top3', 'novelty:1:duplicate', 'cite:b12',
    'cite:s7', 'methods:seeds_variance', 'methods:other:0', 'format:page_limit',
    'writing:undefined_term:0'."""
    checker: CheckerName
    text: str
    """What the LLM sees for this item: the finding, quote, and its source, in one block."""


class MetaInput(_Model):
    """Everything the meta-reviewer LLM sees for one paper. Assembled by code; nothing from
    OpenReview (reviews, ratings, decisions) ever enters this."""

    paper_id: str
    title: str
    venue: str
    """Target venue, e.g. 'ICLR'."""
    venue_year: int | None = None
    blinded_text: str
    """judge_context_full plus blinding: authors, affiliations, acknowledgements, funding,
    code/GitHub URLs, and venue headers/footers removed; title kept."""
    evidence: list[EvidenceItem]
    checker_status: list[CheckerStatus]


class WeaknessSeverity(StrEnum):
    fatal = "fatal"
    major = "major"
    minor = "minor"


class ReviewCategory(StrEnum):
    soundness = "soundness"
    novelty = "novelty"
    significance = "significance"
    presentation = "presentation"
    reproducibility = "reproducibility"


class Strength(_Model):
    text: str
    category: ReviewCategory
    evidence_ids: list[str] = Field(min_length=1)
    """At least one id from the evidence shown. Fluent writing alone is never a strength; cite
    a specific finding (a supported claim, a passed methods item, a real result)."""


class Weakness(_Model):
    text: str
    severity: WeaknessSeverity
    category: ReviewCategory
    evidence_ids: list[str] = Field(min_length=1)
    """At least one id from the evidence shown; every weakness must trace to a specific checker
    finding, never to writing style or general skepticism."""


class SubScoreItem(_Model):
    score: int
    """1-4 on the OpenReview scale (soundness, presentation, contribution)."""
    reason: str
    """One line citing evidence ids."""
    evidence_ids: list[str] = Field(default_factory=list)


class MetaJudgment(_Model):
    """What the meta-reviewer LLM returns for one paper. Raw judgment only: the calibrated
    score is computed by code afterward, never by the model."""

    summary: str
    """3-5 sentences: what the paper claims and does."""
    strengths: list[Strength]
    weaknesses: list[Weakness]
    """Prioritised, fatal first. Blunt, no hedging filler. Every item cites a checker finding;
    fluent or confident prose is never itself a weakness or a strength."""
    questions: list[str]
    """For the authors."""
    would_change_my_mind: str
    """What result or clarification would flip the verdict."""
    soundness: SubScoreItem
    presentation: SubScoreItem
    contribution: SubScoreItem
    raw_rating: int
    """On the venue's rating scale (e.g. ICLR: 1,3,5,6,8,10)."""
    confidence: int
    """1-5, the model's own confidence in this assessment."""


class MetaCalibration(_Model):
    """Computed by code from one or more MetaJudgments plus hard checker signals. Never
    produced by the LLM. `fitted` is False until a calibration model has been fit on real dev
    data (§4); until then `rating` is the median raw judgment rating, unscaled."""

    rating: float
    """Expected mean reviewer rating on the venue scale."""
    rating_percentile: float | None = None
    p_accept: float | None = None
    """ICLR scale only."""
    tier: str = ""
    fitted: bool = False
    features: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    """Every coefficient used, for transparency (empty and unfitted until dev calibration)."""


class BaselineRating(_Model):
    """One rating call with no checker evidence (plan §4 baselines a/b): title+abstract only
    (a), or the full blinded paper with no checker findings (b)."""

    reasoning: str
    """A few sentences: what drives the rating."""
    raw_rating: int
    """On the venue's scale."""
    confidence: int


class ContaminationProbe(_Model):
    """Plan §6: was this paper (title only) accepted at the venue, from the model's own
    memory? A confident, correct answer here means the paper is contaminated -- calibration and
    the meta-reviewer's headline number are reported split by this."""

    accepted_guess: bool
    reasoning: str
    """What, if anything, the model recalls about this paper."""
    confidence: int
    """1-5: how sure the model is that its guess reflects real memory, not a coin flip."""


class MetaReview(_Model):
    paper_id: str
    title: str
    venue: str
    venue_year: int | None = None
    model: str
    prompt_version: str
    generated: datetime
    judgments: list[MetaJudgment]
    """k judgment runs (k=1 for the lite configuration)."""
    dropped_evidence_ids: int = 0
    """Weaknesses/strengths whose evidence_ids did not all resolve; those items were dropped."""
    calibration: MetaCalibration
    checker_status: list[CheckerStatus]
    token_usage: TokenUsage
    baseline_abstract: BaselineRating | None = None
    """Plan §4 baseline (a): rated from title+abstract alone, no checkers. None if not run."""
    baseline_full: BaselineRating | None = None
    """Plan §4 baseline (b): rated from the full blinded text, no checker evidence. None if not
    run."""
    contamination_probe: ContaminationProbe | None = None
    """Plan §6: does the model recall this paper's acceptance from memory alone? None if not
    run."""


class DevRow(_Model):
    """One paper of the 60-paper dev run (plan §10): a MetaReview joined with its ground truth,
    ready for correlation/AUC reporting. Built once after the whole batch so the report code
    stays a pure function of this."""

    paper_id: str
    venue: str
    year: int
    final_status: str
    """accepted / rejected / withdrawn (the ground truth from decisions.jsonl)."""
    mean_reviewer_rating: float
    reviewer_ratings: list[int] = Field(default_factory=list)
    """Every reviewer's rating, for the one-reviewer-vs-the-rest human agreement baseline."""
    meta_raw_rating: float
    """Median raw_rating across judgment runs (extract_features' raw_rating)."""
    baseline_abstract_rating: int | None = None
    baseline_full_rating: int | None = None
    contamination_confidence: int | None = None
    contamination_accepted_guess: bool | None = None
    checkers_ok: int = 0
    checkers_total: int = 0


# --------------------------------------------------------------------------- deadlines


class Deadline(_Model):
    at: datetime
    """The deadline instant, timezone-aware UTC (23:59 AoE is 11:59 UTC the next day)."""
    source_url: str
    """The official page the date was copied from."""
    provisional: bool = False
    """True until the official page states the date and time exactly."""


# --------------------------------------------------------------------------- review orchestration


class ReviewStep(StrEnum):
    """Steps the W1 orchestrator runs (BP §4 DAG; claim_match, anchor, match, recheck come
    later)."""

    parse = "parse"
    format = "format"
    citations = "citations"
    claims = "claims"
    novelty = "novelty"
    methods = "methods"
    writing = "writing"
    meta = "meta"


class StepStatus(StrEnum):
    done = "done"
    failed = "failed"
    skipped = "skipped"
    timed_out = "timed_out"


class StepResult(_Model):
    step: ReviewStep
    status: StepStatus
    cost_usd: float = 0.0
    """Fresh spend charged by the step's budget guard (cache hits are $0)."""
    n_calls: int = 0
    """Uncached LLM calls the step made."""
    seconds: float = 0.0
    error_kind: str | None = None
    """Exception type or reason only; never paper text (BP §7)."""
    output: str | None = None
    """Path of the checker's output JSON."""


# --------------------------------------------------------------------------- MCP tools


class PriorWorkCandidate(_Model):
    """One candidate returned by the MCP tool find_prior_work (no label: the model labels)."""

    rank: int
    """1-based position in the first-stage ranker's order."""
    key: str
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    """Truncated to 1,200 characters."""
    link: str | None = None
    """arXiv abstract page, else the DOI URL."""
    seed_count: int = 0
    """How many of the top search results cite it or are cited by it."""
    matched_queries: list[int] = Field(default_factory=list)
    """1-based indices of the queries whose search returned it (empty: found by expansion)."""
    score: float
    """First-stage ranker score (higher = more likely prior work); only the order matters."""
    bibtex: str


class FindPriorWorkResult(_Model):
    queries: list[str]
    before: date | None = None
    index_version: str
    candidates: list[PriorWorkCandidate]
    n_considered: int
    """Distinct candidates scored (search hits plus one-hop citation neighbours)."""


class PaperRecord(_Model):
    """One paper as returned by the MCP lookups paper and citations."""

    key: str
    kind: str = "arxiv"
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    link: str | None = None
    bibtex: str


class CitationsResult(_Model):
    key: str
    direction: Literal["cites", "cited_by"]
    """'cites': works this paper cites; 'cited_by': works in the index that cite it."""
    papers: list[PaperRecord]
    truncated: bool
    """True when `limit` cut the list."""


class FormatCheckResult(_Model):
    """Output of the MCP tool check_format."""

    venue: str
    stage: SubmissionStage
    rules: str
    """The venue rules applied ('ICLR 2027'), or why none were (D18)."""
    text_engine: str
    """pdftotext (poppler installed) or pypdf."""
    author_names_from: Literal["grobid", "none"]
    """'none': GROBID was not reachable, so names on page 1 were not checked (affiliation,
    email, self-citation, URL and metadata checks still ran)."""
    detected_header: DeclaredVenue
    """What page 1 itself declares; used only to warn about a mismatch."""
    n_pages: int
    main_text_pages: float | None
    """For the chosen venue; None when its page rules are not built."""
    page_limit: int | None
    findings: list[FormatFinding]
    notes: list[str]


class CitationRefResult(_Model):
    """One reference as check_citations reports it."""

    ref_id: str
    raw_string: str
    """As printed (cut to 400 characters)."""
    resolved_title: str | None = None
    resolved_year: int | None = None
    """Earliest year among the resolved record's versions."""
    arxiv_id: str | None = None
    doi: str | None = None
    link: str | None = None
    source: str | None = None
    """Where it was resolved: index, openalex, crossref or arxiv."""
    flags: list[str] = Field(default_factory=list)


class CitationCheckResult(_Model):
    """Output of the MCP tool check_citations (no LLM; resolution and flags only)."""

    parser: Literal["grobid", "fallback"]
    """grobid: a local GROBID container parsed the reference list; fallback: the reference
    list was split from the PDF's text layer (see text_engine)."""
    text_engine: str | None = None
    n_references: int
    n_resolved: int
    index_available: bool
    flags: list[CitationFlag]
    references: list[CitationRefResult]
    notes: list[str]


class ReviewStepSummary(_Model):
    step: str
    status: str
    cost_usd: float
    error_kind: str | None = None


class ReviewPaperResult(_Model):
    """Output of the MCP tool review_paper. Never contains a rating, score or verdict (D11)."""

    phase: Literal["plan", "review"]
    """'plan' (confirm=false): nothing spent; show the estimate and ask the user. 'review':
    the full run."""
    estimate_usd: float
    """About this much (sum of per-step medians for the steps not yet done; D5)."""
    limit_usd: float
    """Hard limit: the run stops calling the model before spending more than this."""
    steps_to_run: list[str]
    steps_already_done: list[str]
    """Finished in an earlier call on the same PDF; rerunning them is served from the cache."""
    novelty_mode: Literal["lite", "full"] = "full"
    """full: 150 candidates per claim go to the LLM reranker; lite: 60 (quicker, cheaper)."""
    spent_usd: float | None = None
    steps: list[ReviewStepSummary] = Field(default_factory=list)
    findings: dict[str, Any] | None = None
    format: FormatCheckResult
    meta_review: "MetaReviewOut | None" = None
    """The allowlisted meta-review (no number anywhere; every sentence passed the D12 leak
    filter). None when the meta step did not run or failed."""
    notes: list[str] = Field(default_factory=list)


class MetaReviewText(_Model):
    text: str


class MetaReviewSink(_Model):
    text: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    """Evidence chips ship later (D13); empty until then."""


class MetaReviewOut(_Model):
    """The only serialized form of a meta-review (MASTER_PLAN §4.4, D11): an allowlist with no
    numeric field. Every string has passed `meta.leak_filter.is_score_leak` (D12)."""

    checks: list[dict[str, Any]] = Field(default_factory=list)
    """[{checker, ok}]: which checkers it was written from."""
    summary: str | None = None
    sink: list[MetaReviewSink] = Field(default_factory=list)
    """'Most likely to sink this paper': the top 3 remaining weaknesses, major first."""
    strengths: list[MetaReviewText] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    would_change_my_mind: str | None = None
    """None when every sentence was filtered; the section is then hidden."""
    left_out_note: str | None = None


ReviewPaperResult.model_rebuild()


class VerifiedQuote(_Model):
    quote: str
    status: Literal["verified", "not_found", "too_short"]
    """verified: the quote is in the PDF word for word (punctuation, hyphenation, ligatures and
    whitespace forgiven; "..." joins fragments that appear in order, close together)."""
    pages: list[int] = Field(default_factory=list)
    """1-based page(s) where it appears; two pages for a quote across a page break; empty when
    it was found only in GROBID's parse (e.g. a table)."""


class VerifyQuotesResult(_Model):
    quotes: list[VerifiedQuote]
    n_verified: int
    text_engine: str
