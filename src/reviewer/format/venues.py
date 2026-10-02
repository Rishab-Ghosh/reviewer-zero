"""Venue rules, each with the official page it comes from and a verbatim quote.

Editions: the most recent call for papers published as of 2026-09-25 (ICLR 2027, NeurIPS 2026,
ICML 2026). Every paper is checked against these current rules, even papers accepted under older
editions (e.g. NeurIPS 2017 had an 8-page limit); reports say so.
"""

from dataclasses import dataclass, field

from reviewer.models import Venue

ICLR_CFP = "https://iclr.cc/Conferences/2027/CallForPapers"
ICLR_AG = "https://iclr.cc/Conferences/2027/AuthorGuidelines"
ICLR_AIP = "https://iclr.cc/Conferences/2027/AIPolicyForAuthors"
NEURIPS_CFP = "https://neurips.cc/Conferences/2026/CallForPapers"
NEURIPS_HB = "https://neurips.cc/Conferences/2026/MainTrackHandbook"
NEURIPS_CL = "https://neurips.cc/public/guides/PaperChecklist"
ICML_CFP = "https://icml.cc/Conferences/2026/CallForPapers"
ICML_AI = "https://icml.cc/Conferences/2026/AuthorInstructions"


@dataclass(frozen=True)
class Rule:
    source: str
    quote: str


@dataclass(frozen=True)
class SectionRule:
    name: str
    heading_patterns: tuple[str, ...]
    """Regexes (case-insensitive) matched against normalized heading lines."""
    required: bool
    """True: mandatory (missing = violation). False: encouraged (missing = info)."""
    rule: Rule
    text_patterns: tuple[str, ...] = ()
    """Regexes that also count as present when found anywhere in the text (e.g. checklist
    answers)."""


@dataclass(frozen=True)
class VenueRules:
    venue: Venue
    edition: str
    submission_pages: int
    camera_ready_pages: int
    page_rule: Rule
    excluded_headings: tuple[str, ...]
    """Kinds of headings (see layout.HEADING_KINDS) whose content does not count toward the
    main-text page limit; the main text ends at the first of them."""
    excluded_rule: Rule
    sections: tuple[SectionRule, ...]
    anonymity_rule: Rule
    acknowledgments_rule: Rule | None
    """None when the venue's pages do not say acknowledgments are banned at submission."""
    self_citation_rule: Rule
    url_rule: Rule
    extra: dict[str, Rule] = field(default_factory=dict)


ICLR = VenueRules(
    venue=Venue.iclr,
    edition="ICLR 2027",
    submission_pages=9,
    camera_ready_pages=10,
    page_rule=Rule(
        ICLR_AG,
        "At the time of submission, the main text should be 9 pages or fewer. [...] for the "
        "camera ready, the page limit will be increased to 10 pages",
    ),
    excluded_headings=(
        "references",
        "appendix",
        "acknowledgments",
        "ethics",
        "reproducibility",
        "ai_use",
    ),
    excluded_rule=Rule(
        ICLR_AG,
        "The list of references does not count towards the page limit [...] Authors may use as "
        "many pages of appendices (after the bibliography) as they wish [...] ethics statements "
        "and acknowledgement etc. do not count for page limit.",
    ),
    sections=(
        SectionRule(
            "AI use statement",
            (
                r"\b(ai|llm|large language models?)\s+(use|usage)\b",
                r"\buse of (large )?language models\b",
                r"\bllm usage\b",
            ),
            required=True,
            rule=Rule(
                ICLR_AIP,
                "we require authors to disclose AI use in a mandatory section "
                "in their paper (this section will not be counted towards the page limit)",
            ),
        ),
        SectionRule(
            "Ethics statement",
            (r"\bethics statement\b",),
            required=False,
            rule=Rule(
                ICLR_AG,
                "The optional ethics statement will not count toward the page "
                "limit, but should not be more than 1 page.",
            ),
        ),
        SectionRule(
            "Reproducibility statement",
            (r"\breproducibility statement\b",),
            required=False,
            rule=Rule(
                ICLR_AG,
                "strongly encouraged to include a paragraph-long "
                "Reproducibility Statement at the end of the main text (before references)",
            ),
        ),
    ),
    anonymity_rule=Rule(
        ICLR_AG,
        "Any paper where author identity is revealed in either the "
        "main text or the supplementary material will be desk rejected.",
    ),
    acknowledgments_rule=None,
    self_citation_rule=Rule(
        ICLR_AG,
        "related arxiv papers by the same authors do not break "
        "anonymity; if cited, these should be cited in third person.",
    ),
    url_rule=Rule(ICLR_AG, "Make an anonymous repository and put the link in your paper."),
)

NEURIPS = VenueRules(
    venue=Venue.neurips,
    edition="NeurIPS 2026",
    submission_pages=9,
    camera_ready_pages=10,
    page_rule=Rule(
        NEURIPS_HB,
        "The main text of a submitted paper is limited to nine content pages, including all "
        "figures and tables. [...] you will be allowed an additional content page for the "
        "camera-ready version.",
    ),
    # Acknowledgments are banned at submission and their camera-ready status is not stated;
    # we exclude them (logged choice). Broader impacts and limitations DO count.
    excluded_headings=("references", "appendix", "checklist", "acknowledgments"),
    excluded_rule=Rule(
        NEURIPS_HB,
        "Additional pages containing references, optional technical appendices and mandatory "
        "paper checklist do not count as content pages.",
    ),
    sections=(
        SectionRule(
            "NeurIPS paper checklist",
            (r"\bchecklist\b",),
            required=True,
            rule=Rule(NEURIPS_CL, "The papers not including the checklist will be desk rejected."),
            text_patterns=(r"answer:\s*\[(yes|no|n/a|na)\]",),
        ),
        SectionRule(
            "Limitations",
            (r"\blimitations?\b",),
            required=False,
            rule=Rule(
                NEURIPS_CL,
                "The authors are encouraged to create a separate "
                '"Limitations" section in their paper.',
            ),
        ),
    ),
    anonymity_rule=Rule(
        NEURIPS_HB,
        "All submissions must be anonymized and may not contain any identifying information",
    ),
    acknowledgments_rule=Rule(
        NEURIPS_HB, "Please do not include acknowledgments at submission time."
    ),
    self_citation_rule=Rule(
        NEURIPS_HB,
        'write "In the previous work of Smith et al. [1]…" '
        'rather than "In our previous work [1]..."',
    ),
    url_rule=Rule(
        NEURIPS_HB,
        "If you are including links to any external material, it is "
        "your responsibility to guarantee anonymous browsing.",
    ),
)

ICML = VenueRules(
    venue=Venue.icml,
    edition="ICML 2026",
    submission_pages=8,
    camera_ready_pages=9,
    page_rule=Rule(
        ICML_CFP,
        "8 pages for the main paper, with unlimited pages for references, the impact "
        "statement, and appendices.",
    ),
    excluded_headings=("references", "appendix", "acknowledgments", "impact"),
    excluded_rule=Rule(
        ICML_AI,
        "the length limit of the paper body is 9 pages, followed by any acknowledgements, the "
        "impact statement (for main track papers), references, and appendices.",
    ),
    sections=(
        SectionRule(
            "Impact statement",
            (r"\bimpact statement\b", r"\bbroader impacts?\b", r"\bsocietal impacts?\b"),
            required=True,
            rule=Rule(
                ICML_CFP,
                "Authors are required to include a statement of the potential "
                "broader impact of their work",
            ),
        ),
    ),
    anonymity_rule=Rule(
        ICML_AI,
        "papers that explicitly or implicitly reveal the authors' identities will be rejected.",
    ),
    acknowledgments_rule=Rule(
        ICML_AI,
        "They should refrain from including acknowledgements, "
        "grant numbers, or links to public code repositories",
    ),
    self_citation_rule=Rule(
        ICML_AI, "authors should refer to their prior work in the third person wherever possible."
    ),
    url_rule=Rule(
        ICML_AI,
        "Authors should not include links to non-anonymized repositories; "
        "instead, they should submit the code base itself or anonymized repositories.",
    ),
)

VENUES: dict[Venue, VenueRules] = {v.venue: v for v in (ICLR, NEURIPS, ICML)}
