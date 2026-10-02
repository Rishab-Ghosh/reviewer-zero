"""Format and anonymization checks. Pure functions over the PDF text layer (no network, no LLM).

Severity policy (logged in STATUS):
- Anonymization findings are violations only when the PDF is at the submission stage (anonymous,
  "under review", "submitted to"). Camera-ready papers and preprints legitimately name their
  authors, so there the same findings are informational.
- Venue rules (page limits, required sections) are violations only for the venue the PDF
  declares, and only when the paper's declared year is not older than the rule's edition.
  Otherwise they are reported as info ("would exceed ICML 2026's limit").
"""

import re
from dataclasses import dataclass
from typing import Literal

from reviewer.format.layout import Heading, heading_candidates, page_lines
from reviewer.format.venues import VENUES, SectionRule, VenueRules
from reviewer.models import (
    DeclaredVenue,
    FormatCheck,
    FormatFinding,
    FormatSeverity,
    SubmissionStage,
    Venue,
    VenuePages,
)

EDITION_YEAR = {Venue.iclr: 2027, Venue.neurips: 2026, Venue.icml: 2026}


@dataclass(frozen=True)
class Context:
    pages: list[str]
    metadata: dict[str, str]
    headings: list[Heading]
    declared: DeclaredVenue
    refs_span: tuple[int, int] | None
    author_names: list[str]
    """Author names GROBID found in the header (may be empty or noisy)."""
    rules_mode: Literal["declared", "none"] = "declared"
    """'none' (MASTER_PLAN D18): a venue whose rules are not built (ACL/ARR, Other, CVPR until
    D19): no venue page/section rules, acknowledgments only as info, and no finding cites
    another venue's rule page. Every anonymization check still runs."""

    @property
    def at_submission(self) -> bool:
        return self.declared.stage == SubmissionStage.submission

    def body_pages(self) -> list[tuple[int, str]]:
        """(page number, text) for every page outside the reference list."""
        lo, hi = self.refs_span or (0, -1)
        return [(i, t) for i, t in enumerate(self.pages, 1) if not lo <= i <= hi]


def anon_severity(ctx: Context) -> FormatSeverity:
    return FormatSeverity.violation if ctx.at_submission else FormatSeverity.info


_RULE_FALLBACK: dict[str, Venue] = {
    "anonymity_rule": Venue.neurips,
    "self_citation_rule": Venue.neurips,
    "url_rule": Venue.icml,
}


def _rule_source(ctx: Context, attr: str) -> str | None:
    """The rule page a finding cites: the declared venue's, else the long-standing fallback
    venue's; None in rules_mode 'none' (D18), so no finding cites a venue it is not for."""
    if ctx.rules_mode == "none":
        return None
    venue = ctx.declared.venue or _RULE_FALLBACK[attr]
    return getattr(VENUES[venue], attr).source


def _snippet(text: str, start: int, end: int, pad: int = 90) -> str:
    return " ".join(text[max(0, start - pad) : end + pad].split())


# --------------------------------------------------------------------------- anonymization

EMAIL_RE = re.compile(r"(\{[^{}]{1,120}\}|[\w.+-]+)\s?@\s?[\w-]+(\.[\w-]+)+")
AFFILIATION_RE = re.compile(
    r"\b(Universit(y|ät|é|at|a|e)|Institut(e)?|College|Laborator(y|ies)|School of|Department|"
    r"Dept\.|Inc\.|Ltd\.?|Corporation|Academy|Centre|Center for|Google|Microsoft|DeepMind|Meta AI|"
    r"FAIR|OpenAI|Anthropic|NVIDIA|Amazon|IBM|Tencent|Alibaba|ByteDance|Huawei|Baidu|Apple|"
    r"Salesforce|Samsung|Adobe|Mila|ETH|EPFL|MIT|CMU|Stanford|Berkeley|Tsinghua|Peking)\b"
)
ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
ANON_HEADER_RE = re.compile(r"anonymous", re.IGNORECASE)


def header_lines(page1: str) -> list[str]:
    """Page-1 lines before the abstract (title and author block)."""
    lines = page_lines(page1)
    for i, line in enumerate(lines):
        if ABSTRACT_RE.match(line) or line.replace(" ", "").upper() == "ABSTRACT":
            return lines[:i]
    return lines[:25]


def check_author_block(ctx: Context) -> list[FormatFinding]:
    head = header_lines(ctx.pages[0]) if ctx.pages else []
    head_text = " ".join(head)
    if ANON_HEADER_RE.search(head_text):
        return []
    names = [
        n
        for n in ctx.author_names
        if len(n.split()) >= 2 and n in head_text and not AFFILIATION_RE.search(n)
    ]
    affiliations = [ln for ln in head if AFFILIATION_RE.search(ln)][:4]
    if not names and not affiliations:
        return []
    evidence = []
    if names:
        evidence.append("names on page 1: " + ", ".join(names[:8]))
    evidence += [f"affiliation line: {ln[:140]}" for ln in affiliations]
    return [
        FormatFinding(
            check=FormatCheck.author_block,
            severity=anon_severity(ctx),
            detail="author names or affiliations printed above the abstract",
            page=1,
            evidence=evidence,
            source=_rule_source(ctx, "anonymity_rule"),
        )
    ]


def check_emails(ctx: Context) -> list[FormatFinding]:
    hits: list[tuple[int, str]] = []
    for p, text in ctx.body_pages():
        for m in EMAIL_RE.finditer(text):
            if "anonymous" in m.group(0).lower():
                continue
            hits.append((p, m.group(0)))
    if not hits:
        return []
    return [
        FormatFinding(
            check=FormatCheck.email,
            severity=anon_severity(ctx),
            detail=f"{len(hits)} email address(es) in the text",
            page=hits[0][0],
            evidence=[f"p{p}: {e}" for p, e in hits[:5]],
            source=_rule_source(ctx, "anonymity_rule"),
        )
    ]


SELF_CITATION_RES = [
    re.compile(
        r"\bour\s+(?:own\s+)?(?:previous|prior|earlier|preliminary|past|recent)\s+"
        r"(?:work|paper|study|studies|research|publication|version|method|approach|analysis|"
        r"results?|findings?)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:have\s+)?(?:previously|earlier)\s+(?:showed|shown|proposed|introduced|"
        r"presented|demonstrated|developed|studied|reported)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:in|see)\s+our\s+(?:(?:\w+\s+){0,2})?(?:paper|work)\s*[\[(]", re.IGNORECASE),
]
CITATION_MARK_RE = re.compile(
    r"\[\d+(?:[,–-]\s*\d+)*\]|\([A-Z][\w-]+(?: et al\.)?,? \d{4}\w?\)|"
    r"[A-Z][\w-]+ et al\."
)


def check_self_citation(ctx: Context) -> list[FormatFinding]:
    out = []
    for p, text in ctx.body_pages():
        flat = " ".join(re.sub(r"(\w)-\n(\w)", r"\1\2", text).split())
        for rx in SELF_CITATION_RES:
            for m in rx.finditer(flat):
                snip = _snippet(flat, m.start(), m.end())
                cited = bool(CITATION_MARK_RE.search(flat[m.start() : m.end() + 120]))
                out.append(
                    FormatFinding(
                        check=FormatCheck.self_citation,
                        severity=anon_severity(ctx),
                        detail=(
                            "first-person reference to the authors' own prior work"
                            + (" with a citation" if cited else "")
                        ),
                        page=p,
                        evidence=[f'p{p}: "…{snip}…"', f"matched: {m.group(0)}"],
                        source=_rule_source(ctx, "self_citation_rule"),
                    )
                )
    # one finding per distinct match text and page
    seen, unique = set(), []
    for f in out:
        key = (f.page, f.evidence[1])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?P<host>(?:[a-z0-9-]+\.)+(?:com|org|io|net|edu|ai|dev|science|"
    r"site|page|me|cn|uk|de|fr|ca|ch|jp|kr|info|app))(?P<path>/[^\s)\]}>,;\"'“”]*)?",
    re.IGNORECASE,
)
CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co", "codeberg.org")
ANON_HOSTS = ("anonymous.4open.science", "openreview.net", "anonymous.")
OWN_CONTEXT_RE = re.compile(
    r"\bour\s+(?:\w+\s+){0,2}(?:code|codebase|implementation|repository|repo|project|website|"
    r"page|models?|checkpoints?|data(?:set)?s?|library|package|tool|demo|benchmark)\b"
    r"|\bwe\s+(?:\w+\s+){0,2}(?:release|open[- ]source|publish|make\s+\w+\s+available)"
    r"|\b(?:code|models?|data(?:set)?s?|implementation|checkpoints?|weights|project page|website|"
    r"demo)\s+(?:is|are|will be|can be)?\s*(?:publicly\s+|freely\s+|made\s+)?"
    r"(?:available|released|found)\b|\bavailable at\b|\bcode:|\bproject page:",
    re.IGNORECASE,
)


THIRD_PARTY_RE = re.compile(
    r"\b(based on|builds? (up)?on|built (up)?on|uses?|using|adapted from|borrowed from|"
    r"taken from|open-sourced|provided by|courtesy of)\b",
    re.IGNORECASE,
)
"""Lead-ins that point to someone else's code ("Our implementation uses Apex's FMHA code (...)",
"is based on the open-sourced PyTorch implementation in ...")."""


def url_identity(host: str, path: str) -> str | None:
    """Why this URL could identify its owner, or None (third-party or anonymous host)."""
    host = host.lower()
    if any(a in host for a in ANON_HOSTS) or "anonymous" in path.lower():
        return None
    if host in CODE_HOSTS:
        owner = path.strip("/").split("/")[0] if path.strip("/") else ""
        return f"{host} account '{owner}'" if owner else None
    if host.endswith((".github.io", ".gitlab.io")):
        return f"personal/project site {host}"
    if host == "sites.google.com" or "~" in path:
        return f"personal page {host}{path[:40]}"
    return None


def check_urls(ctx: Context) -> list[FormatFinding]:
    out: list[FormatFinding] = []
    seen: set[str] = set()
    for p, text in ctx.body_pages():
        fixed = re.sub(r"(https?:)\s*//\s*", r"\1//", text)
        fixed = re.sub(r"(\.(?:com|io|org))/\s*\n\s*", r"\1/", fixed)
        for m in URL_RE.finditer(fixed):
            why = url_identity(m.group("host"), m.group("path") or "")
            url = m.group(0).rstrip(".")
            if not why or url.lower() in seen:
                continue
            # the sentence (up to 160 chars) leading into the link
            before = fixed[max(0, m.start() - 160) : m.start()]
            window = re.split(r"(?<=[.!?])\s+(?=[A-Z])", before)[-1]
            third_party = bool(THIRD_PARTY_RE.search(window))
            own = (p == 1 or bool(OWN_CONTEXT_RE.search(window))) and not third_party
            if not own:
                continue  # a third-party repository cited as a resource
            seen.add(url.lower())
            out.append(
                FormatFinding(
                    check=FormatCheck.identifying_url,
                    severity=anon_severity(ctx),
                    detail=f"non-anonymous link presented as the authors' own: {why}",
                    page=p,
                    evidence=[
                        f"p{p}: {url}",
                        f'context: "…{_snippet(fixed, m.start(), m.end(), 110)}…"',
                    ],
                    source=_rule_source(ctx, "url_rule"),
                )
            )
    return out


def check_acknowledgments(ctx: Context) -> list[FormatFinding]:
    acks = [h for h in ctx.headings if h.kind == "acknowledgments" and h.page >= 2]
    if not acks:
        return []
    h = acks[0]
    if ctx.rules_mode == "none":
        rules: list[VenueRules] = []
    else:
        rules = [VENUES[ctx.declared.venue]] if ctx.declared.venue else list(VENUES.values())
    banned = [r for r in rules if r.acknowledgments_rule]
    severity = anon_severity(ctx) if banned else FormatSeverity.info
    src = banned[0].acknowledgments_rule.source if banned else None  # type: ignore[union-attr]
    lines = page_lines(ctx.pages[h.page - 1])
    body = " ".join(lines[h.line : h.line + 3])
    return [
        FormatFinding(
            check=FormatCheck.acknowledgments,
            severity=severity,
            detail="acknowledgments/funding section present"
            + (
                ""
                if banned
                else " (check your venue's rule)"
                if ctx.rules_mode == "none"
                else " (the declared venue's pages do not ban it at submission)"
            ),
            page=h.page,
            evidence=[f"p{h.page}: {body[:220]}"],
            source=src,
        )
    ]


def check_pdf_metadata(ctx: Context) -> list[FormatFinding]:
    author = ctx.metadata.get("Author", "")
    if not author or ANON_HEADER_RE.search(author):
        return []
    return [
        FormatFinding(
            check=FormatCheck.pdf_metadata,
            severity=anon_severity(ctx),
            detail="PDF metadata Author field names people",
            evidence=[f"Author: {author[:200]}"]
            + [f"{k}: {v[:120]}" for k, v in ctx.metadata.items() if k in ("Title", "Creator")],
            source=_rule_source(ctx, "anonymity_rule"),
        )
    ]


# --------------------------------------------------------------------------- venue rules


def target_venues(ctx: Context) -> list[VenueRules]:
    if ctx.rules_mode == "none":
        return []
    return [VENUES[ctx.declared.venue]] if ctx.declared.venue else list(VENUES.values())


def rule_applies(ctx: Context, rules: VenueRules) -> bool:
    """A venue rule is binding only for the declared venue and a paper not older than the
    rule's edition (older papers were accepted under older rules)."""
    if ctx.declared.venue != rules.venue:
        return False
    year = ctx.declared.year
    return year is None or year >= EDITION_YEAR[rules.venue]


def check_page_limits(ctx: Context, pages: list[VenuePages]) -> list[FormatFinding]:
    out = []
    for rules in target_venues(ctx):
        vp = next(v for v in pages if v.venue == rules.venue)
        camera = ctx.declared.stage == SubmissionStage.camera_ready
        limit = vp.camera_ready_limit if camera else vp.submission_limit
        if vp.pages_used <= limit:
            continue
        binding = rule_applies(ctx, rules)
        kind = "camera-ready" if camera else "submission"
        out.append(
            FormatFinding(
                check=FormatCheck.page_limit,
                severity=FormatSeverity.violation if binding else FormatSeverity.info,
                venue=rules.venue,
                detail=(
                    f"main text uses {vp.pages_used} pages ({vp.main_text_pages:.2f}); "
                    f"{rules.edition} {kind} limit is {limit}"
                    + ("" if binding else " (not binding: different or older venue)")
                ),
                page=vp.pages_used,
                evidence=[
                    f"main text ends at: {vp.ends_at or 'no end heading found (whole PDF)'}",
                    f'rule: "{rules.page_rule.quote}"',
                ],
                source=rules.page_rule.source,
            )
        )
    return out


def section_present(ctx: Context, rule: SectionRule) -> str | None:
    """The line where the section appears, or None."""
    for p, text in enumerate(ctx.pages, 1):
        for line in page_lines(text):
            if len(line) > 160:
                continue
            for form in heading_candidates(line):
                if any(re.match(pat, form) for pat in rule.heading_patterns):
                    return f"p{p}: {line[:120]}"
        for pat in rule.text_patterns:
            if m := re.search(pat, text, re.IGNORECASE):
                return f"p{p}: …{m.group(0)}…"
    return None


def check_sections(ctx: Context) -> list[FormatFinding]:
    out = []
    for rules in target_venues(ctx):
        binding = rule_applies(ctx, rules)
        for rule in rules.sections:
            if section_present(ctx, rule):
                continue
            if rule.required:
                check, severity = (
                    FormatCheck.required_section,
                    (FormatSeverity.violation if binding else FormatSeverity.info),
                )
            else:
                check, severity = FormatCheck.recommended_section, FormatSeverity.info
            out.append(
                FormatFinding(
                    check=check,
                    severity=severity,
                    venue=rules.venue,
                    detail=f"{rules.edition}: {rule.name} "
                    f"{'(required)' if rule.required else '(encouraged)'} not found"
                    + (
                        ""
                        if binding or not rule.required
                        else " (not binding: different or older venue)"
                    ),
                    evidence=[f'rule: "{rule.rule.quote}"'],
                    source=rule.rule.source,
                )
            )
    return out


# --------------------------------------------------------------------------- references

UNRESOLVED_RE = re.compile(r"\?\?|\[\s*\?\s*\]")


def check_unresolved(ctx: Context) -> list[FormatFinding]:
    hits = []
    for p, text in ctx.body_pages():
        for m in UNRESOLVED_RE.finditer(text):
            hits.append((p, _snippet(text, m.start(), m.end(), 60)))
    if not hits:
        return []
    return [
        FormatFinding(
            check=FormatCheck.unresolved_reference,
            severity=FormatSeverity.warning,
            detail=f"{len(hits)} unresolved reference marker(s) ('??' or '[?]')",
            page=hits[0][0],
            evidence=[f'p{p}: "…{s}…"' for p, s in hits[:6]],
        )
    ]


LABEL = r"(?:[A-Z]\.?\d+(?:\.\d+)*|\d+(?:\.\d+)*|[A-Z](?![A-Za-z]))"
CAPTION_RE = re.compile(
    rf"^\s*(?P<kind>Figure|Fig\.|Table|TABLE|FIGURE)\s+(?P<label>{LABEL})"
    rf"\s*(?:[:|]|\.(?!\d)|\s[-–—]\s)\s*\S"
)
MENTION_RE = re.compile(
    rf"\b(?P<kind>(?i:figures?|figs?\.|tables?|tabs?\.))\s*~?\s*"
    rf"(?P<list>{LABEL}[a-z]?(?:\s*\([a-z]\))?(?:\s*(?:,\s*(?:and\s+)?|and|&|–|—|-|to)\s*{LABEL}[a-z]?)*)"
)


def _norm_label(label: str) -> str:
    return label.replace(" ", "").rstrip(".").upper()


def _expand(mention: str) -> set[str]:
    labels = re.findall(LABEL, mention)
    out = {_norm_label(x) for x in labels}
    for a, b in re.findall(r"(\d+)\s*(?:–|—|-|to)\s*(\d+)", mention):
        if int(a) < int(b) <= int(a) + 30:
            out |= {str(i) for i in range(int(a), int(b) + 1)}
    return out


def check_unreferenced(ctx: Context) -> list[FormatFinding]:
    """Captions ('Figure 3: ...') whose label is never mentioned elsewhere ('Figure 3',
    'Figs. 2-4', 'Tables 1 and 2'). A caption's own label is blanked before searching, so it
    does not count as its own mention; mentions split across lines are found on the joined
    page text."""
    captions: dict[tuple[str, str], tuple[int, str]] = {}
    mentions: dict[str, set[str]] = {"figure": set(), "table": set()}
    for p, text in enumerate(ctx.pages, 1):
        lines = page_lines(text)
        for i, line in enumerate(lines):
            if m := CAPTION_RE.match(line):
                kind = "table" if m.group("kind").lower().startswith("tab") else "figure"
                captions.setdefault((kind, _norm_label(m.group("label"))), (p, line[:120]))
                lines[i] = line[m.end("label") :]
        for m in MENTION_RE.finditer(" ".join(lines)):
            kind = "table" if m.group("kind").lower().startswith("tab") else "figure"
            mentions[kind] |= _expand(m.group("list"))
    out = []
    for (kind, label), (p, line) in sorted(captions.items(), key=lambda kv: kv[1][0]):
        if label in mentions[kind]:
            continue
        out.append(
            FormatFinding(
                check=FormatCheck.unreferenced_table
                if kind == "table"
                else FormatCheck.unreferenced_figure,
                severity=FormatSeverity.warning,
                detail=f"{kind.capitalize()} {label} is never referenced in the text",
                page=p,
                evidence=[f"caption p{p}: {line}"],
            )
        )
    return out


def run_all(ctx: Context, pages: list[VenuePages]) -> list[FormatFinding]:
    return [
        *check_page_limits(ctx, pages),
        *check_author_block(ctx),
        *check_emails(ctx),
        *check_self_citation(ctx),
        *check_urls(ctx),
        *check_acknowledgments(ctx),
        *check_pdf_metadata(ctx),
        *check_sections(ctx),
        *check_unresolved(ctx),
        *check_unreferenced(ctx),
    ]
