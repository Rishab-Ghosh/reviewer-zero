"""Find a paper's own contributions statement in its introduction. Pure; no I/O."""

import re

from reviewer.models import Section

_I = re.IGNORECASE

TRIGGERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bour (?:main |key |primary |major |core |principal )?contributions?\b", _I),
    re.compile(r"\bwe make the following (?:\w+ )?contributions?\b", _I),
    re.compile(
        r"\bthe (?:main |key |primary |major )?contributions? of (?:this|our) "
        r"(?:paper|work|study|article)\b",
        _I,
    ),
    re.compile(r"\b(?:this paper|this work|this study|we) contributes?\b", _I),
    re.compile(r"\bin summary,? (?:we|our|this)\b", _I),
    re.compile(r"\bto summari[sz]e\b", _I),
    re.compile(r"\bcontributions? (?:is|are) (?:two|three|four|five)-?fold\b", _I),
    re.compile(r"\b(?:two|three|four|five)-?fold contributions?\b", _I),
    re.compile(r"\bsummari[sz]ed? (?:our|the) (?:main |key )?contributions?\b", _I),
)

_INTRO = re.compile(r"\bintroduction\b", _I)
_RELATED = re.compile(r"\b(?:related work|prior work|previous work|background|literature)\b", _I)

_BULLET_START = re.compile(
    r"^\s*(?:[•\-–—*▪◦·●○■□➢➤✓]"  # bullet glyphs
    r"|\(?(?:[ivx]{1,4}|\d{1,2}|[a-h])[.)]\s"  # (i) i) 1. 1) (a) a)
    r"|(?:first|second|third|fourth|fifth|firstly|secondly|thirdly|finally|lastly|last)\b"
    r"|we (?:propose|introduce|show|present|demonstrate|develop|design|release|provide"
    r"|conduct|evaluate|find|establish|derive|prove|study|analy[sz]e|identify|extend"
    r"|construct|curate|achieve|formulate|offer|build|create|collect|open-source|make)\b)",
    _I,
)
MAX_ITEM_WORDS = 120
MAX_UNMARKED_ITEM_WORDS = 45
MAX_ITEMS = 8
MAX_FALLBACK_SECTIONS = 3


def has_trigger(text: str) -> bool:
    return any(p.search(text) for p in TRIGGERS)


def is_bullet_like(paragraph: str) -> bool:
    return bool(_BULLET_START.match(paragraph)) and len(paragraph.split()) <= MAX_ITEM_WORDS


def _announces_list(paragraph: str) -> bool:
    stripped = paragraph.rstrip()
    return stripped.endswith(":") or bool(re.search(r"\bfollowing\b", stripped, _I))


def intro_span(sections: list[Section]) -> list[Section]:
    """The introduction (first heading matching 'introduction', else the first section) plus
    its subsections (following sections with a deeper level)."""
    if not sections:
        return []
    start = next((i for i, s in enumerate(sections) if _INTRO.search(s.heading)), 0)
    span = [sections[start]]
    for s in sections[start + 1 :]:
        if s.level <= sections[start].level:
            break
        span.append(s)
    return span


def fallback_span(sections: list[Section], exclude: list[Section]) -> list[Section]:
    """Early sections before related work (at most a few), not already searched."""
    out: list[Section] = []
    for s in sections[:MAX_FALLBACK_SECTIONS]:
        if _RELATED.search(s.heading):
            break
        if not any(s is e for e in exclude):
            out.append(s)
    return out


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def from_trigger_sentence(paragraph: str) -> str:
    """Drop the sentences before the first one containing a trigger phrase, so a long intro
    paragraph that ends in "Our contributions are as follows:" yields just that sentence."""
    sentences = _SENTENCE_END.split(paragraph)
    for i, sentence in enumerate(sentences):
        if has_trigger(sentence):
            return " ".join(sentences[i:])
    return paragraph


def capture_statement(paragraphs: list[str], trigger_index: int) -> str:
    """The trigger paragraph (from its trigger sentence on) plus the list items / short
    bullet-like paragraphs after it."""
    trigger = paragraphs[trigger_index]
    captured = [from_trigger_sentence(trigger)]
    following = paragraphs[trigger_index + 1 :]
    # Items are either marked (bullets, numbering, "First,", "We propose ...") or, when the
    # trigger announces a list ("...:"), short unmarked paragraphs (GROBID often drops bullet
    # glyphs). The first item decides which kind we collect; anything else ends the list.
    marked = bool(following) and is_bullet_like(following[0])
    unmarked_ok = not marked and _announces_list(trigger)
    for para in following[:MAX_ITEMS]:
        is_item = (marked and is_bullet_like(para)) or (
            unmarked_ok and len(para.split()) <= MAX_UNMARKED_ITEM_WORDS
        )
        if not is_item:
            break
        captured.append(para)
    return "\n".join(captured)


def _search(sections: list[Section]) -> str | None:
    for section in sections:
        for i, para in enumerate(section.paragraphs):
            if has_trigger(para):
                return capture_statement(section.paragraphs, i)
    return None


_CONTRIB_HEADING = re.compile(r"^\s*(?:our |main |key )?contributions?\s*[.:]?\s*$", _I)


def _contributions_heading(sections: list[Section]) -> str | None:
    """A section headed just 'Contributions' in the first few sections. GROBID turns an inline
    paragraph heading ("Contributions. • We propose ...") into its own section."""
    for section in sections[: MAX_FALLBACK_SECTIONS + 2]:
        if _CONTRIB_HEADING.match(section.heading) and section.paragraphs:
            return "\n".join(section.paragraphs)
    return None


def find_contributions(sections: list[Section]) -> str | None:
    """Return the contributions statement from the introduction, or None if there is none."""
    intro = intro_span(sections)
    found = _search(intro)
    if found is not None:
        return found
    found = _contributions_heading(sections)
    if found is not None:
        return found
    return _search(fallback_span(sections, intro))
