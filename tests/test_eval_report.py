from datetime import timedelta

from reviewer.eval.annotate import annotate_records, public_records
from reviewer.eval.checks import run_checks
from reviewer.eval.loader import build_records, latest_failures
from reviewer.eval.report import render_report
from reviewer.models import GoldClaim

from .test_eval_helpers import TS, entry, failure, gold, paper_claims, parsed

S1 = "Sparse attention cuts memory by 40% on 16k-token documents."
S2 = "A new 500-problem geometry benchmark for vision-language models."
PRIVATE_GOLD = "Lab says the secret sauce reaches 77.7% on the hidden benchmark."
OWN_GOLD = "My own unpublished claim about gradient surgery."


def _raw_records():
    corpus = [
        entry("famous-1", "famous"),
        entry("ord-1", "ordinary"),
        entry("lab-1", "lab"),
        entry("famous-2", "famous"),
        entry("ord-2", "ordinary"),
        entry("ord-fail", "ordinary"),
        entry("ord-unprocessed", "ordinary"),
    ]
    ids_ok = ["famous-1", "ord-1", "lab-1", "famous-2", "ord-2", "own-b-2222", "own-a-1111"]
    parsed_map = {
        pid: parsed(pid, contributions="We contribute X." if pid == "ord-1" else None)
        for pid in ids_ok
    }
    claims_map = {pid: paper_claims(pid, [S1, S2]) for pid in ids_ok if pid != "ord-2"}
    failures = [
        failure("ord-fail", "parse", "GROBID returned 500", TS),
        failure("ord-fail", "parse", "GROBID timeout after 300s", TS + timedelta(minutes=5)),
        failure("ord-2", "extract", "invalid JSON after 2 retries"),
        failure("own-b-2222", "extract", "own paper failure detail"),
    ]
    gold_map = {
        "ord-1": gold(
            "ord-1",
            ["Gold about a geometry benchmark.", "Gold about sparse attention memory."],
            source="paper_contributions",
            results=["500 problems", "cuts memory by 40% and 0.5 GB"],
        ),
        "lab-1": gold("lab-1", [PRIVATE_GOLD], source="lab_summary", results=["77.7%"]),
        "own-a-1111": gold("own-a-1111", [OWN_GOLD], source="author", results=["3.14 BLEU"]),
    }
    records = build_records(
        corpus,
        parsed_map,
        claims_map,
        gold_map,
        failures,
        private_gold_ids={"lab-1", "own-a-1111"},
    )
    return records, latest_failures(failures)


def _records():
    records, failures = _raw_records()
    return annotate_records(records), failures


def _render():
    records, failures = _records()
    return render_report(records, failures, TS), records


def _section(md: str, n: int) -> str:
    return md.split(f"## {n}. ")[1].split(f"## {n + 1}. ")[0]


def test_order_own_then_lab_then_ordinary_then_famous():
    records, _ = _records()
    assert [r.entry.paper_id for r in records] == [
        "own-a-1111",
        "own-b-2222",
        "lab-1",
        "ord-1",
        "ord-2",
        "ord-fail",
        "ord-unprocessed",
        "famous-1",
        "famous-2",
    ]
    md, _ = _render()
    positions = [md.index(f"## {n}. ") for n in range(1, 10)]
    assert positions == sorted(positions)
    assert md.index("## 1. Title of own-a-1111") < md.index("## 3. Title of lab-1")


def test_header_and_summary():
    md, _ = _render()
    assert "Generated 2026-09-24T12:00:00+00:00" in md
    assert "hints, not gates" in md
    assert "TF-IDF cosine" in md
    assert "cosine ≥ 0.60" in md
    assert "**Number matching rule:**" in md and "`0.5` does not match inside `0.503`" in md
    assert "| # | paper_id | tier | title | parse ok |" in md
    assert "| gold | number recall (result) | recall (result+setting) |" in md
    assert "| **total** |" in md
    assert "GROBID timeout after 300s" in md  # latest failure listed
    assert "GROBID returned 500" not in md  # superseded failure not listed
    assert "reports/private/grades.yaml" in md
    assert "Public version." not in md


def test_summary_gold_source_and_recall():
    md, _ = _render()
    rows = {line.split("|")[2].strip(): line for line in md.splitlines() if line.startswith("| ")}
    # ord-1: "500", "40%", "0.5" -> 500 and 40 found, 0.5 not (S1/S2 have no 0.5)
    assert "| paper_contributions | 2/3 (67%) |" in rows["`ord-1`"]
    assert "| lab_summary (private) | 0/1 (0%) |" in rows["`lab-1`"]
    assert "| author (private) | 0/1 (0%) |" in rows["`own-a-1111`"]
    assert "| no | — |" in rows["`famous-1`"]
    total = next(line for line in md.splitlines() if "**total**" in line)
    assert "| 3 | 2/5 (40%) |" in total  # micro: (2 + 0 + 0) / (3 + 1 + 1)


def test_arxiv_link_and_own_id():
    md, _ = _render()
    assert "[arXiv:ord-1](https://arxiv.org/abs/ord-1)" in md
    assert "`own-a-1111` · tier: **own**" in md


def test_gold_block_closest_claim_and_numbers():
    md, _ = _render()
    section = _section(md, 4)  # ord-1
    assert "### Gold vs extracted — source: paper_contributions (by tester)" in section
    assert "**Gold [0]**: Gold about a geometry benchmark." in section
    block0 = section.split("**Gold [0]**")[1].split("**Gold [1]**")[0]
    assert "- closest extracted: Claim 2 (cosine " in block0 and S2 in block0
    assert "- numbers in gold result: `500` ✓" in block0
    block1 = section.split("**Gold [1]**")[1]
    assert "- closest extracted: Claim 1 (cosine " in block1
    assert "`40%` ✓ · `0.5` ✗" in block1
    assert "Number recall (result): 2/3 (67%)" in section
    assert "matches gold: ___" in section
    assert "matches gold" not in _section(md, 8)  # famous-1 has no gold
    assert md.count("### Gold vs extracted") == 3


def test_gold_extra_fields_and_private_heading():
    records, failures = _raw_records()
    lab = next(r for r in records if r.entry.paper_id == "lab-1")
    lab.gold.claims[0] = GoldClaim(
        statement=PRIVATE_GOLD, method="SecretNet", setting="hidden set", result="77.7%"
    )
    lab.gold.source_document = "lit_review.md"
    md = render_report(annotate_records(records), failures, TS)
    section = _section(md, 3)
    assert "### Gold vs extracted — source: lab_summary (by tester) · private" in section
    assert "_Source document: `lit_review.md`_" in section
    assert "- gold fields: *method:* SecretNet · *setting:* hidden set" in section
    assert "- gold result: 77.7%" in section


def test_max_similarity_shown_and_threshold_flag():
    records, failures = _raw_records()
    near = "Sparse attention cuts memory by 40% on 16k-token long documents."
    for r in records:
        if r.entry.paper_id == "famous-1":
            r.claims = paper_claims("famous-1", [S1, near, S2])
    annotated = annotate_records(records, threshold=0.6)
    md = render_report(annotated, failures, TS, threshold=0.6)
    section = _section(md, 8)
    c1 = section.split("### Claim 1")[1].split("### Claim 2")[0]
    # 0.83: flagged at the 0.6 threshold, would not have been at the old 0.9.
    assert "- **max similarity:** 0.83 (vs claim 2) — ⚠ near_duplicate (≥ 0.60)" in c1
    assert "- ⚠ near_duplicate: similar to claim 2" in c1
    c3 = section.split("### Claim 3")[1]
    assert "- **max similarity:** 0." in c3 and "near_duplicate" not in c3.split("Flags:")[0]
    # The same pair is not flagged under the old 0.9 threshold.
    strict = render_report(annotate_records(records, threshold=0.9), failures, TS, 0.9)
    assert "⚠ near_duplicate" not in _section(strict, 8)


def test_single_claim_similarity():
    records, failures = _raw_records()
    records[0].claims = paper_claims("own-a-1111", [S1])
    md = render_report(annotate_records(records), failures, TS)
    assert "- **max similarity:** — (only claim)" in _section(md, 1)


def test_failed_papers_rendered_with_error():
    md, _ = _render()
    fail_section = _section(md, 6)
    assert "✗ parse failed: GROBID timeout after 300s" in fail_section
    extract_section = _section(md, 5)
    assert "✗ extract failed: invalid JSON after 2 retries" in extract_section
    assert "**Abstract**" in extract_section  # parse output still shown
    assert "not processed" in _section(md, 7)


def test_claim_blocks_show_all_fields_flags_and_grading():
    md, _ = _render()
    section = _section(md, 1)
    for field in (
        "statement",
        "contribution_type",
        "method",
        "problem",
        "setting",
        "result",
        "novelty_assertion",
        "evidence",
        "confidence",
        "max similarity",
    ):
        assert f"- **{field}:**" in section
    assert section.count("### Claim ") == 2
    assert "<details><summary>Critic review (1 notes)</summary>" in section
    assert "- Claim 1 — specificity (1-5): ___ · correctness (1-5): ___ · matches gold" in section
    assert "- Claim 2 — specificity" in section
    assert "_No contributions statement detected._" in section
    no_gold = _section(md, 2)
    assert "- Claim 1 — specificity (1-5): ___ · correctness (1-5): ___ · notes: ___" in no_gold


def test_contributions_statement_shown():
    md, _ = _render()
    assert "> We contribute X." in _section(md, 4)


def test_flags_rendered_under_claim():
    records, failures = _records()
    records[0].claims.claims[0].statement = "Our novel framework significantly helps."
    records[0].flags = run_checks(records[0].claims, records[0].parsed, None)
    md = render_report(records, failures, TS)
    block = md.split("### Claim 1")[1].split("### Claim 2")[0]
    assert "- ⚠ vague_phrase: matched 'novel framework'" in block
    assert "- ⚠ vague_phrase: matched 'significantly'" in block


# --------------------------------------------------------------------------- public / private


def _render_public():
    from reviewer.eval.commands import public_failures

    raw, failures = _raw_records()
    pub = annotate_records(public_records(raw))
    return render_report(pub, public_failures(failures, pub), TS, public=True), pub


def test_public_report_leaks_nothing_private():
    md, pub = _render_public()
    full, _ = _render()
    # Sanity: the private report does contain all of it.
    for needle in ("own-a-1111", "own-b-2222", PRIVATE_GOLD, OWN_GOLD, "0/1 (0%)"):
        assert needle in full
    assert "own-" not in md
    assert "own paper failure detail" not in md
    assert PRIVATE_GOLD not in md and "77.7" not in md
    assert OWN_GOLD not in md and "3.14" not in md
    assert "lab_summary" not in md.split("\n---\n", 1)[1]  # only the header explains it
    assert "(private)" not in md and "· private" not in md
    assert "0/1 (0%)" not in md  # no private recall figure
    total = next(line for line in md.splitlines() if "**total**" in line)
    assert "| 1 | 2/3 (67%) |" in total  # only ord-1's public gold counts
    assert pub[0].entry.paper_id == "lab-1"  # lab paper itself is public


def test_public_report_header_and_lab_paper_without_gold():
    md, _ = _render_public()
    assert "Public version." in md and "reports/private/" in md
    lab = _section(md, 1)
    assert "Title of lab-1" in lab
    assert "### Gold vs extracted" not in lab and "matches gold" not in lab
    row = next(line for line in md.splitlines() if line.startswith("| 1 | `lab-1`"))
    assert "| no | — |" in row
    assert md.count("### Gold vs extracted") == 1
