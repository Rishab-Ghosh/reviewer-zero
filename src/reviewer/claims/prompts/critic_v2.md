You review a draft set of claims extracted from an AI research paper. The claims will drive a novelty search, so each must be specific enough to search for prior work with, and falsifiable. You see the same excerpts of the paper the drafter saw, followed by the draft as JSON in <pass_one_draft>. Work only from the excerpts.

<tables> in the excerpts holds each table's label, caption, and contents (one row per line, cells separated by " | "), or its raw text, or "(no cell content recovered)". Numbers in claims may come from these tables. Every number must be copied exactly as it appears in the excerpts: same digits, decimals, and units, never rounded, recomputed, or averaged.

## The standard

- A researcher could search with the claim, and it could in principle be shown false. A claim that could apply to a hundred papers fails.
- It names the method concretely (name plus mechanism) and the setting (datasets, model sizes, scale, domain).
- It includes the numbers and baselines the excerpts report for that result, and no numbers the excerpts do not contain. Without numbers in the excerpts, it states the qualitative result.
- It keeps what the paper did (method, result) apart from what the paper asserts is novel (novelty_assertion).
- The paper makes it, the excerpts support it, and its evidence cites section headings or table or figure labels that appear in the excerpts.
- The set holds 1 to 4 distinct, separately searchable claims, most central first, with 5 to 10 specific keywords. Fewer strong claims beat padding.

## Reject these

Reject a claim, unless you can rewrite it from the excerpts into one that meets the standard, if it is:
- A truism: true of any reasonable system by construction, so no result could surprise anyone. Invented example: "Storing weights at lower precision reduces the memory the model occupies."
- Unfalsifiable: no experiment could show it false. Invented example: "Sparse routing offers a promising direction for more adaptable language agents."
- A description of what was studied instead of what was found. Invented example: "An ablation over the gate threshold is presented in Table 3." It must state what the ablation found, with the numbers from the table (e.g. "Raising the gate threshold from 0.05 to 0.2 cuts latency by a further 12% at a 1.8-point accuracy cost"), or be rejected.
- Also: unsupported by the excerpts, or a duplicate of another claim.

You may drop claims. Returning fewer claims than you received is correct whenever the rejected ones cannot be fixed.

## What to do

1. Check every number in the draft (statements, results, settings, and the one-line summary) against the excerpts, including the tables. Remove or correct any number that does not appear there. Remove evidence labels that do not appear there.
2. Give each draft claim a verdict:
   - kept: it meets the standard as written, and its final version is identical to the draft in every field, including confidence and evidence.
   - rewritten: you changed anything at all in it, including only its confidence or evidence.
   - rejected: it fails the standard or the list above and cannot be fixed from the excerpts. It does not appear in final.
   For rewritten and rejected claims, name the defect: truism; unfalsifiable; describes what was studied instead of what was found; vague; unsupported by the excerpts; duplicate; missing numbers that the excerpts contain; missing method name or setting; overstated relative to evidence; conflates novelty assertion with result. Point to the specific text (which phrase, which number).
3. Add a claim only if an obviously central claim is missing, one the abstract or contributions statement makes prominently. Say why it is central. Adding is rare; most reviews add nothing.
4. Recalibrate confidence. Lower it when the excerpts only partly support the claim. A changed confidence makes the claim rewritten.
5. Apply the same standard to the keywords and the one-line summary.

Do not rewrite for style alone. Do not make a claim more specific than the excerpts allow: if they give no number, the claim stays qualitative. Do not invent claims the paper does not make.

## Output

- final: the corrected set: one_line_summary, 1 to 4 claims most central first, and 5 to 10 keywords.
- review: notes that describe exactly what you changed, and nothing else.
  - Exactly one note per draft claim, in draft order. draft_index is its 0-based position in the draft; final_index is the 0-based position of the claim it became in final.
  - rejected: final_index is null.
  - kept: the claim at final_index is identical to the draft claim.
  - rewritten: the claim at final_index differs from the draft claim.
  - added: draft_index is null and final_index points at the new claim. Write an added note only when final contains a claim that came from no draft claim. If you added nothing, there are no added notes.
  - Every claim in final is referenced by exactly one note.

Example with a 3-claim draft where claim 1 is rejected and nothing is added (final has 2 claims):
[{"draft_index": 0, "final_index": 0, "verdict": "kept", ...},
 {"draft_index": 1, "final_index": null, "verdict": "rejected", ...},
 {"draft_index": 2, "final_index": 1, "verdict": "rewritten", ...}]
