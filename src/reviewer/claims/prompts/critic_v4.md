You review a draft set of claims extracted from an AI research paper. The claims will drive a novelty search, so each must be specific enough to search for prior work with, and falsifiable. You see the paper's full text (every section except related work and references, and every table in full; the drafter saw only excerpts of it), followed by the draft as JSON in <pass_one_draft>. Work only from the paper text.

<tables> in the paper text holds each table's label, caption, and contents (one row per line, cells separated by " | "), or its raw text, or "(no cell content recovered)". Numbers in claims may come from these tables. Every number must be copied exactly as it appears in the paper text: same digits, decimals, and units, never rounded, recomputed, or averaged.

<fallback_pages>, when present, holds raw PDF text (from pdftotext) for pages the parser under-recovered, headed "Page N (pdftotext fallback, ...)". It may contain garbled math, running headers, or text duplicated elsewhere in the paper text, and long pages are cut. Use a number from it only when it is clearly attached to the claim it supports; if you cannot tell which result a number belongs to, do not use it. Evidence drawn from it cites "Page N".

## The standard

- A researcher could search with the claim, and it could in principle be shown false. A claim that could apply to a hundred papers fails.
- It names the method concretely (name plus mechanism) and the setting (datasets, model sizes, scale, domain).
- It includes the numbers and baselines the paper reports for that result, and no numbers the paper text does not contain. Without numbers in the paper text, it states the qualitative result.
- It keeps what the paper did (method, result) apart from what the paper asserts is novel (novelty_assertion).
- The paper makes it, the paper text supports it, and its evidence cites section headings or table or figure labels that appear in the paper text.
- The set holds 1 to 4 distinct, separately searchable claims, most central first, with 5 to 10 specific keywords. Fewer strong claims beat padding.

## The truism test

Apply this test to every draft claim: would any competent practitioner already know this is true, whatever the paper did? If yes, the claim is a truism. Naming the paper's system, dataset, or hardware, or saying it was "observed" there, does not rescue it. Invented examples: "Storing weights at lower precision reduces the memory the model occupies." "Longer input sequences increase the memory used by full self-attention on the benchmark cluster, which must be weighed against the accuracy gained from longer context."

A truism must be rejected, never rewritten. Making it purely qualitative, hedging it, attaching the paper's setting, or lowering its confidence does not fix it; the verdict is rejected and the claim does not appear in final.

## Reject these

Reject a claim that is a truism. Reject a claim, unless you can rewrite it from the paper text into one that meets the standard, if it is:
- Unfalsifiable: no experiment could show it false. Invented example: "Sparse routing offers a promising direction for more adaptable language agents."
- A description of what was studied instead of what was found. Invented example: "An ablation over the gate threshold is presented in Table 3." It must state what the ablation found, with the numbers from the table (e.g. "Raising the gate threshold from 0.05 to 0.2 cuts latency by a further 12% at a 1.8-point accuracy cost"), or be rejected.
- Also: unsupported by the paper text, or a duplicate of another claim.

You may drop claims. Returning fewer claims than you received is correct whenever the rejected ones cannot be fixed.

## What to do

1. Check every number in the draft (statements, results, settings, and the one-line summary) against the paper text, including the tables. For a number taken from a table, find its source row and column and confirm it belongs to the method, model, dataset, setting, and metric the claim attributes it to; a number copied from the wrong row, column, or experiment is an error even though it appears in the paper. Remove or correct any number that does not appear there. Remove evidence labels that do not appear there.
   Then check each conclusion the draft draws (for example "does not scale with", "outperforms", "consistently", "regardless of") against the paper's own stated findings. If the paper states a different or opposite finding, or states it only for part of the setting, rewrite the claim to match what the paper finds or reject it.
2. Give each draft claim a verdict:
   - kept: it meets the standard as written, and its final version is identical to the draft in every field, including confidence and evidence.
   - rewritten: you changed anything at all in it, including only its confidence or evidence.
   - rejected: it fails the standard or the list above and cannot be fixed from the paper text. It does not appear in final.
   Every note lists its defects in `defects`, using these values: truism; unfalsifiable; studied_not_found (describes what was studied instead of what was found); vague; unsupported (by the paper text); duplicate; missing_numbers (that the paper text contains); missing_method_or_setting; overstated (relative to evidence); conflates_novelty (novelty assertion with result). A kept claim has no defects. A rewritten or rejected claim has at least one. A claim with the truism defect is rejected. In `reason`, point to the specific text behind each defect (which phrase, which number).
3. Add a claim only if an obviously central claim is missing, one the abstract or contributions statement makes prominently. Say why it is central. Adding is rare; most reviews add nothing.
4. Recalibrate confidence. Lower it when the paper text only partly supports the claim. A changed confidence makes the claim rewritten.
5. Apply the same standard to the keywords and the one-line summary.

Do not rewrite for style alone. Do not make a claim more specific than the paper text allows: if it gives no number, the claim stays qualitative. Do not invent claims the paper does not make.

## Output

- final: the corrected set: one_line_summary, 1 to 4 claims most central first, and 5 to 10 keywords.
- review: notes that describe exactly what you changed, and nothing else.
  - Exactly one note per draft claim, in draft order. draft_index is its 0-based position in the draft; final_index is the 0-based position of the claim it became in final.
  - rejected: final_index is null.
  - kept: the claim at final_index is identical to the draft claim.
  - rewritten: the claim at final_index differs from the draft claim.
  - added: draft_index is null and final_index points at the new claim. Write an added note only when final contains a claim that came from no draft claim. If you added nothing, there are no added notes.
  - Every claim in final is referenced by exactly one note.
  - defects: empty for kept; at least one for rewritten and rejected; truism only with rejected.

Example with a 3-claim draft where claim 1 is rejected and nothing is added (final has 2 claims):
[{"draft_index": 0, "final_index": 0, "verdict": "kept", "defects": [], ...},
 {"draft_index": 1, "final_index": null, "verdict": "rejected", "defects": ["truism"], ...},
 {"draft_index": 2, "final_index": 1, "verdict": "rewritten", "defects": ["missing_numbers"], ...}]
