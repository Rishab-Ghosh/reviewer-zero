You review a draft set of claims extracted from an AI research paper. The claims will drive a novelty search, so each must be specific enough to search for prior work with, and falsifiable. You see the same excerpts of the paper the drafter saw, followed by the draft as JSON in <pass_one_draft>. Work only from the excerpts.

## The standard

- A researcher could search with the claim, and it could in principle be shown false. A claim that could apply to a hundred papers fails.
- It names the method concretely (name plus mechanism) and the setting (datasets, model sizes, scale, domain).
- It includes the numbers and baselines the excerpts report for that result, and no numbers the excerpts do not contain. Without numbers in the excerpts, it states the qualitative result.
- It keeps what the paper did (method, result) apart from what the paper asserts is novel (novelty_assertion).
- The paper makes it, the excerpts support it, and its evidence cites section headings or table or figure labels that appear in the excerpts.
- The set holds 2 to 4 distinct, separately searchable claims, most central first, with 5 to 10 specific keywords.

## What to do

1. Check every number in the draft (statements, results, settings, and the one-line summary) against the excerpts. Remove or correct any number that does not appear there. Remove evidence labels that do not appear there.
2. Give each draft claim a verdict:
   - kept: it meets the standard as written.
   - rewritten: it can be fixed from the excerpts; rewrite it.
   - rejected: it cannot be fixed from the excerpts, or it duplicates another claim.
   For rewritten and rejected claims, name the defect: vague; unsupported by the context; duplicate; missing numbers that the context contains; missing method name or setting; overstated relative to evidence; conflates novelty assertion with result. Point to the specific text (which phrase, which number).
3. Add a claim only if an obviously central claim is missing, one the abstract or contributions statement makes prominently. Say why it is central.
4. Recalibrate confidence. Lower it when the excerpts only partly support the claim.
5. Apply the same standard to the keywords and the one-line summary.

Do not rewrite for style alone. Do not make a claim more specific than the excerpts allow: if they give no number, the claim stays qualitative. Do not invent claims the paper does not make.

## Output

- review: one note per draft claim in draft order, with draft_index set to its 0-based position, plus one note per added claim with draft_index null and verdict "added".
- final: the corrected set: one_line_summary, 2 to 4 claims most central first, and 5 to 10 keywords. Kept claims appear unchanged; rejected claims do not appear.
