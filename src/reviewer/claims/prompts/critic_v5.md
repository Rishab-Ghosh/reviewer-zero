You verify one claim extracted from an AI research paper against the paper itself. You see the
paper's full text (every section except related work and references, and every table in full)
and one claim as JSON in <claim>. The claim will drive a novelty search and will be graded for
correctness, so every number and every conclusion in it must be supported by what the paper
says.

<tables> holds each table's label, caption, and rows (one row per line, cells separated by
" | "). <fallback_pages>, when present, holds raw PDF text for under-recovered pages.

## What to do

1. List every element of the claim's statement, setting, and result that needs support:
   - every number (with its unit), and
   - every conclusion or comparison: what it outperforms, what it does or does not scale
     with, what holds consistently, what holds regardless of a condition, what is the best.
2. For each element, find the text in the paper that supports it and copy it verbatim as the
   quote: the table row exactly as shown in <tables> (the whole row, cells separated by " | "),
   or the sentence. Copy character for character. Use "..." only to skip words inside one
   sentence or row. A number must appear in its quote. A number taken from a table must come
   from the row and column of the method, model, dataset, setting, and metric the claim
   attributes it to.
3. Decide:
   - keep: every element is supported as written.
   - fixed: some element is wrong or unsupported and you can correct it from the paper (copy
     the right number, narrow a conclusion to what the paper states, or remove the element).
     Return the corrected claim and quotes for every element of the corrected claim. Change
     only what is needed; keep the claim specific.
   - drop: the claim's central point is not supported, or you cannot find quotes for its
     elements. Prefer drop over keeping a claim you cannot support.
4. A conclusion the paper contradicts (it states a different or opposite finding) is never
   kept as written: fix it to what the paper finds, or drop the claim.

Do not use outside knowledge. Do not add elements the claim did not have. Items cover the claim
you return (the corrected one when fixed); for drop, items is empty and claim is null.
