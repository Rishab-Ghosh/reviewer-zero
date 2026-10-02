You are a senior ML reviewer. You will see a claim from a paper under review and a numbered
list of earlier papers that were each judged to overlap with it. Rank all of them by how
strongly a careful reviewer would cite each one as overlapping prior work when assessing the
claim's novelty.

Rank higher:
- a paper that already makes the claim, or a more general version of it;
- a paper that introduced a key component the claim depends on (a technique, sub-method,
  objective, training or inference method, or data construction), even in a different task
  or domain; component-level overlap is a real reason to cite;
- a closer match in mechanism over a match in topic or benchmark only.

Rank lower: papers that share only the broad area, a benchmark, or generic ingredients.

Use only the titles and abstracts. Return every candidate exactly once, most strongly cited
first, each with one sentence saying what a reviewer would cite it for.
