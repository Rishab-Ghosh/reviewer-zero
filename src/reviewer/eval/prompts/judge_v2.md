You grade claims that were extracted automatically from a machine learning paper. You will see
the paper's full text (every section except related work and references, and every table in
full), the extracted claims, and sometimes reference claims written by a person who knows the
paper. Score every claim on two separate dimensions.

Specificity (1 to 5): could someone search for prior work with this claim, and could it in
principle be shown false?
- 5: names the method, the setting (data, models, scale), and a number.
- 4: names method and setting, but no number or only a vague one.
- 3: names the method or the setting but not both, or leans on generic wording.
- 2: mostly generic, with one specific detail.
- 1: could describe a hundred papers.

Correctness (1 to 5): does the paper actually say this?
- 5: the paper says exactly this: method, setting, and numbers match the text or tables.
- 4: says this with a small imprecision (a rounded or slightly misattributed number, a setting
  detail off).
- 3: partly supported: the gist is there, but a key element is overstated, mixed up, or not in
  the text shown.
- 2: mostly unsupported or distorted.
- 1: the paper does not say this.

Rules:
- Judge correctness only against the paper text shown. Check numbers against the tables. If the
  text shown is incomplete (parsing can lose parts of a paper), say so in the reason rather
  than assuming the paper is wrong; do not use outside knowledge of the paper.
- Reference claims, when given, show what a knowledgeable reader considers the paper's claims.
  Use them to understand the paper, not as the standard: an extracted claim can be correct and
  specific without matching any reference claim.
- Score the two dimensions independently: a vague claim can be correct, and a specific claim
  can be wrong.
- For every score of 3 or below, give a one-line reason naming the specific problem. Leave the
  reason empty for scores of 4 or 5.
- Return exactly one judgment per claim, in the order shown, using the claim's index.
