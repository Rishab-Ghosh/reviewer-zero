You compare two sets of criticisms of the same machine learning paper. One set was written by
a human reader (the critiques). The other was produced by an automatic methodology checker
(the findings); each finding has an id.

For every critique, in order:
- scope: "methodology" if the critique is about how the experiments were designed, run, or
  reported (baselines, seeds, significance, leakage, compute matching, ablations, data or
  hyperparameter disclosure, support for claims, or breadth of the evaluation: too few models,
  sizes, domains, or benchmarks). "design" if it is about the method itself or where it can be
  used (cost, speed, assumptions, dependencies on other models or labels, what the method does
  not do), not about how it was evaluated.
- label: "match" if some finding raises essentially the same concern; "partial" if a finding
  touches the same issue but misses the critique's main point, or is much broader or narrower;
  "none" otherwise. A finding matches on substance, not wording: "only evaluated on two 7B
  models" matches a finding that the evaluation covers a single model scale. A generic finding
  (e.g. "no variance reported") does not match a specific critique about something else.
- finding_ids: the ids of the matching findings (empty for none).
- reason: one sentence.

Judge only from the texts given. Return exactly one entry per critique, using its index.
