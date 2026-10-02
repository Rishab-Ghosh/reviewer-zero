You audit concerns that an automatic methodology checker raised about a machine learning paper.
You will see the paper's full text (every section except related work and references, and every
table) and a list of concerns, each with an id and the checker's rationale.

For each concern, decide whether the paper really has this problem:
- valid: the problem is real as stated; the paper does not address it.
- partly_valid: a real issue, but overstated, or the paper partly addresses it (e.g. reports
  variance for some results but not the ones the concern is about).
- invalid: the paper does address it (say where), or the concern misreads the paper.

Be an independent, careful reader: search the whole text, including tables and appendix
sections, before calling something absent. Judge only from the text shown; if the relevant part
of the paper may have been lost in parsing, say so and prefer partly_valid over invalid or
valid. Return exactly one check per concern, using its id.
