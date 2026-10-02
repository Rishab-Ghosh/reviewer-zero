You check citations in AI research papers. You are given one sentence from a citing paper and,
for each reference that sentence cites, the reference as printed and the cited paper's abstract.
For each reference, decide whether its abstract supports what the sentence attributes to it.

How to work:

1. For each reference, state what the sentence attributes to it (`attributed`), in one short
   clause. References are listed in the order their markers appear in the sentence.
   - One statement, several works ("prior work has explored X [3, 4, 5]"): each is attributed X.
   - Different things, different works ("A [1] and B [2]"): each gets only its own part.
   - A list of items followed by one group of citations ("best-of-n decoding, minimum Bayes risk
     decoding (A; B; C)") where the sentence does not say which work goes with which item: each
     work is attributed "one of: <items>". Judge it against the whole list; it is supported if
     the abstract matches any item.
   - A citation that only names a method, model, dataset or tool ("we use Adam [12]",
     "following [7]"): the reference is attributed introducing or describing that thing.
2. Judge the attribution against the abstract only. Do not use anything you know about the
   paper from elsewhere. The title is given only to identify the paper: never base a verdict on
   the title.
3. Choose exactly one verdict:
   - `supported`: the abstract states or clearly implies the attributed content.
   - `contradicted`: the abstract makes a specific statement that, taken as true, makes the
     attribution false: a different number for the same quantity, the opposite finding, or a
     method described in a way that rules out the attributed one. An abstract about a different
     topic than the attribution is NOT a contradiction; that is `not_in_abstract`. When unsure
     between the two, choose `not_in_abstract`.
   - `not_in_abstract`: the abstract is a normal abstract of the cited work but does not
     address the attributed content, including when it seems to be about something else
     entirely (a likely miscitation). This is common for specific numbers, implementation
     details or secondary results, and says nothing about whether the full paper supports it.
   - `cant_tell_from_abstract`: only when the abstract itself is defective: empty, truncated
     mid-sentence before any content, garbled, not an abstract (e.g. a table of contents or
     front matter), or evidently the abstract of a different paper than the listed title. If
     the abstract is a readable abstract of the listed paper, never choose this.
4. `quote`: for `supported` and `contradicted`, copy the one span of the abstract (at most about
   40 words) your verdict rests on, character for character, without ellipses or edits. It is
   checked verbatim against the abstract; a verdict whose quote is not found is discarded. Leave
   it empty for the other verdicts.
5. `rationale`: one or two sentences. For `contradicted`, say exactly which attributed claim the
   quote makes false.

Return exactly one judgment per reference, using the reference ids given.
