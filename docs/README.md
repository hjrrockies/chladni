# What is in here, and what is not

Four kinds of document live under `docs/`, and they are separated by **half-life** rather
than by whether they were written before or after the code. The test each one has to pass:

> Would a stranger reading the code *at this commit* be helped by it, and is it still true
> *at this commit*?

Yes to both, and it belongs here. Helpful but true only as of a date, and it belongs in
`notes/`, which is not tracked.

### `conventions.md` --- a contract

Notation, array shapes, orientation, the boundary-condition encoding, and which symbol means
what. Not a record of a past decision: a live constraint the code has to satisfy, which is
why it sits beside the code in version control rather than in `design/`. Shipped docstrings
cite it.

### `bessel_evaluation.tex` / `.pdf` --- exposition

Why the code is correct, at a level of detail that does not fit in a docstring. The `.pdf` is
tracked as well as the source so a reader needs no TeX installation; rebuild with
`latexmk -pdf` and then `latexmk -c`.

### `design/` --- design records

One per layer, each written **before** its implementation as the spec the tests were built
against, then grown an **"As built: deltas from this spec"** section rather than being
rewritten.

That convention is what makes these worth keeping. A design document that is edited to match
the code tells you nothing about why the code is what it is, and a design document that is
not edited quietly lies. Appending does neither: the body records what was intended, the
delta section records what the implementation found, and the gap between them is usually the
most useful thing in the file --- it is where the measurements are, and the alternatives that
were tried and rejected.

So: **do not edit above the delta section.** Supersede, do not revise.

### Not here: `TODO.md`

The backlog sits at the root and is tracked. It passes the half-life test the way
`conventions.md` does --- not by being permanent, but by being **kept true at HEAD**: an item
is deleted when it is done, never annotated as done. What was outstanding on a past date is
what `git log` is for. It is at the root rather than under `docs/` because it is addressed to
whoever picks the work up, not to a reader of the code.

### Not here: `notes/`

Working notes, research memos, work orders, benchmark run logs --- anything addressed to a
particular moment and a reader who no longer exists. Useful while it is useful; not part of
what the repository claims to be true. `CLAUDE.md` is ignored separately, being tooling
configuration rather than documentation.

### Later

User-facing documentation is a fifth kind, with tracked source and an ignored build
directory. `docs/_build/` is already ignored for it.
