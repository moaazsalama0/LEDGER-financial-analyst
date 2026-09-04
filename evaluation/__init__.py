"""Phase 2: scoring what the parser actually recovered against TAT-DQA gold.

This package is not part of the service. It imports ``app`` and the contract,
never the other way round, so nothing here can affect what ``/process``
returns. It lives in the repository because the number it produces is the one
that settles a backend choice, and a number nobody can reproduce settles
nothing.

What is being measured, precisely: TAT-DQA gives, for every question, the
literal strings a reader must find in the document to answer it (``facts``),
and the magnitude the figures are printed in (``scale``). A fact the parser
did not recover cannot be retrieved, cited, or calculated with, however good
the rest of the pipeline is. So fact recoverability is the ceiling on
end-to-end accuracy attributable to document processing, and it is measurable
today, before retrieval or the agent exist.

See ``evaluation.matching`` for exactly what counts as recovered, including
the ways this metric can flatter a parser.
"""

from __future__ import annotations

__all__ = ["dataset", "harness", "matching", "report", "textonly_baseline"]
