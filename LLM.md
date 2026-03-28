# LLM-Assisted Development

This project is built using LLM-based coding tools, primarily Claude Code
and Codex.

Clean-room development — where one team studies a reference implementation
and produces a specification, and a separate team builds from that
specification — has a long history in open source: Compaq's IBM BIOS clone,
WINE, Samba, and Mono were all built this way. This project draws on the
same principle, with LLMs as the separation boundary.

We have no control over or insight into the training data of the LLMs we
use, and we recognize that this is not a perfect implementation of the
clean-room process. It is as close as we can get, and we built dedicated
tooling to enforce it.

## Process

LLM agents are used throughout development — writing code, debugging, testing,
and reviewing. The human developer directs the work, reviews and approves code
changes, and makes architectural decisions.

## Third-party isolation

Where this project operates in a space with existing implementations under
different licenses, a structural separation is maintained:

- The **development agent** — the LLM session that writes and modifies source
  code — never sees third-party source code.
- Separate, persistent **review sessions** are maintained for each relevant
  third-party project. These sessions have context about third-party
  implementations and are consulted for analysis, comparison, and critique —
  but they produce analysis and critique, never code, pseudocode, or
  structural blueprints.
- This separation is enforced by tooling
  ([review](https://github.com/folknor/review)), not by convention alone.

This mirrors the clean-room design pattern: one team reads the reference
material and produces a specification, a separate team writes the
implementation from the specification.

## Why this document exists

The legal status of LLM-generated code is unsettled. This document exists to
be transparent about how LLMs were used and what steps were taken to avoid
introducing code derived from differently-licensed sources. This is a
risk-reduction process, not a guarantee of non-infringement.
