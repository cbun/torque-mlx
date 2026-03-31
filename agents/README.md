# Agents

This folder defines an **event-driven coding harness** for building a project with many small agents (roles) instead of one large, context-hungry agent.

The intent is to:

- Keep changes **local** (bounded context discipline).
- Encode project “memory” in **contracts + docs**, not chat history.
- Coordinate via **append-only events** and durable artifacts.

Start here:

- `agents/DNA.md`: shared behavioral constraints (the “genome”).
- `agents/config.yaml`: the only project-specific configuration (update per repo).
- `agents/events.md`: event + artifact conventions for the harness.
- `agents/roles/`: per-role manifests (what each agent can read/write and what it emits).
- `agents/init.md`: how to copy/bootstrap this harness into a new repo.
