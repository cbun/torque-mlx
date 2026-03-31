# Agentic Coding Harness (Drop-In)

This harness is designed to be **copied into any repo** so multiple narrow agents can coordinate via **artifacts + events** rather than one “god” agent reading the entire codebase.

## Core Rules (Read This First)

- **Minimize global understanding.** Prefer changes that can be validated locally within one bounded context.
- **Contract-first.** If a boundary changes, update the contract/schema/tests before (or alongside) implementation.
- **No cross-context edits by default.** Treat cross-context changes as explicit “campaigns” with a rollback plan.
- **Prove > explain.** Every change must include a minimal proof (tests/checks) tied to acceptance criteria.
- **Prefer adapters over coupling.** Integrations live behind adapters; provider semantics must not leak into core.

## Where The Harness Lives

- `agents/`: role manifests, shared “DNA”, and the event/artifact conventions.
- `work/`: append-only work items and their artifacts (intent → scope → spec → patch → proof).

Project-specific settings live in `agents/config.yaml`. Everything else under `agents/` should be reusable across repos.

If you are writing code in this repo, follow `agents/DNA.md` and the role manifests in `agents/roles/`.
