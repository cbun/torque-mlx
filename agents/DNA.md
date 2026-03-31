# Agent DNA (Shared Rules)

These are the shared constraints all coding roles follow. They exist to prevent context-crash, reduce blast radius, and keep the system evolvable.

## Protocol

Every work item follows: `intent → scope → spec → patch → proof`.

- **Intent**: what we want, in user terms.
- **Scope**: bounded context(s), entrypoints, contracts impacted.
- **Spec**: acceptance criteria + invariants + risks.
- **Patch**: smallest change that satisfies the spec.
- **Proof**: commands run + tests passed + evidence.

## Boundary Discipline

- Default: **single bounded context per change**.
- Cross-context work requires an explicit **campaign** marker + rollback plan.
- Do not “reach across” contexts by importing internals. Prefer:
  - a versioned contract/schema
  - an adapter
  - an explicit public API surface

## Contract-First Development

- External boundaries must be versioned and validated:
  - event subjects/schemas
  - public module interfaces
  - API schemas (OpenAPI/JSON Schema/etc.)
- When changing a boundary:
  - update contract + tests first
  - preserve backward compatibility when feasible
  - otherwise: bump version and provide migration notes

## Safe-by-Default Behavior

- Prefer reversible changes and feature flags.
- Fail closed on ambiguous semantics.
- When uncertain, emit a clarification question rather than guessing.

## “Prove” Standard

Every patch must include:

- the acceptance criteria it satisfies
- what you ran to validate it (tests/checks)
- any residual risks or follow-ups

## Search & Retrieval Hygiene

- Prefer targeted `rg` queries over reading whole folders.
- Use `docs/` as the source of truth for “why”; update it if it’s missing.

