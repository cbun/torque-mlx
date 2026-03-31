# Event + Artifact Conventions (Coding Harness)

The harness is **event-driven**. Agents coordinate by appending events to a log and producing artifacts in a work-item folder.

## Event Log

- File: `work/events.jsonl`
- Format: one JSON object per line.
- Events are append-only. Corrections are new events.

### Envelope (v1)

Fields (suggested minimum):

- `kind`: event type (string)
- `subject`: work item id (string)
- `time`: ISO timestamp (string)
- `by`: role id (string)
- `payload`: structured data (object)

Example:

```json
{"kind":"IntentCreated","subject":"work/2026-02-03-001","time":"2026-02-03T19:40:00Z","by":"boss","payload":{"summary":"Add onboarding flow skeleton"}} 
```

## Work Items

- Folder: `work/items/<id>/`
- Artifacts are markdown or machine-readable files.
- Agents should write small artifacts that reference paths and contracts.

### Standard Artifacts

- `intent.md`
- `scope.md`
- `spec.md`
- `patch.md` (summary + pointers to diffs)
- `proof.md`
- `review.md` (if needed)

## Core Event Kinds (Minimal Ontology)

- `IntentCreated`
- `ScopeResolved`
- `SpecProposed`
- `SpecApproved`
- `ContractProposed`
- `ContractApproved`
- `ImplementationProposed`
- `ReviewRequested`
- `ChangesRequested`
- `Approved`
- `ProofRequested`
- `ProofGreen`
- `ProofRed`
- `WorkBlocked` (needs clarification, missing access, etc.)

Keep this list small. Prefer adding metadata to payloads over inventing new kinds.

