# Harness Initialization (New Repo)

This harness is intended to be **drop-in**. For a brand new project:

1. Copy these files/folders into the repo root:
   - `AGENTS.md`
   - `agents/`
   - `work/`
2. Edit `agents/config.yaml`:
   - set `project_name`
   - adjust `code_roots` to match your layout (once it exists)
   - add `proof_commands` once you have tooling
3. Start a work item:
   - create `work/items/<id>/intent.md`
   - append a corresponding `IntentCreated` line to `work/events.jsonl`

Notes:

- The harness stays generic by keeping repo-specific assumptions in `agents/config.yaml`.
- The manifests in `agents/roles/` list common code roots (`src/`, `apps/`, `services/`, etc.). If your repo uses different roots, update `agents/config.yaml` (and optionally the allowlists).

