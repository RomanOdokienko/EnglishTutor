# Project documentation

This folder is the documentation entry point. Documents are maintained with the
code and describe the active refactor from legacy LLM-centric indicators to
versioned derived metrics.

| Document | Purpose |
| --- | --- |
| ../CLAUDE.md | Fast orientation for a fresh session: repo map, the pipeline, and the load-bearing implementation gotchas |
| ../Architecture.md | System context, components, runtime flows and deployment topology |
| migration/legacy-to-derived-v1.md | Cutover plan, compatibility and exit criteria |
| metrics-and-taxonomy.md | Canonical metric definitions, categories and versioning rules |
| contracts.md | File and HTTP contract baseline; schema migration plan |
| operations.md | Local workflow, deployment handoff and recovery actions |
| quality.md | Test and LLM-evaluation strategy |
| data-scope.md | Data inventory and explicitly deferred policy decisions |
| adr/ | Immutable records of accepted architectural decisions |

## Documentation rules

- Architecture, contracts and decisions are updated in the same change as the
  implementation that alters them.
- A new metrics or taxonomy version requires an entry in the migration log and
  a historical recomputation before comparing old and new sessions.
- Current code and generated sample data are evidence for as-is documentation;
  accepted ADRs define target-state intent when the two differ.
- Secrets never appear in documentation, examples, generated artifacts or
  committed session data.
