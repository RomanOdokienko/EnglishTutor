# ADR 0004: Formal data policies are deferred

Status: Deferred
Date: 2026-07-13

## Context

The repository retains audio, transcripts, names and analyses. Audio may be
sent to AssemblyAI and transcript-derived content may be sent to OpenAI. The
team has chosen to continue the local-development refactor before adopting
formal consent, retention or deletion policies.

## Decision

No formal data policy is adopted in this refactor phase. The data inventory and
known exposure points are documented in docs/data-scope.md. This is not an
approval to expose the write API publicly without further decisions.

## Consequences

- The project may continue local development and documentation work.
- A formal policy and access-control decision are required before a public
  production launch.
- Secrets must remain outside Git regardless of the deferred policy status.
