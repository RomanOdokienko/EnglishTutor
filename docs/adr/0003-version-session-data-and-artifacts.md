# ADR 0003: Version session data and generated artifacts in Git

Status: Accepted
Date: 2026-07-13

## Context

The project needs reproducible analysis and a static artifact that can be
published without reconstructing the historical archive. Raw audio,
transcripts, metadata, analyses and out/web provide that record.

## Decision

Commit sessions/, data/ and out/ to Git, including raw audio and transcripts.
web/ is the editable source; out/web is generated and committed as the Vercel
publishing artifact.

## Consequences

- Repository history contains sensitive session material and grows with audio.
- Before large audio files become routine, the team must decide and configure a
  large-file mechanism such as Git LFS and document its restore workflow.
- Generated artifacts are rebuilt before a release and never hand-edited.
- A future data-policy ADR may supersede retention aspects of this decision.
