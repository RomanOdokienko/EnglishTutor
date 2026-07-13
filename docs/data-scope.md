# Data scope and deferred policy decisions

## Current data inventory

| Data | Location | External transfer |
| --- | --- | --- |
| Participant names and speaker aliases | data/people.json and session metadata | May be included with analysis context |
| Session metadata | sessions/<date>/meta.json | May be included with analysis context |
| Conversation transcript | sessions/<date>/transcript.txt and analysis outputs | Sent to OpenAI for optional analysis and annotations |
| Raw recorded audio | sessions/<date>/audio.<ext> | Sent to AssemblyAI for transcription |
| Analysis, annotations and exercises | out/sessions and out/history | Model-generated content may originate from OpenAI |
| Static publishing copy | out/web | Published through Vercel when deployed |

The accepted repository policy is to commit inputs and generated artifacts,
including raw audio and transcripts.

## Explicitly deferred

No formal retention period, participant-consent process, deletion workflow
beyond manual session deletion, data-processing agreement, or public
privacy-policy text has been adopted yet.

This deferral permits local development; it does not remove the sensitivity of
the data or the need for a decision before public production use.

## Non-negotiable controls during the deferral

- Do not commit API keys, tokens, cookies or provider responses containing
  credentials.
- Restrict repository access to people authorised to access session material.
- Treat every audio file and transcript as sensitive when sharing branches,
  logs, screenshots or issue reports.
- Audit any untracked file whose name suggests a token or credential before
  adding it to Git.
- Do not expose unauthenticated write endpoints to the public internet.

## Future decision checklist

Before public launch, decide who can record, upload, view and delete sessions;
what consent is required; how long audio/transcripts are retained; how a
participant requests deletion; which provider terms apply; and what authentication
protects the backend.
