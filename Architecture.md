# Architecture overview — English Tutor

## Status and scope

This is the canonical architecture overview for the current refactor. Detailed
contracts, migration steps and operational instructions are linked from
[docs/README.md](docs/README.md).

The product analyses English conversation sessions, surfaces progress indicators
and helps participants choose practice exercises. The current implementation is
local-first for development. The accepted production topology is a static
frontend on Vercel with a separately deployed backend.

### Accepted architectural decisions

- The canonical progress series is participant.derived. Legacy LLM summary
  fields remain only as temporary compatibility data.
- The metrics and grammar taxonomy are versioned. A change that affects
  comparability increments its version and triggers a historical re-analysis.
- sessions/, data/ and out/ are versioned in Git. Raw audio and transcripts are
  retained with the session.
- web/ contains frontend source. out/web/ is generated and is committed so it
  can be published as the static Vercel artifact.
- Formal privacy, retention and consent policies are explicitly deferred. This
  is a known risk, not an implicit policy.

## System context

~~~mermaid
flowchart LR
  U[Session participant] --> B[Browser]
  B -->|static HTML, CSS and JS| V[Vercel static frontend]
  B -->|HTTPS API calls| S[English Tutor backend]
  S <--> F[(Versioned files: sessions, data and out)]
  S --> O[OpenAI Responses API]
  S --> A[AssemblyAI transcription API]
~~~

During local development the same Python server supplies the generated frontend
from out/web and the API. In production Vercel supplies only the static
frontend; all write operations and all access to files or provider keys happen
on the backend.

## Containers and responsibilities

| Container | Responsibility | Durable data it owns |
| --- | --- | --- |
| Browser frontend | Record audio, upload transcripts, show session insights and progress, invoke API actions | Browser-local configured API base and optional UI cache |
| Static frontend artifact | Publish the generated pages and session views | out/web; copied from web/ and enriched with the API base |
| Python backend | Validate requests, create/delete sessions, invoke analysis, rebuild artifacts and expose static data locally | Accesses all repository data; has provider credentials through environment variables |
| Analysis pipeline | Parse transcripts, call optional LLM steps, derive canonical metrics, build history and the pre-call briefing | out/sessions/<date>/analysis.json, out/history.json and out/briefing.json |
| Session archive | Preserve source material for reproducibility | sessions/<date>/meta.json, transcript.txt and optional audio.<ext> |
| OpenAI | Optional model analysis, annotations and practice exercises | No project data at rest in this repository; receives request payloads |
| AssemblyAI | Audio transcription for recorded sessions | Receives raw audio |

## Source of truth and generated artifacts

| Location | Role | Versioned |
| --- | --- | --- |
| sessions/<date>/ | Original session record: metadata, transcript and optional raw audio | Yes |
| data/people.json | Speaker-label aliases | Yes |
| web/ | Editable frontend source | Yes |
| out/sessions/<date>/analysis.json | Derived session analysis | Yes; reproducible from retained input where providers are not needed |
| out/history.json | Derived cross-session index | Yes |
| out/web/ | Generated static publishing artifact | Yes |

out/ must never be edited by hand. The build copies files from web/ into
out/web, injects ENGLISH_TUTOR_API_BASE_URL into HTML templates, copies history
and publishes session analyses for static reading.

## Main runtime flows

### Text upload

1. The browser posts a transcript, optional date/topic/duration and speaker
   mapping to the backend.
2. The backend writes the session inputs and updates the alias registry.
3. The analysis pipeline produces or updates the per-session analysis, history
   and generated frontend artifact.
4. The browser loads history and analysis JSON from the configured API base.

### Audio recording

1. The browser records audio and sends bytes to the backend.
2. The backend saves audio.<ext> in the dated session folder.
3. The backend requests multichannel transcription from AssemblyAI, creates the
   transcript and then follows the text-upload analysis flow.

### Metrics migration and historical comparison

1. The pipeline derives metrics from transcript, speaker_map and stored
   annotation_items without a new LLM request.
2. It writes analysis_version.metrics and analysis_version.taxonomy with every
   output.
3. The reanalyze endpoint or CLI recomputes all saved analyses when a formula
   or taxonomy changes.
4. Progress UI reads participant.derived. Legacy LLM fields are not the
   canonical series.

## Deployment model

| Environment | Frontend | Backend | Data and secrets |
| --- | --- | --- | --- |
| Local development | Served from out/web by Python server | Python server on loopback by default | Local repository and environment variables |
| Production target | Vercel serves out/web | A separately hosted persistent Python service | Backend-attached persistent storage and backend environment variables |

The production frontend must be built with the public backend URL in
ENGLISH_TUTOR_API_BASE_URL. The backend must allow that exact Vercel origin
through ENGLISH_TUTOR_CORS_ORIGIN. Vercel is not a writable session store and
must not receive OpenAI or AssemblyAI secrets.

## Cross-cutting constraints

- Date identifiers are YYYY-MM-DD and are validated before destructive session
  deletion.
- API keys remain environment variables; no credential is stored in source,
  generated assets or committed session data.
- The backend currently exposes write endpoints without application
  authentication. Public production exposure requires a separate access-control
  decision before launch.
- Recorded audio, transcripts, names and model annotations are sensitive
  project data. The currently deferred policy decision is recorded in
  docs/data-scope.md.

## Known refactor boundaries

- web/ is the replacement source tree for the previously edited generated web
  files.
- Old LLM fluency and grammar summary fields are retained for compatibility
  while derived metrics v1 become the trusted series.
- The existing JSON Schema files describe the previous output shape and are not
  yet the authoritative validation contract. docs/contracts.md defines the
  migration requirement.
