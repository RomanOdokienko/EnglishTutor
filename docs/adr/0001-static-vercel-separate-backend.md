# ADR 0001: Static Vercel frontend with a separate backend

Status: Accepted
Date: 2026-07-13

## Context

The frontend can be published as static files, but upload, recording,
transcription, session deletion, analysis and rebuild actions need Python,
persistent writable storage and provider secrets. A static Vercel deployment
cannot be the writable session store.

## Decision

Deploy out/web as a static Vercel artifact. Deploy server.py and the session
archive on a separate persistent backend. Continue using one local Python server
for development until the backend is deployed.

The generated frontend receives the backend base URL through
ENGLISH_TUTOR_API_BASE_URL. The backend permits the exact frontend origin
through ENGLISH_TUTOR_CORS_ORIGIN.

## Consequences

- No OpenAI or AssemblyAI key is exposed to Vercel or browser code.
- The backend needs persistent storage that survives restarts and deployments.
- Vercel deployment is an artifact publication step, not a backend deployment.
- The backend API needs access control before public production use.
- Local development remains supported without a remote backend.
