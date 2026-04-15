# Paperless-ngx ML UI Integration

**Milestone:** Initial implementation (April 2026)
**Scope:** Frontend integration of two complementary ML features into the Paperless-ngx web UI
**Status:** Components built, wired to the data and serving contracts, running in a custom Docker image

## 1. Overview

This document describes the UI layer of the Paperless-ngx ML platform. The UI extends the open-source Paperless-ngx document management system with two pages that surface the team's ML features to end users:

- **Handwriting review** — a queue of flagged low-confidence HTR transcriptions that users can correct. Corrections become labelled `(image, text)` pairs for the HTR retraining loop.
- **Semantic search** — a natural-language search interface that calls the retrieval model and collects relevance feedback (clicks, thumbs up / thumbs down) for the retrieval retraining loop.

Both pages are implemented as new Angular routes inside a fork of the upstream Paperless-ngx repository. The fork is built into a custom Docker image and deployed via Docker Compose on top of the stock Paperless-ngx service (PostgreSQL, Redis, webserver). The UI talks to the team's FastAPI serving layer for inference and to the Paperless-ngx backend for feedback persistence; both integration points fall back to mock data when unreachable so the UI remains usable for development and demos.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Paperless-ngx Angular frontend (custom build)       │        │
│  │  ─ existing pages (dashboard, docs, settings, …)     │        │
│  │  ─ /ml/htr-review   ← new                            │        │
│  │  ─ /ml/search       ← new                            │        │
│  └──────────────────────────────────────────────────────┘        │
└──────┬────────────────────────┬─────────────────────────────────┘
       │                        │
       │ /api/ml/*              │ /ml-api/predict/*
       │ (data team scope)      │ (serving team scope)
       ▼                        ▼
┌──────────────────────┐   ┌──────────────────────┐
│  Paperless webserver │   │  FastAPI serving     │
│  Django + DRF        │   │  TrOCR + sentence-   │
│  + new ML views      │   │  transformers        │
│                      │   │                      │
│  ─ writes to:        │   │  ─ /predict/htr      │
│    Postgres          │   │  ─ /predict/search   │
│    Redpanda topics   │   │  ─ /health           │
└──────────────────────┘   └──────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────────┐        ┌──────────────────┐
│  Postgres + MinIO│        │  Qdrant          │
│  Redpanda        │        │  (vector index)  │
└──────────────────┘        └──────────────────┘
```

The UI is deliberately split into two integration points so that serving and data can evolve independently. Inference calls go to `/ml-api/*` (proxied to the FastAPI serving layer). Feedback writes go to `/api/ml/*` (served by new Django views on the Paperless backend).

## 3. Components

### 3.1 Handwriting review (`/ml/htr-review`)

**Purpose.** Surface the queue of regions where the HTR model's confidence fell below the flag threshold, allow users to correct the transcription, and capture the correction as a labelled training example.

**Data flow.**
1. Page load fetches `GET /api/ml/htr/queue/` and renders a list of documents, each with one or more regions.
2. For each region, the UI displays the predicted `htr_output`, a confidence badge (green ≥85%, yellow ≥60%, red below), and an editable textarea pre-filled with the prediction.
3. Clicking **Save correction** posts to `POST /api/ml/htr/corrections/` with the full correction payload.
4. A global "Contribute to training" switch in the page header drives the `opted_in` field on each correction, respecting the data contribution preference described in the data design document.

**Expected queue response shape:**
```json
[
  {
    "document_id": "a3f7c2e1-9b4d-4e8a-b5c6-1234567890ab",
    "title": "Invoice — Acme Corp 2026-03",
    "uploaded_at": "2026-04-13T09:42:00Z",
    "regions": [
      {
        "region_id": "c9d3e5a7-6b2f-4c0d-8e4a-fedcba987654",
        "page_id": "b8e2d4f6-7a3c-4b1e-9d5f-abcdef012345",
        "crop_s3_url": "s3://paperless-images/.../c9d3e5a7.png",
        "htr_output": "Total: $5,OOO",
        "htr_confidence": 0.54
      }
    ]
  }
]
```

**Correction payload:**
```json
{
  "region_id": "c9d3e5a7-6b2f-4c0d-8e4a-fedcba987654",
  "document_id": "a3f7c2e1-9b4d-4e8a-b5c6-1234567890ab",
  "original_text": "Total: $5,OOO",
  "corrected_text": "Total: $5,000",
  "opted_in": true
}
```

Field names align with the `handwritten_regions` and `htr_corrections` tables from the data design document so the backend writes a row and (per the data doc) emits a `paperless.corrections` Redpanda event with no field translation.

### 3.2 Semantic search (`/ml/search`)

**Purpose.** Give users a natural-language search interface over merged OCR + HTR text and capture the relevance signals needed to retrain the retrieval bi-encoder.

**Data flow.**
1. User types a query, selects a mode (semantic / keyword / hybrid), and submits.
2. The UI generates a client-side `session_id` on first search (UUID v4) and reuses it for feedback on that result set.
3. The request goes to `POST /ml-api/predict/search` with the exact payload from the serving contract: `{session_id, query_text, user_id, top_k}`.
4. Results render as cards with the document ID (truncated), chunk index, chunk text, similarity score, and feedback buttons. Clicks navigate to the document view and fire-and-forget a `click` feedback event. Thumbs buttons fire explicit `thumbs_up` or `thumbs_down` feedback events.
5. If the serving response flips `fallback_to_keyword` to true (max similarity below threshold), a warning banner recommends switching to keyword mode.
6. Model version and inference time are displayed below the result count for observability.

**Feedback payloads** are written to `POST /api/ml/search/feedback/` and map directly to the `search_feedback` table:
```json
{
  "session_id": "...",
  "document_id": "...",
  "feedback_type": "click" | "thumbs_up" | "thumbs_down"
}
```

## 4. Contracts

### 4.1 Serving (inference)

Both contracts are owned by the serving team and specified in `contracts/*.json` and `serving/src/fastapi_app/app.py`.

- **`POST /predict/search`**
  - Request: `{session_id, query_text, user_id, top_k}`
  - Response: `{session_id, query_text, results: [{document_id, chunk_index, chunk_text, similarity_score}], fallback_to_keyword, model_version, inference_time_ms}`
- **`POST /predict/htr`**
  - Request: `{document_id, page_id, region_id, crop_s3_url, image_base64?, image_width?, image_height?, …}`
  - Response: `{region_id, htr_output, htr_confidence, htr_flagged, model_version, inference_time_ms}`
  - Not called directly by the UI in this milestone. HTR inference happens at upload time, driven by the ingestion pipeline, not by the review page.

### 4.2 Data (feedback and queue)

These endpoints do not yet exist. They are the Paperless backend views the data role needs to implement to close the feedback loop. Each endpoint also emits a corresponding Redpanda event per the data design document.

| Endpoint | Method | Purpose | Writes to | Emits topic |
|---|---|---|---|---|
| `/api/ml/htr/queue/` | GET | Return flagged regions without corrections | — (read-only join) | — |
| `/api/ml/htr/corrections/` | POST | Save a user correction | `htr_corrections` | `paperless.corrections` |
| `/api/ml/search/feedback/` | POST | Save click / thumbs feedback | `search_feedback` | `paperless.feedback` |

Until these views land, both pages detect 404s and fall back to static mock data (three mock documents for review, five mock chunks for search). A banner on each page indicates when the mock fallback is active.

## 5. Implementation details

### 5.1 Repository layout

The fork lives at `paperless-ngx-fork/` alongside the compose deployment folder `paperless/`. New files are scoped to `src-ui/src/app/components/ml/`:

```
paperless-ngx-fork/
└── src-ui/src/app/
    ├── app-routing.module.ts                 (modified: 2 new routes)
    └── components/
        ├── app-frame/app-frame.component.html (modified: 2 sidebar links)
        └── ml/
            ├── htr-review/
            │   ├── htr-review.component.ts
            │   ├── htr-review.component.html
            │   └── htr-review.component.scss
            └── semantic-search/
                ├── semantic-search.component.ts
                ├── semantic-search.component.html
                └── semantic-search.component.scss
```

All new components are Angular standalone components (matching the pattern used elsewhere in the codebase) and import `CommonModule`, `FormsModule`, `RouterModule`, and `PageHeaderComponent`. No existing components were modified beyond the two integration points (routing and sidebar).

### 5.2 Routing

Two new routes are registered under the top-level app frame, both gated by the standard permissions pattern used by the other pages. The routes are children of the main `AppFrameComponent` so they inherit the sidebar and header chrome.

```ts
{ path: 'ml/htr-review', component: HtrReviewComponent, ... }
{ path: 'ml/search',     component: SemanticSearchComponent, ... }
```

### 5.3 Build and deployment

The custom image is built from the fork using the upstream multi-stage `Dockerfile`, which compiles the Angular app with pnpm + ng and bakes it into the Python runtime image. Compose is configured to build locally instead of pulling from GHCR:

```yaml
  webserver:
    image: paperless-ngx-ml:latest
    build:
      context: ../paperless-ngx-fork
```

Build and deploy commands:
```powershell
cd paperless
docker compose build webserver
docker compose up -d
```

**One-time setup gotcha on Windows.** Git's default `core.autocrlf=true` converts shell scripts, Python scripts, and s6-overlay service definition files from LF to CRLF on checkout. Several of these files (the `deduplicate.py` shebang and every s6 `type`/`up`/`down` file) fail at build or container-start time with cryptic errors (`env: 'python3\r': No such file or directory`, `s6-rc-compile: fatal: invalid ... type`). Fix: after cloning the fork, convert all text files under `docker/rootfs/` and all `.py`/`.sh`/`Dockerfile` files back to LF. A PowerShell one-liner is in the project's build notes.

### 5.4 Configuration

- **`PAPERLESS_SECRET_KEY`** must be set in `docker-compose.env`. The upstream prebuilt image ships with a dev default; source builds require the key explicitly. Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(64))"` and commit an example file (never the real key) to the repo.
- **ML serving base URL.** The UI calls `/ml-api/predict/*` as a relative path. Until this path is wired to the FastAPI serving layer, requests 404 and the UI falls back to mock data. See Section 6.

## 6. Remaining integration work

The UI is complete but two pieces of plumbing are still required before it serves live predictions:

1. **ML serving proxy.** The browser cannot call the FastAPI serving directly without CORS, and the FastAPI app does not currently install `CORSMiddleware`. Three options, in order of simplicity:
   - **Nginx sidecar.** Add a small nginx container that listens on 8000, proxies `/ml-api/*` to `fastapi_server:8000`, and passes everything else to the Paperless webserver. Requires a shared Docker network but zero code changes to either service.
   - **Django view proxy.** Add a view in the Paperless backend that forwards `/ml-api/*` requests to the configured `FASTAPI_URL`. Gives you native Paperless auth enforcement on ML calls.
   - **Direct + CORS.** Add `CORSMiddleware` to the FastAPI app with the Paperless origin allowed, publish the serving port, and change the UI's `mlApiBase` to an absolute URL. Simplest for local dev, not recommended for production.

2. **Data backend views.** The three endpoints in the table in Section 4.2 need to be implemented as Django views in the Paperless webserver. Each should validate the payload against the schemas in the data design document, write the row to Postgres, and emit the corresponding Redpanda event. Once these are live, remove the mock fallback in the components (or keep it behind a build flag for offline demos).

3. **Document ID bridging.** Search results return UUIDs from the vector index, but Paperless documents are keyed by integer IDs. Clicking a search result currently navigates to `/documents/{uuid}` and 404s. The data pipeline needs to either store a `ml_document_uuid` custom field on each Paperless document (then the UI resolves uuid → integer before navigation) or use Paperless's integer ID as the canonical key throughout the pipeline. This is a joint data/backend decision.

## 7. Testing and evaluation

**Current state.** Both pages render correctly with mock data and gracefully handle 404s from unreachable endpoints. Manual verification covers:
- Route navigation from the sidebar
- HTR correction flow (edit, save, saved-state toggle)
- Opt-in switch affecting the correction payload
- Search flow with mode toggle
- Click tracking and thumbs feedback
- Mock banner visibility when backends are unreachable

**What's not tested yet.**
- End-to-end with live serving (requires Section 6 item 1)
- End-to-end with live data backend (requires Section 6 item 2)
- Permissions integration (the new routes currently have no `canActivate` guard; this should be added once the team decides on the right permission scope — probably `View` on `Document`)
- Internationalization (strings are marked with `i18n` attributes but only English is included)

## 8. Summary

The UI layer of the Paperless-ngx ML platform is in place: two new pages, six new component files, two small edits to existing files, and a custom Docker image that builds cleanly from the fork. Both pages talk to the exact contracts owned by the serving and data teams, so backend work can proceed without further UI changes. The remaining integration tasks are infrastructure-level (proxy + data backend views) rather than frontend, and each page degrades gracefully to mock data until its backend is wired up.
