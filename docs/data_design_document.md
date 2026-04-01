# Data Design Document
## Paperless-ngx ML Platform — Data Team

**Team:** Dongting Gao (Training), Yikai Sun (Serving), Elnath Zhao (Data)  
**Version:** 1.1

---

## 1. Overview

This document describes the data architecture for the Paperless-ngx ML platform. The system extends a document management service with two ML features: handwriting text recognition (HTR) at upload time, and semantic search at query time. The data platform must support real-time inference on incoming documents and queries, and periodic retraining from accumulated user feedback.

---

## 2. Data Repositories

### 2.1 PostgreSQL — Application State

**Purpose:** Source of truth for document metadata, user feedback, and training event logs.

**Written by:** Paperless-ngx application on every upload, correction, and search interaction.  
**Updated:** Continuously during service operation.  
**Versioning:** Not versioned directly — serves as the transactional record from which Iceberg snapshots are derived.

#### Tables

**`documents`**
```
id              UUID        PRIMARY KEY
filename        TEXT        NOT NULL
uploaded_at     TIMESTAMPTZ DEFAULT NOW()
source          VARCHAR(16) CHECK (source IN ('user_upload', 'synthetic', 'test'))
page_count      INTEGER
tesseract_text  TEXT
htr_text        TEXT
merged_text     TEXT
is_test_doc     BOOLEAN     DEFAULT FALSE
deleted_at      TIMESTAMPTZ
```
Stores one row per uploaded document. `merged_text` is the concatenation of Tesseract OCR output and HTR output, and is the text encoded into the vector index. `source` distinguishes real uploads from synthetic traffic generator uploads and test documents.

**`document_pages`**
```
id              UUID        PRIMARY KEY
document_id     UUID        REFERENCES documents(id)
page_number     INTEGER
image_s3_url    TEXT        NOT NULL
tesseract_text  TEXT
htr_text        TEXT
htr_confidence  FLOAT
htr_flagged     BOOLEAN     DEFAULT FALSE
created_at      TIMESTAMPTZ DEFAULT NOW()
```
One row per page. `image_s3_url` points to the page image in MinIO. `htr_flagged` is set when confidence falls below the threshold, triggering the user review UI.

**`handwritten_regions`**
```
id              UUID        PRIMARY KEY
page_id         UUID        REFERENCES document_pages(id)
crop_s3_url     TEXT        NOT NULL
htr_output      TEXT
htr_confidence  FLOAT
created_at      TIMESTAMPTZ DEFAULT NOW()
```
One row per detected handwritten region within a page. `crop_s3_url` points to the cropped region image in MinIO. This is the granular unit fed to the HTR model.

**`htr_corrections`**
```
id              UUID        PRIMARY KEY
region_id       UUID        REFERENCES handwritten_regions(id)
user_id         UUID
original_text   TEXT
corrected_text  TEXT        NOT NULL
corrected_at    TIMESTAMPTZ DEFAULT NOW()
opted_in        BOOLEAN     DEFAULT TRUE
excluded_from_training BOOLEAN DEFAULT FALSE
```
Written when a user edits an HTR transcription. `opted_in` respects the user's data contribution preference. `excluded_from_training` is set by the batch pipeline for duplicates or policy-excluded records.

**`query_sessions`**
```
id              UUID        PRIMARY KEY
query_text      TEXT        NOT NULL
queried_at      TIMESTAMPTZ DEFAULT NOW()
user_id         UUID
is_test_account BOOLEAN     DEFAULT FALSE
result_doc_ids  UUID[]
```
One row per search query. `result_doc_ids` captures the ranked list of documents returned, needed to identify shown-but-not-clicked negatives.

**`search_feedback`**
```
id              UUID        PRIMARY KEY
session_id      UUID        REFERENCES query_sessions(id)
document_id     UUID        REFERENCES documents(id)
feedback_type   VARCHAR(16) CHECK (feedback_type IN ('click', 'thumbs_up', 'thumbs_down'))
created_at      TIMESTAMPTZ DEFAULT NOW()
```
Written on user interaction with search results. `click` is an implicit positive signal; `thumbs_up` / `thumbs_down` are explicit. Only explicit feedback is used for retrieval retraining (see Section 5).

**`dataset_registry`**
```
id              UUID        PRIMARY KEY DEFAULT gen_random_uuid()
name            TEXT        NOT NULL
version         TEXT
download_url    TEXT
downloaded_at   TIMESTAMPTZ DEFAULT NOW()
record_count    INTEGER
s3_path         TEXT
license         TEXT
```
Records the provenance of every external dataset ingested into the platform, enabling lineage tracking from training data back to source.

---

### 2.2 MinIO — Object Storage

**Purpose:** Durable storage for raw images, cropped regions, ingested external datasets, and Iceberg table data.  
**Written by:** Paperless-ngx upload handler (page images), HTR preprocessing service (crops), ingestion pipeline (external datasets), Airflow DAGs (Iceberg Parquet files).  
**Versioning:** Iceberg manages versioning within `paperless-datalake`. Raw image buckets are append-only; files are never overwritten.

#### Buckets

**`paperless-images`**
```
documents/{document_id}/page_{n}.png         Raw full-page scan images
documents/{document_id}/regions/{region_id}.png  Cropped handwritten region images
```
All uploaded document images. Written once at upload time, never modified. Soft-deleted documents retain their images until a cleanup job runs.

**`paperless-datalake`**
```
warehouse/
  iam_dataset/
    train/          Parquet shards — IAM line images (PNG binary) + ground-truth text
    validation/     Parquet shards — validation split
    test/           Parquet shards — test split
    train_augmented/      Augmented training images (3x synthetic expansion)
    validation_augmented/ Augmented validation images
    test_augmented/       Augmented test images
    metadata.json   Dataset provenance and schema info
  squad_dataset/
    train/          Parquet shards — (triplet_id, query, passage, label) retrieval triplets
    validation/     Parquet shards — validation split
    metadata.json   Dataset provenance and schema info
  htr_training/
    data/           Parquet files — versioned HTR fine-tuning pairs
    metadata/       Iceberg metadata and manifests
  retrieval_training/
    data/           Parquet files — versioned retrieval triplets
    metadata/       Iceberg metadata and manifests
```

**`paperless-staging`**
```
temp/             Temporary files during ETL processing
```
Temporary staging area for ETL jobs. Cleared after successful ingestion.

#### IAM Dataset Parquet Schema
```
image_id        STRING    Unique identifier (e.g., "train_000042")
image_png       BINARY    Raw PNG bytes of the handwriting line image
transcription   STRING    Ground-truth text transcription
split           STRING    'train' | 'validation' | 'test' | 'train_augmented' | ...
```

#### SQuAD Dataset Parquet Schema
```
triplet_id      STRING    MD5-derived unique ID for deduplication
query           STRING    Question text from SQuAD 2.0
passage         STRING    Context paragraph
label           INT32     1 = relevant (answerable), 0 = irrelevant (unanswerable or mismatched)
split           STRING    'train' | 'validation'
```

---

### 2.3 Qdrant — Vector Index

**Purpose:** Stores dense embeddings for all document chunks, enabling approximate nearest-neighbor search at query time.  
**Written by:** Document indexing service, triggered after each upload's `merged_text` is finalized.  
**Updated:** On new upload, on HTR correction (re-encode updated merged_text), on document deletion (soft-delete payload flag).  
**Versioning:** Not snapshot-versioned. Qdrant is the online serving index only — the training data is versioned separately in Iceberg.

#### Collections

**`document_chunks`**
```
id:         UUID  (chunk-level, derived from document_id + chunk_index)
vector:     float[768]   Dense embedding from bi-encoder
payload:
  document_id:   UUID
  chunk_index:   INTEGER
  chunk_text:    TEXT
  uploaded_at:   TIMESTAMPTZ
  is_deleted:    BOOLEAN
```
Each document is split into fixed-size overlapping text chunks (e.g., 256 tokens, 32-token stride). Each chunk is encoded independently and stored as one point. At query time, the query vector is matched against all non-deleted chunk vectors using HNSW approximate nearest-neighbor search; top-k chunks are returned and deduplicated to document level.

---

### 2.4 Redpanda — Event Stream

**Purpose:** Decouples the upload API from downstream processing. Carries upload events, correction events, and search feedback events so that inference services and batch pipelines can consume them independently without adding load to PostgreSQL.  
**Written by:** API service on every upload, correction submission, and search feedback action.  
**Versioning:** Not versioned directly. Redpanda retains events by offset; the batch Airflow DAG tracks a watermark (Kafka offset checkpoint) to consume incrementally.

#### Topics

| Topic | Event schema | Consumers |
|---|---|---|
| `paperless.uploads` | `{document_id, page_count, uploaded_at, source}` | HTR preprocessing service, document indexing service |
| `paperless.corrections` | `{region_id, document_id, corrected_text, corrected_at, opted_in}` | Re-indexing service, Airflow HTR DAG |
| `paperless.queries` | `{session_id, query_text, queried_at, result_doc_ids}` | Airflow retrieval DAG |
| `paperless.feedback` | `{session_id, document_id, feedback_type, created_at}` | Airflow retrieval DAG |

---

### 2.5 Iceberg Tables — Versioned Training Data

**Purpose:** Durable, versioned training and evaluation datasets consumed by the training team. Every write creates a new snapshot with a stable `snapshot_id` that can be recorded alongside model artifacts for full reproducibility.  
**Written by:** Airflow DAGs (batch pipeline).  
**Catalog backend:** PostgreSQL (`iceberg_catalog` database).  
**Storage backend:** MinIO (`paperless-datalake/warehouse/`).

#### `htr_training.fine_tune_pairs`
```
region_id        STRING    Source region UUID
crop_s3_url      STRING    Path to cropped image in MinIO
corrected_text   STRING    Ground-truth transcription (user-corrected)
source           STRING    'iam' | 'user_correction'
split            STRING    'train' | 'val' | 'test'
correction_date  DATE      Used for time-based splitting
writer_id        STRING    Anonymized user ID (for deduplication)
excluded         BOOLEAN   Policy exclusion flag
snapshot_id      BIGINT    Iceberg snapshot this row was written in
```

#### `retrieval_training.triplets`
```
session_id       STRING    Source query session UUID
query_text       STRING    User search query
pos_doc_id       STRING    Clicked or thumbs-up document
neg_doc_id       STRING    Shown-but-not-clicked or thumbs-down document
pos_chunk_text   STRING    Relevant chunk text at query time
neg_chunk_text   STRING    Non-relevant chunk text
feedback_type    STRING    'click' | 'thumbs_up' | 'thumbs_down'
query_date       DATE      Used for time-based splitting
split            STRING    'train' | 'val' | 'test'
snapshot_id      BIGINT    Iceberg snapshot this row was written in
```

---

## 3. Data Flow Diagrams

### 3.1 Upload-time (real-time inference path)

```
User uploads document
        │
        ▼
┌─────────────────┐
│   API service   │──── writes ──▶ documents (PostgreSQL)
└────────┬────────┘
         │ publishes
         ▼
┌─────────────────────────┐
│  paperless.uploads      │  (Redpanda topic)
└────────┬────────────────┘
         │ consumed by
         ▼
┌─────────────────────────────────────────┐
│  HTR preprocessing service              │
│  1. Convert PDF → page images → MinIO  │
│  2. Detect handwritten regions          │
│  3. Crop regions → MinIO               │
│  4. Write document_pages + regions      │
│     (PostgreSQL)                        │
│  5. Call HTR model → transcription      │
│     + confidence score                  │
│  6. Flag low-confidence regions         │
│  7. Merge HTR + Tesseract → merged_text │
└────────────────┬────────────────────────┘
                 │ triggers
                 ▼
┌─────────────────────────────────────────┐
│  Document indexing service              │
│  1. Chunk merged_text                   │
│  2. Encode each chunk → vector          │
│  3. Upsert into Qdrant                  │
└─────────────────────────────────────────┘
```

### 3.2 Query-time (real-time inference path)

```
User submits search query
        │
        ▼
┌─────────────────────────────────────────┐
│  Search service                         │
│  1. Encode query text → query vector    │
│  2. ANN search in Qdrant (top-k chunks) │
│  3. Deduplicate to document level       │
│  4. If max similarity < threshold:      │
│     fall back to Paperless keyword      │
│  5. Return ranked results               │
└────────┬────────────────────────────────┘
         │ publishes
         ▼
┌──────────────────────────────┐
│  paperless.queries (Redpanda)│
└──────────────────────────────┘

User clicks result / gives feedback
        │
        ▼
┌──────────────────────────────────┐
│  API service                     │
│  Writes search_feedback (PG)     │
│  Publishes paperless.feedback    │
└──────────────────────────────────┘
```

### 3.3 Batch pipeline (training data production)

```
Airflow — runs daily
        │
        ├─── DAG: htr_training_data
        │         1. Read htr_corrections (PostgreSQL)
        │         2. Filter: opted_in=true, active correction,
        │            no duplicate regions
        │         3. Time-based split (train/val/test)
        │         4. Write → Iceberg htr_training.fine_tune_pairs
        │            (new snapshot)
        │
        └─── DAG: retrieval_training_data
                  1. Read query_sessions + search_feedback (PostgreSQL)
                  2. Filter: explicit feedback only, no test accounts,
                     no empty sessions
                  3. Construct (query, pos_doc, neg_doc) triplets
                  4. Snapshot document chunk text at query time
                     (from PostgreSQL, not current Qdrant state)
                  5. Time-based split
                  6. Write → Iceberg retrieval_training.triplets
                     (new snapshot)
```

### 3.4 Ingestion pipeline (external data)

```
External datasets
        │
        ├─── IAM Handwriting (HuggingFace: Teklia/IAM-line)
        │         1. Download via HuggingFace datasets library
        │         2. Extract line images as PNG binary + transcription text
        │         3. Write Parquet shards → MinIO paperless-datalake/warehouse/iam_dataset/{split}/
        │         4. Synthetic expansion: rotation, noise, contrast, blur (3x)
        │            → write to {split}_augmented/
        │
        └─── SQuAD 2.0 (rajpurkar.github.io)
                  1. Download train-v2.0.json and dev-v2.0.json directly
                  2. Flatten articles → QA pairs
                  3. Format as retrieval triplets with hard negative mining
                  4. Write Parquet shards → MinIO paperless-datalake/warehouse/squad_dataset/{split}/
```

---

## 4. Versioning and Lineage

Every Iceberg table write produces a new snapshot. Before each training run, the training service records:

```json
{
  "model": "htr_v2",
  "iceberg_table": "htr_training.fine_tune_pairs",
  "snapshot_id": 8473920174,
  "trained_at": "2025-04-01T14:00:00Z"
}
```

This `snapshot_id` is stored with the model artifact, enabling exact reproduction of the training dataset for any past model version by pointing PyIceberg at that specific snapshot.

For the IAM and SQuAD external datasets, lineage is recorded in two places:

1. **`metadata.json`** uploaded alongside the Parquet shards in MinIO, containing the source URL, download timestamp, record counts, and schema.
2. **`dataset_registry`** PostgreSQL table, recording the provenance of every external dataset:

```
dataset_registry
  name          TEXT    'iam' | 'squad_2.0'
  version       TEXT
  download_url  TEXT
  downloaded_at TIMESTAMPTZ
  record_count  INTEGER
  s3_path       TEXT
  license       TEXT
```

---

## 5. Candidate Selection and Leakage Prevention

### HTR training pairs
- **Include:** Only `htr_corrections` rows where `corrected_text != ''` and `opted_in = true` (user actively typed a correction, not just dismissed the flag)
- **Exclude:** Test/synthetic documents (`source != 'user_upload'`), duplicate region scans (same `writer_id` + identical `corrected_text` within 7 days), documents where `deleted_at` is set
- **Split:** Time-based — train on corrections older than 14 days, validate on the most recent 14 days. This prevents the model from training on styles it will be evaluated on.
- **Leakage prevention:** Training inputs are always raw crop images, never the model's own prior HTR output. Preprocessing (normalization, augmentation parameters) is fitted on the training split only.

### Retrieval training triplets
- **Include:** Only sessions with explicit feedback (`thumbs_up` or `thumbs_down`) or confirmed clicks; exclude sessions from test accounts (`is_test_account = true`)
- **Exclude:** Sessions where the user issued the same query within 5 minutes (likely a reformulation, not independent signal), documents with `deleted_at` set at training time
- **Split:** Time-based — train on sessions older than 7 days, validate on the most recent 7 days
- **Leakage prevention:** `pos_chunk_text` and `neg_chunk_text` are snapshotted from PostgreSQL at `query_date`, not from current document state. This ensures the training example reflects what the model would have seen at inference time, not a later-edited version of the document.

---

## 6. Services and Write Summary

| Service | Writes to | When |
|---|---|---|
| API / Paperless-ngx | `documents`, `query_sessions`, `search_feedback` (PG); `paperless.*` topics (Redpanda) | On every upload, query, feedback |
| HTR preprocessing | `document_pages`, `handwritten_regions` (PG); page + crop images (MinIO) | On `paperless.uploads` event |
| Document indexing | `document_chunks` (Qdrant) | After HTR preprocessing completes |
| Re-indexing service | `document_chunks` (Qdrant) | On `paperless.corrections` event |
| Ingestion pipeline | IAM + SQuAD Parquet shards (MinIO); `dataset_registry` (PG) | One-shot on initial setup; re-run for updates |
| Augmentation pipeline | Augmented IAM Parquet shards (MinIO) | After IAM ingestion; re-run for updates |
| Airflow HTR DAG | `htr_training.fine_tune_pairs` (Iceberg) | Daily |
| Airflow retrieval DAG | `retrieval_training.triplets` (Iceberg) | Daily |
