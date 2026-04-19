-- ──────────────────────────────────────────────
--  Paperless-ngx ML Platform — Application tables
--  Database: paperless
-- ──────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Documents ────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT        NOT NULL,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
    source          VARCHAR(16) NOT NULL DEFAULT 'user_upload'
                    CHECK (source IN ('user_upload', 'synthetic', 'test')),
    page_count      INTEGER,
    tesseract_text  TEXT,
    htr_text        TEXT,
    merged_text     TEXT,
    is_test_doc     BOOLEAN     DEFAULT FALSE,
    deleted_at      TIMESTAMPTZ,
    -- External identifier from the upstream Paperless instance. Required by
    -- the HTR consumer when re-processing documents, which uses
    -- `ON CONFLICT (paperless_doc_id)` to upsert. UNIQUE also creates the
    -- non-partial index that ON CONFLICT needs.
    paperless_doc_id INTEGER UNIQUE
);

-- Idempotent upgrade path for stacks whose `documents` table already exists
-- (the CREATE TABLE IF NOT EXISTS above is a no-op in that case, so the
-- column + UNIQUE constraint would otherwise never land). Safe to re-run
-- on fresh databases too — both statements short-circuit when the column
-- and index already exist.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS paperless_doc_id INTEGER;
CREATE UNIQUE INDEX IF NOT EXISTS documents_paperless_doc_id_key
    ON documents(paperless_doc_id);

-- ── Document pages ───────────────────────────
CREATE TABLE IF NOT EXISTS document_pages (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID        NOT NULL REFERENCES documents(id),
    page_number     INTEGER     NOT NULL,
    image_s3_url    TEXT        NOT NULL,
    tesseract_text  TEXT,
    htr_text        TEXT,
    htr_confidence  FLOAT,
    htr_flagged     BOOLEAN     DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Handwritten regions ──────────────────────
CREATE TABLE IF NOT EXISTS handwritten_regions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id         UUID        NOT NULL REFERENCES document_pages(id),
    crop_s3_url     TEXT        NOT NULL,
    htr_output      TEXT,
    htr_confidence  FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── HTR corrections (user feedback) ─────────
CREATE TABLE IF NOT EXISTS htr_corrections (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id               UUID        NOT NULL REFERENCES handwritten_regions(id),
    user_id                 UUID,
    original_text           TEXT,
    corrected_text          TEXT        NOT NULL,
    corrected_at            TIMESTAMPTZ DEFAULT NOW(),
    opted_in                BOOLEAN     DEFAULT TRUE,
    excluded_from_training  BOOLEAN     DEFAULT FALSE
);

-- ── Query sessions ───────────────────────────
CREATE TABLE IF NOT EXISTS query_sessions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT        NOT NULL,
    queried_at      TIMESTAMPTZ DEFAULT NOW(),
    user_id         UUID,
    is_test_account BOOLEAN     DEFAULT FALSE,
    result_doc_ids  UUID[]
);

-- ── Search feedback ──────────────────────────
CREATE TABLE IF NOT EXISTS search_feedback (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL REFERENCES query_sessions(id),
    document_id     UUID        NOT NULL REFERENCES documents(id),
    feedback_type   VARCHAR(16) NOT NULL
                    CHECK (feedback_type IN ('click', 'thumbs_up', 'thumbs_down')),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Dataset registry ─────────────────────────
CREATE TABLE IF NOT EXISTS dataset_registry (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT        NOT NULL UNIQUE,
    version         TEXT,
    download_url    TEXT,
    downloaded_at   TIMESTAMPTZ,
    record_count    INTEGER,
    s3_path         TEXT,
    license         TEXT,
    notes           TEXT
);

-- ── Indexes ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_pages_document_id        ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_regions_page_id          ON handwritten_regions(page_id);
CREATE INDEX IF NOT EXISTS idx_corrections_region_id    ON htr_corrections(region_id);
CREATE INDEX IF NOT EXISTS idx_corrections_corrected_at ON htr_corrections(corrected_at);
CREATE INDEX IF NOT EXISTS idx_feedback_session_id      ON search_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at      ON search_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_queried_at      ON query_sessions(queried_at);
CREATE INDEX IF NOT EXISTS idx_documents_source         ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at    ON documents(uploaded_at);
