-- Add archived_at column to htr_corrections.
--
-- When the airflow `archive_corrections` DAG successfully uploads a
-- correction's denormalized JSON to MinIO, it sets archived_at = NOW().
-- This column is the water-mark the DAG polls each run to find new
-- corrections still needing archival (WHERE archived_at IS NULL).
--
-- Once archived, a correction's immutable record lives at:
--   s3://paperless-datalake/user_corrections/date=YYYY-MM-DD/<id>.json
--
-- Training (batch_htr.py) reads from that MinIO path, not from this
-- Postgres table. The Postgres table remains the source of truth for
-- the review UI (queue of corrections per document), and the archive
-- is the source of truth for training.

ALTER TABLE htr_corrections
    ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;

-- Partial index: only un-archived rows, since the DAG's polling query
-- is always WHERE archived_at IS NULL. A full index would grow
-- unboundedly; this one stays small regardless of total corrections.
CREATE INDEX IF NOT EXISTS idx_corrections_archived_at_null
    ON htr_corrections (corrected_at)
    WHERE archived_at IS NULL;
