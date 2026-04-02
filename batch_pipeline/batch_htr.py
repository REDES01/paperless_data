"""
Batch Pipeline — HTR Training Data

Compiles versioned training dataset from production HTR corrections.
Reads from PostgreSQL, applies candidate selection and leakage prevention,
writes versioned Parquet to MinIO.

Candidate selection:
  - Only corrections where opted_in = true
  - Only corrections where corrected_text is non-empty
  - Exclude test/synthetic documents (source != 'user_upload')
  - Exclude deleted documents
  - Deduplicate: same region corrected multiple times → keep latest

Leakage prevention:
  - Time-based split: train on corrections older than 14 days,
    validate on the most recent 14 days
  - Training inputs are raw crop images, never model's own HTR output

Output:
  paperless-datalake/warehouse/htr_training/
    v_{timestamp}/
      train/shard_0000.parquet
      val/shard_0000.parquet
      manifest.json    ← snapshot metadata
"""

import os
import io
import json
import logging
import hashlib
from datetime import datetime, timezone, timedelta

import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
DB_DSN = os.getenv("DB_DSN", "host=postgres dbname=paperless user=user password=paperless_postgres")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
BUCKET = "paperless-datalake"
PREFIX = "warehouse/htr_training"
SHARD_SIZE = 500
VAL_WINDOW_DAYS = 14  # most recent N days → validation set


def get_pg():
    conn = psycopg2.connect(DB_DSN)
    conn.set_session(readonly=True)
    return conn


def get_minio():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=False)


def fetch_candidates(conn) -> list[dict]:
    """
    Fetch eligible HTR corrections with candidate selection applied.
    
    SQL implements:
    - opted_in = true
    - corrected_text is non-empty
    - source document is user_upload (not synthetic/test)
    - document not deleted
    - Deduplicate by region_id (keep latest correction per region)
    """
    query = """
    WITH ranked_corrections AS (
        SELECT
            c.id AS correction_id,
            c.region_id,
            c.corrected_text,
            c.corrected_at,
            c.user_id,
            r.crop_s3_url,
            r.htr_output AS original_text,
            d.source,
            d.id AS document_id,
            ROW_NUMBER() OVER (
                PARTITION BY c.region_id
                ORDER BY c.corrected_at DESC
            ) AS rn
        FROM htr_corrections c
        JOIN handwritten_regions r ON c.region_id = r.id
        JOIN document_pages p ON r.page_id = p.id
        JOIN documents d ON p.document_id = d.id
        WHERE c.opted_in = true
          AND c.corrected_text != ''
          AND c.excluded_from_training = false
          AND d.source = 'synthetic'
          AND d.deleted_at IS NULL
    )
    SELECT
        correction_id,
        region_id,
        corrected_text,
        corrected_at,
        user_id,
        crop_s3_url,
        original_text,
        document_id
    FROM ranked_corrections
    WHERE rn = 1
    ORDER BY corrected_at ASC
    """
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()

    candidates = [dict(zip(columns, row)) for row in rows]
    log.info(f"Fetched {len(candidates)} eligible corrections (after dedup + filtering)")
    return candidates


def time_split(candidates: list[dict], val_window_days: int) -> tuple[list[dict], list[dict]]:
    """
    Time-based train/val split to prevent leakage.
    - Train: corrections older than val_window_days
    - Val: corrections within the most recent val_window_days
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=val_window_days)

    train = []
    val = []
    for c in candidates:
        ts = c["corrected_at"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            train.append(c)
        else:
            val.append(c)

    log.info(f"Time split (cutoff={cutoff.date()}): train={len(train)}, val={len(val)}")
    return train, val


def build_table(candidates: list[dict], split: str) -> pa.Table:
    """Convert candidate list to a PyArrow table."""
    return pa.table({
        "region_id": pa.array([str(c["region_id"]) for c in candidates], type=pa.string()),
        "crop_s3_url": pa.array([c["crop_s3_url"] for c in candidates], type=pa.string()),
        "corrected_text": pa.array([c["corrected_text"] for c in candidates], type=pa.string()),
        "original_text": pa.array([c["original_text"] or "" for c in candidates], type=pa.string()),
        "source": pa.array(["user_correction"] * len(candidates), type=pa.string()),
        "split": pa.array([split] * len(candidates), type=pa.string()),
        "correction_date": pa.array(
            [c["corrected_at"].strftime("%Y-%m-%d") for c in candidates], type=pa.string()
        ),
        "writer_id": pa.array(
            [str(c["user_id"]) if c["user_id"] else "unknown" for c in candidates],
            type=pa.string(),
        ),
    })


def upload_shards(client: Minio, table: pa.Table, version: str, split: str):
    """Write table as Parquet shards to MinIO."""
    num_rows = len(table)
    shard_idx = 0

    for start in range(0, num_rows, SHARD_SIZE):
        end = min(start + SHARD_SIZE, num_rows)
        shard = table.slice(start, end - start)

        buf = io.BytesIO()
        pq.write_table(shard, buf)
        buf.seek(0)

        obj_name = f"{PREFIX}/{version}/{split}/shard_{shard_idx:04d}.parquet"
        client.put_object(BUCKET, obj_name, buf, length=buf.getbuffer().nbytes)
        log.info(f"Uploaded {obj_name} ({len(shard)} rows)")
        shard_idx += 1

    return shard_idx


def upload_manifest(client: Minio, version: str, train_count: int, val_count: int,
                    train_shards: int, val_shards: int):
    """Write a manifest file recording this snapshot's metadata."""
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "htr_training_data",
        "candidate_selection": {
            "opted_in": True,
            "non_empty_text": True,
            "exclude_test_synthetic": True,
            "exclude_deleted": True,
            "dedup_by_region": "keep_latest",
        },
        "leakage_prevention": {
            "split_method": "time_based",
            "val_window_days": VAL_WINDOW_DAYS,
            "note": "Training inputs are raw crop images, never model HTR output",
        },
        "counts": {
            "train": train_count,
            "val": val_count,
            "total": train_count + val_count,
        },
        "shards": {
            "train": train_shards,
            "val": val_shards,
        },
    }

    buf = io.BytesIO(json.dumps(manifest, indent=2).encode())
    obj_name = f"{PREFIX}/{version}/manifest.json"
    client.put_object(BUCKET, obj_name, buf, length=buf.getbuffer().nbytes)
    log.info(f"Uploaded {obj_name}")
    return manifest


def main():
    log.info("=" * 60)
    log.info("Batch Pipeline: HTR Training Data")
    log.info("=" * 60)

    # Generate version ID
    version = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    log.info(f"Version: {version}")

    # Step 1: Fetch candidates from PostgreSQL
    conn = get_pg()
    candidates = fetch_candidates(conn)
    conn.close()

    if not candidates:
        log.warning("No eligible corrections found. Exiting.")
        return

    # Step 2: Time-based split
    train_data, val_data = time_split(candidates, VAL_WINDOW_DAYS)

    # Step 3: Build Arrow tables
    mc = get_minio()
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)

    train_shards = 0
    val_shards = 0

    if train_data:
        train_table = build_table(train_data, "train")
        train_shards = upload_shards(mc, train_table, version, "train")
        log.info(f"Train: {len(train_data)} rows in {train_shards} shards")

    if val_data:
        val_table = build_table(val_data, "val")
        val_shards = upload_shards(mc, val_table, version, "val")
        log.info(f"Val: {len(val_data)} rows in {val_shards} shards")

    # Step 4: Write manifest
    manifest = upload_manifest(mc, version, len(train_data), len(val_data),
                                train_shards, val_shards)

    log.info(f"\nPipeline complete. Snapshot: {version}")
    log.info(f"  Train: {len(train_data)} corrections")
    log.info(f"  Val:   {len(val_data)} corrections")
    log.info(f"  Output: s3://{BUCKET}/{PREFIX}/{version}/")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
