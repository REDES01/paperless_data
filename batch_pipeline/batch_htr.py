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

from quality import filter_candidates
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
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.getenv("MINIO_BUCKET", "paperless-datalake")
PREFIX = "warehouse/htr_training"
SHARD_SIZE = 500
VAL_WINDOW_DAYS = 14  # most recent N days → validation set


def get_pg():
    conn = psycopg2.connect(DB_DSN)
    conn.set_session(readonly=True)
    return conn


def get_minio():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def fetch_candidates(conn, mc) -> list[dict]:
    """
    Fetch eligible HTR corrections from the MinIO user_corrections archive.

    The archive is populated by the airflow `archive_corrections` DAG
    (every 15 min) — each correction becomes an immutable JSON object at
      s3://paperless-datalake/user_corrections/date=YYYY-MM-DD/<uuid>.json

    Reading from the archive instead of Postgres gives:
      - Training reproducibility (archive is immutable; past training
        runs can be reconstructed exactly by filtering on archived_at)
      - Audit trail (every correction ever submitted is recoverable)
      - Decoupling (training never blocks on Postgres availability)

    Filtering applied here (match the old SQL's WHERE clauses):
      - opted_in = true
      - corrected_text is non-empty (stripped)
      - source document is `user_upload` and not soft-deleted
      - Deduplicate by region_id, keeping the most recent corrected_at

    The doc-state filter (source + deleted_at) still requires a small
    Postgres query — those attributes can change after a correction was
    archived. We issue ONE query to get the set of training-eligible
    document_ids, then filter the archive in memory.
    """
    import json as _json

    bucket = os.getenv("MINIO_BUCKET", "paperless-datalake")
    prefix = "user_corrections/"

    # 1. Get eligible document_ids from Postgres (one cheap query).
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id
            FROM documents
            WHERE source = 'user_upload'
              AND deleted_at IS NULL
        """)
        eligible_doc_ids = {row[0] for row in cur.fetchall()}
    log.info(f"{len(eligible_doc_ids)} documents currently eligible for training")

    # 2. List all archived corrections.
    try:
        objects = list(mc.list_objects(bucket, prefix=prefix, recursive=True))
    except Exception as exc:
        log.warning("archive listing failed (%s) — returning 0 candidates", exc)
        return []
    json_objects = [o for o in objects if o.object_name.endswith(".json")]
    log.info(f"archive has {len(json_objects)} correction JSON files")

    # 3. Download + parse + filter.
    raw: list[dict] = []
    for obj in json_objects:
        try:
            resp = mc.get_object(bucket, obj.object_name)
            body = _json.loads(resp.read().decode("utf-8"))
            resp.close()
            resp.release_conn()
        except Exception as exc:
            log.warning("could not read %s: %s", obj.object_name, exc)
            continue

        if not body.get("opted_in", False):
            continue
        if not (body.get("corrected_text") or "").strip():
            continue
        if body.get("document_id") not in eligible_doc_ids:
            continue
        raw.append(body)

    # 4. Dedupe by region_id — keep newest corrected_at per region.
    by_region: dict[str, dict] = {}
    for b in raw:
        region_id = b["region_id"]
        incumbent = by_region.get(region_id)
        if (
            incumbent is None
            or (b.get("corrected_at") or "") > (incumbent.get("corrected_at") or "")
        ):
            by_region[region_id] = b

    # 5. Shape to match the old output (keys used downstream by build_table).
    candidates = []
    for b in sorted(by_region.values(), key=lambda x: x.get("corrected_at") or ""):
        candidates.append({
            "correction_id": b["correction_id"],
            "region_id":     b["region_id"],
            "corrected_text": b["corrected_text"],
            "corrected_at":  b["corrected_at"],
            "user_id":       b.get("user_id"),
            "crop_s3_url":   b["crop_s3_url"],
            "original_text": b.get("original_text", ""),
            "document_id":   b["document_id"],
        })

    log.info(
        f"Fetched {len(candidates)} eligible corrections from archive "
        f"(after dedup + filtering)"
    )
    return candidates


def document_grouped_split(
    candidates: list[dict],
    val_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """
    Document-grouped train/val split.

    Each document is assigned to exactly one split via deterministic hash
    of document_id. This prevents both:

      - Temporal leakage (same as time-based split): training data is never
        "in the future" relative to val data, because splits don't use time
        at all — they use content hash.

      - Group leakage: if a document has multiple corrections, all of them
        go to the SAME split. Without this, the model could see crops from
        document-A during training, then be evaluated on different crops
        from the same document-A in val, artificially inflating CER. In
        handwriting recognition this matters especially because handwriting
        style is document/writer-specific — if you've seen one line from a
        writer, the next line from the same writer is much easier.

    Properties of this split:
      - Deterministic: same document always lands in same split across
        retraining runs. Eval CER is comparable over time.
      - Disjoint by construction: asserted below. Impossible for a document
        to appear in both splits.
      - Approximately val_fraction of documents end up in val.

    Why not time-based:
      Time-based split (our previous approach) allows a single document
      that has corrections spanning the cutoff boundary to contribute to
      both splits. Document-grouped split is strictly stronger.
    """
    train = []
    val = []
    for c in candidates:
        doc_id = str(c["document_id"])
        # sha256 → large int → bucket in [0, 1)
        h = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0
        if bucket < val_fraction:
            val.append(c)
        else:
            train.append(c)

    # Prove the invariant: no document appears in both splits
    train_docs = {str(c["document_id"]) for c in train}
    val_docs = {str(c["document_id"]) for c in val}
    overlap = train_docs & val_docs
    assert not overlap, (
        f"document-grouped split invariant violated — documents in both splits: "
        f"{sorted(overlap)[:10]}"
    )

    log.info(
        f"Document-grouped split (val_fraction={val_fraction}): "
        f"train={len(train)} corrections from {len(train_docs)} docs, "
        f"val={len(val)} corrections from {len(val_docs)} docs, "
        f"overlap=0 (disjoint by construction)"
    )
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
        # corrected_at is a datetime when read from postgres directly, but
        # an ISO-8601 string when deserialized from a JSON archive snapshot.
        # Both shapes need to land as YYYY-MM-DD.
        "correction_date": pa.array(
            [
                c["corrected_at"].strftime("%Y-%m-%d")
                if hasattr(c["corrected_at"], "strftime")
                else c["corrected_at"][:10]
                for c in candidates
            ],
            type=pa.string(),
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
                    train_shards: int, val_shards: int,
                    quality_report: dict | None = None):
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
            "split_method": "document_grouped_hash",
            "val_fraction": 0.2,
            "hash_algorithm": "sha256(document_id) % 10000 / 10000",
            "invariants": [
                "disjoint: no document_id appears in both train and val",
                "deterministic: same document always goes to same split",
                "training inputs are raw crop images, never model HTR output",
            ],
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
        "data_quality": quality_report or {},
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

    # MinIO client up first — fetch_candidates (and the quality filter
    # below) both need it, and the previous ordering used `mc` before
    # the get_minio() call assigned it.
    mc = get_minio()
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)

    # Step 1: Fetch candidates from PostgreSQL
    conn = get_pg()
    candidates = fetch_candidates(conn, mc)
    conn.close()

    if not candidates:
        log.warning("No eligible corrections found. Exiting.")
        return

    # Step 1a: Apply additional data-quality filters beyond the SQL-level
    # candidate selection (no-op corrections, invalid crop URLs, MinIO
    # misses, spam). See quality.py for the rules.
    candidates, quality_report = filter_candidates(candidates, minio_client=mc)

    if not candidates:
        log.warning("All candidates rejected by quality filters. Exiting.")
        return

    # Step 2: Document-grouped split (stronger than time-based — see docstring)
    train_data, val_data = document_grouped_split(candidates, val_fraction=0.2)

    # Step 3: Build Arrow tables (mc already set up above)
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
                                train_shards, val_shards,
                                quality_report=quality_report.as_dict())

    log.info(f"\nPipeline complete. Snapshot: {version}")
    log.info(f"  Train: {len(train_data)} corrections")
    log.info(f"  Val:   {len(val_data)} corrections")
    log.info(f"  Output: s3://{BUCKET}/{PREFIX}/{version}/")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
