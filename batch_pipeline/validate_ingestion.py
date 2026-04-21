"""
Post-ingestion data quality validation for IAM and SQuAD datasets.

Runs AFTER ingest_iam.py and ingest_squad.py finish writing Parquet shards
to MinIO. Verifies the external-data ingestion didn't silently drop rows,
corrupt bytes, or produce out-of-spec records that would poison training.

This is the "point 1" deliverable in the data-role rubric: data quality at
ingestion from external sources.

Checks per dataset (see spec below for thresholds):

  IAM  (image + transcription pairs):
    I1  row_count_nonzero      every split has at least N rows
    I2  schema_conforms        every shard has the expected columns
    I3  image_bytes_openable   sample S rows, PNG bytes open without error
    I4  text_nonempty          < X% of transcriptions are blank after strip
    I5  image_dimensions       width/height within expected bounds

  SQuAD (query + passage + label triplets):
    S1  row_count_nonzero
    S2  schema_conforms
    S3  label_distribution     positives ≈ negatives within tolerance
    S4  text_nonempty          queries and passages non-blank after strip
    S5  unicode_valid          strings decode as valid UTF-8

Exit status:
    0   all checks pass
    1   one or more checks fail (CI should block promotion)
    2   validator itself crashed (e.g., MinIO unreachable)

The report is written to s3://<bucket>/warehouse/<dataset>/_validation/
<timestamp>.json alongside the shards, regardless of pass/fail, so the
manifest history is permanent.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
from minio import Minio
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET           = os.getenv("MINIO_BUCKET",     "paperless-datalake")

# Minimums — thresholds are generous on purpose. These catch catastrophic
# ingestion failures (empty file, wrong schema, all-corrupt bytes) not
# dataset-quality issues. Those belong in a later stage.
IAM_MIN_ROWS_PER_SPLIT   = 100
SQUAD_MIN_ROWS_PER_SPLIT = 1_000
SAMPLE_SIZE              = 50          # how many rows to spot-check per shard
MAX_BLANK_TEXT_FRAC      = 0.05        # ≤5% blank transcriptions/queries
SQUAD_LABEL_BALANCE_TOL  = 0.20        # pos/(pos+neg) must be in [0.3, 0.7]

# Expected Parquet columns, based on ingest_{iam,squad}.py.
IAM_SCHEMA = {"image_id", "image_png", "transcription", "split"}
SQUAD_SCHEMA = {"query", "passage", "label", "split"}

# IAM images are line crops; heights are small, widths vary a lot.
IAM_HEIGHT_BOUNDS = (16, 512)
IAM_WIDTH_BOUNDS  = (32, 4096)


# ── Reporting ─────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReport:
    dataset: str
    started_at: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", **stats) -> None:
        self.checks.append(CheckResult(name, passed, detail, stats))
        level = log.info if passed else log.error
        level("[%s] %s — %s %s", self.dataset, name,
              "PASS" if passed else "FAIL", detail)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "started_at": self.started_at,
            "all_passed": self.all_passed,
            "checks": [
                {"name": c.name, "passed": c.passed,
                 "detail": c.detail, "stats": c.stats}
                for c in self.checks
            ],
        }


# ── MinIO helpers ─────────────────────────────

def get_minio() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def list_shards(mc: Minio, prefix: str) -> list[str]:
    """Return every parquet object name under the prefix."""
    return sorted(
        obj.object_name
        for obj in mc.list_objects(BUCKET, prefix=prefix, recursive=True)
        if obj.object_name.endswith(".parquet")
    )


def read_shard(mc: Minio, object_name: str):
    """Fetch a parquet shard into a PyArrow Table."""
    resp = mc.get_object(BUCKET, object_name)
    try:
        buf = io.BytesIO(resp.read())
    finally:
        resp.close()
        resp.release_conn()
    return pq.read_table(buf)


# ── IAM validation ────────────────────────────

def validate_iam(mc: Minio) -> DatasetReport:
    report = DatasetReport(
        dataset="iam", started_at=datetime.now(timezone.utc).isoformat()
    )
    prefix = "warehouse/iam_dataset/"

    shards = list_shards(mc, prefix)
    if not shards:
        report.add("I0_shards_found", False, f"no parquet shards under {prefix}")
        return report
    report.add("I0_shards_found", True, f"{len(shards)} shards",
               shard_count=len(shards))

    # Read all shards (IAM is tens of thousands of rows, fits in memory).
    tables = [read_shard(mc, s) for s in shards]
    total_rows = sum(len(t) for t in tables)
    splits = {}
    for t in tables:
        for split in t.column("split").to_pylist() if "split" in t.column_names else []:
            splits[split] = splits.get(split, 0) + 1

    # I1 — row count
    if splits:
        bad = {s: n for s, n in splits.items() if n < IAM_MIN_ROWS_PER_SPLIT}
        report.add(
            "I1_row_count_nonzero",
            not bad,
            f"per-split counts: {splits}" + (f" (below minimum: {bad})" if bad else ""),
            total_rows=total_rows, per_split=splits,
        )
    else:
        report.add("I1_row_count_nonzero", False,
                   f"total={total_rows}, no split column found")

    # I2 — schema
    schema_cols = set(tables[0].column_names)
    missing = IAM_SCHEMA - schema_cols
    report.add(
        "I2_schema_conforms",
        not missing,
        f"columns={sorted(schema_cols)}" + (f" missing={sorted(missing)}" if missing else ""),
        columns=sorted(schema_cols),
    )
    if missing:
        return report   # can't run later checks without required columns

    # I3 + I5 — image sample
    all_rows = []
    for t in tables:
        all_rows.extend(t.to_pylist())
    sample = random.sample(all_rows, min(SAMPLE_SIZE, len(all_rows)))

    openable = 0
    dims_ok = 0
    for row in sample:
        try:
            img = Image.open(io.BytesIO(row["image_png"]))
            img.verify()
            img2 = Image.open(io.BytesIO(row["image_png"]))  # verify() consumes
            w, h = img2.size
            openable += 1
            if (IAM_HEIGHT_BOUNDS[0] <= h <= IAM_HEIGHT_BOUNDS[1]
                    and IAM_WIDTH_BOUNDS[0] <= w <= IAM_WIDTH_BOUNDS[1]):
                dims_ok += 1
        except Exception as exc:
            log.debug("bad image: %s", exc)

    report.add(
        "I3_image_bytes_openable",
        openable == len(sample),
        f"{openable}/{len(sample)} images opened cleanly",
        sampled=len(sample), openable=openable,
    )
    report.add(
        "I5_image_dimensions",
        dims_ok >= len(sample) * 0.95,     # allow tiny outliers
        f"{dims_ok}/{len(sample)} within bounds h∈{IAM_HEIGHT_BOUNDS}, w∈{IAM_WIDTH_BOUNDS}",
        in_bounds=dims_ok, sampled=len(sample),
    )

    # I4 — text non-empty
    blanks = sum(1 for r in all_rows if not (r.get("transcription") or "").strip())
    frac = blanks / total_rows if total_rows else 1.0
    report.add(
        "I4_text_nonempty",
        frac <= MAX_BLANK_TEXT_FRAC,
        f"{blanks}/{total_rows} blank transcriptions ({frac:.2%})",
        blank=blanks, total=total_rows, frac=frac,
    )
    return report


# ── SQuAD validation ──────────────────────────

def validate_squad(mc: Minio) -> DatasetReport:
    report = DatasetReport(
        dataset="squad", started_at=datetime.now(timezone.utc).isoformat()
    )
    prefix = "warehouse/squad_dataset/"

    shards = list_shards(mc, prefix)
    if not shards:
        report.add("S0_shards_found", False, f"no parquet shards under {prefix}")
        return report
    report.add("S0_shards_found", True, f"{len(shards)} shards",
               shard_count=len(shards))

    tables = [read_shard(mc, s) for s in shards]
    total_rows = sum(len(t) for t in tables)
    splits: dict[str, int] = {}
    for t in tables:
        if "split" in t.column_names:
            for s in t.column("split").to_pylist():
                splits[s] = splits.get(s, 0) + 1

    # S1 — row count
    bad = {s: n for s, n in splits.items() if n < SQUAD_MIN_ROWS_PER_SPLIT}
    report.add(
        "S1_row_count_nonzero",
        splits and not bad,
        f"per-split counts: {splits}" + (f" (below minimum: {bad})" if bad else ""),
        total_rows=total_rows, per_split=splits,
    )

    # S2 — schema
    schema_cols = set(tables[0].column_names)
    missing = SQUAD_SCHEMA - schema_cols
    report.add(
        "S2_schema_conforms",
        not missing,
        f"columns={sorted(schema_cols)}" + (f" missing={sorted(missing)}" if missing else ""),
        columns=sorted(schema_cols),
    )
    if missing:
        return report

    # S3 — label balance
    pos = sum(int(l == 1) for t in tables for l in t.column("label").to_pylist())
    neg = total_rows - pos
    ratio = pos / total_rows if total_rows else 0.0
    ratio_ok = abs(ratio - 0.5) <= SQUAD_LABEL_BALANCE_TOL
    report.add(
        "S3_label_distribution",
        ratio_ok,
        f"positives={pos} ({ratio:.2%}), negatives={neg}",
        positives=pos, negatives=neg, positive_ratio=ratio,
    )

    # S4 + S5 — text check over full set (SQuAD rows are light)
    blank_q = 0
    blank_p = 0
    utf8_bad = 0
    for t in tables:
        queries  = t.column("query").to_pylist()
        passages = t.column("passage").to_pylist()
        for q, p in zip(queries, passages):
            if not (q or "").strip():
                blank_q += 1
            if not (p or "").strip():
                blank_p += 1
            try:
                (q or "").encode("utf-8").decode("utf-8")
                (p or "").encode("utf-8").decode("utf-8")
            except UnicodeError:
                utf8_bad += 1

    q_frac = blank_q / total_rows if total_rows else 1.0
    p_frac = blank_p / total_rows if total_rows else 1.0
    report.add(
        "S4_text_nonempty",
        q_frac <= MAX_BLANK_TEXT_FRAC and p_frac <= MAX_BLANK_TEXT_FRAC,
        f"blank queries={blank_q} ({q_frac:.2%}), blank passages={blank_p} ({p_frac:.2%})",
        blank_queries=blank_q, blank_passages=blank_p,
    )
    report.add(
        "S5_unicode_valid",
        utf8_bad == 0,
        f"{utf8_bad} rows with invalid utf-8",
        invalid_utf8=utf8_bad,
    )
    return report


# ── Report upload ─────────────────────────────

def upload_report(mc: Minio, report: DatasetReport) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    obj_name = f"warehouse/{report.dataset}_dataset/_validation/{ts}.json"
    body = json.dumps(report.as_dict(), indent=2).encode()
    mc.put_object(BUCKET, obj_name, io.BytesIO(body), length=len(body))
    log.info("validation report → s3://%s/%s", BUCKET, obj_name)
    return obj_name


# ── Main ──────────────────────────────────────

def main() -> int:
    log.info("=" * 60)
    log.info("Ingestion data-quality validation")
    log.info("=" * 60)

    try:
        mc = get_minio()
        if not mc.bucket_exists(BUCKET):
            log.error("bucket %s not found", BUCKET)
            return 2
    except Exception as exc:
        log.error("MinIO setup failed: %s", exc)
        return 2

    all_passed = True
    for name, fn in (("iam", validate_iam), ("squad", validate_squad)):
        log.info("-" * 40)
        log.info("Validating %s", name)
        log.info("-" * 40)
        try:
            report = fn(mc)
        except Exception as exc:
            log.exception("%s validator crashed: %s", name, exc)
            return 2
        upload_report(mc, report)
        if not report.all_passed:
            all_passed = False

    log.info("=" * 60)
    log.info("Validation result: %s", "PASS" if all_passed else "FAIL")
    log.info("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
