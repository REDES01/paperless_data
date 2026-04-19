"""
Data quality checks for the HTR training-set compilation pipeline.

Used by batch_htr.py to filter candidate corrections *beyond* the SQL-level
filters (opted_in, non-empty, dedup), and to produce a rejection log that
documents WHAT was filtered out and WHY. The rejection log ships alongside
the training manifest so we have a paper trail for every dataset snapshot.

Required quality checks (one per rejection reason):

  R1  no_op_correction         corrected_text == original_text (stripped, case-insensitive)
  R2  empty_after_strip        corrected_text is whitespace-only after strip()
  R3  crop_url_invalid         crop_s3_url missing the expected s3:// scheme
  R4  crop_missing_in_minio    MinIO HEAD returns 404 (optional, behind --check-minio)
  R5  correction_too_long      corrected_text length > MAX_CORRECTION_CHARS (spam guard)

R4 is optional because it requires a live MinIO; unit tests skip it. The
others are pure-Python and always run.

The module is deliberately dependency-free at import time — no psycopg,
no pyarrow, no boto/minio. Only minio is imported lazily if R4 is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────
MAX_CORRECTION_CHARS = 2000    # > ~40 lines of text is almost certainly spam

# ── Rejection reasons ─────────────────────────
R1_NO_OP                 = "R1_no_op_correction"
R2_EMPTY_AFTER_STRIP     = "R2_empty_after_strip"
R3_CROP_URL_INVALID      = "R3_crop_url_invalid"
R4_CROP_MISSING_IN_MINIO = "R4_crop_missing_in_minio"
R5_CORRECTION_TOO_LONG   = "R5_correction_too_long"


@dataclass
class QualityReport:
    """Summary of what happened during filtering."""
    total_candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    # Sampled bad rows (at most `sample_limit` per reason) so we can eyeball them.
    samples: dict[str, list[dict]] = field(default_factory=dict)

    sample_limit: int = 5

    def reject(self, reason: str, candidate: dict) -> None:
        self.rejected += 1
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        bucket = self.samples.setdefault(reason, [])
        if len(bucket) < self.sample_limit:
            bucket.append({
                "correction_id": str(candidate.get("correction_id", "")),
                "region_id":     str(candidate.get("region_id", "")),
                "crop_s3_url":   candidate.get("crop_s3_url", ""),
                "original_text": (candidate.get("original_text") or "")[:80],
                "corrected_text": (candidate.get("corrected_text") or "")[:80],
            })

    def accept(self) -> None:
        self.accepted += 1

    def as_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_counts": dict(self.rejections),
            "rejection_samples": self.samples,
        }


# ── Individual checks (pure functions) ────────

def _is_no_op(candidate: dict) -> bool:
    """R1 — corrected text is identical to the original HTR output."""
    original = (candidate.get("original_text") or "").strip().lower()
    corrected = (candidate.get("corrected_text") or "").strip().lower()
    return bool(original) and original == corrected


def _is_empty(candidate: dict) -> bool:
    """R2 — corrected text is whitespace-only (DB has NOT NULL but allows blanks)."""
    return not (candidate.get("corrected_text") or "").strip()


def _has_invalid_url(candidate: dict) -> bool:
    """R3 — crop_s3_url doesn't look like an s3:// URL."""
    url = candidate.get("crop_s3_url") or ""
    return not url.startswith("s3://")


def _is_too_long(candidate: dict, max_chars: int = MAX_CORRECTION_CHARS) -> bool:
    """R5 — corrected text is suspiciously long (spam / paste-in of whole doc)."""
    return len(candidate.get("corrected_text") or "") > max_chars


# ── MinIO check (optional) ────────────────────

def _minio_object_exists(minio_client, bucket: str, key: str) -> bool:
    """HEAD the object; return True if it exists, False on 404."""
    try:
        minio_client.stat_object(bucket, key)
        return True
    except Exception as exc:
        # minio S3Error has a `code` attribute; "NoSuchKey" means 404.
        # Any other exception we treat as "unknown" → reject conservatively.
        log.debug("MinIO stat failed for %s/%s: %s", bucket, key, exc)
        return False


def _parse_s3_url(url: str) -> tuple[str, str] | None:
    """Split s3://bucket/key into (bucket, key). Returns None on malformed."""
    if not url.startswith("s3://"):
        return None
    rest = url[len("s3://"):]
    if "/" not in rest:
        return None
    bucket, _, key = rest.partition("/")
    return bucket, key


# ── Orchestrator ──────────────────────────────

def filter_candidates(
    candidates: Iterable[dict],
    *,
    minio_client=None,
) -> tuple[list[dict], QualityReport]:
    """
    Apply all quality checks and return (accepted, report).

    `candidates` is the output of batch_htr.fetch_candidates — a list of
    dicts with at minimum: correction_id, region_id, corrected_text,
    original_text, crop_s3_url.

    If `minio_client` is provided, R4 (crop existence) is enforced via HEAD.
    Otherwise R4 is skipped (useful for unit tests and dry runs).
    """
    report = QualityReport()
    accepted: list[dict] = []

    for c in candidates:
        report.total_candidates += 1

        # Order matters: cheapest checks first, MinIO last.
        if _is_empty(c):
            report.reject(R2_EMPTY_AFTER_STRIP, c)
            continue
        if _is_too_long(c):
            report.reject(R5_CORRECTION_TOO_LONG, c)
            continue
        if _is_no_op(c):
            report.reject(R1_NO_OP, c)
            continue
        if _has_invalid_url(c):
            report.reject(R3_CROP_URL_INVALID, c)
            continue

        if minio_client is not None:
            parsed = _parse_s3_url(c["crop_s3_url"])
            if parsed is None:
                report.reject(R3_CROP_URL_INVALID, c)
                continue
            bucket, key = parsed
            if not _minio_object_exists(minio_client, bucket, key):
                report.reject(R4_CROP_MISSING_IN_MINIO, c)
                continue

        report.accept()
        accepted.append(c)

    log.info(
        "quality: %d candidates → %d accepted, %d rejected (%s)",
        report.total_candidates, report.accepted, report.rejected,
        ", ".join(f"{k}={v}" for k, v in sorted(report.rejections.items())),
    )
    return accepted, report
