"""
Tests for batch_pipeline/quality.py

Exercise each rejection reason with synthetic candidate dicts. No DB, no
MinIO — every test runs in <1s. R4 uses a fake MinIO client that hardcodes
a miss/hit per key.

Run with:
    pytest batch_pipeline/tests/test_quality.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `batch_pipeline` importable when pytest runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from quality import (  # noqa: E402
    R1_NO_OP,
    R2_EMPTY_AFTER_STRIP,
    R3_CROP_URL_INVALID,
    R4_CROP_MISSING_IN_MINIO,
    R5_CORRECTION_TOO_LONG,
    QualityReport,
    filter_candidates,
)


# ── Factory helpers ───────────────────────────

def make(**over) -> dict:
    """Return a known-good candidate dict; override fields as needed."""
    base = {
        "correction_id": "11111111-1111-1111-1111-111111111111",
        "region_id":     "22222222-2222-2222-2222-222222222222",
        "corrected_text": "hello world",
        "original_text":  "helo wrld",                  # noticeably different
        "crop_s3_url":    "s3://paperless-images/abc/def.png",
    }
    base.update(over)
    return base


# ── Happy path ────────────────────────────────

def test_good_candidate_accepted():
    good = make()
    accepted, report = filter_candidates([good])
    assert accepted == [good]
    assert report.accepted == 1
    assert report.rejected == 0
    assert report.rejections == {}


def test_multiple_good_candidates():
    cs = [make(correction_id=f"c{i}", region_id=f"r{i}") for i in range(5)]
    accepted, report = filter_candidates(cs)
    assert len(accepted) == 5
    assert report.accepted == 5
    assert report.rejected == 0


# ── R1: no-op ─────────────────────────────────

def test_r1_identical_text_rejected():
    c = make(corrected_text="hello", original_text="hello")
    accepted, report = filter_candidates([c])
    assert accepted == []
    assert report.rejections == {R1_NO_OP: 1}


def test_r1_case_insensitive():
    c = make(corrected_text="Hello", original_text="hello")
    _, report = filter_candidates([c])
    assert report.rejections == {R1_NO_OP: 1}


def test_r1_whitespace_insensitive():
    c = make(corrected_text="  hello  ", original_text="hello")
    _, report = filter_candidates([c])
    assert report.rejections == {R1_NO_OP: 1}


def test_r1_empty_original_is_NOT_a_noop():
    # When there's no original HTR output, any non-empty correction is a real
    # addition and should not be rejected as a no-op.
    c = make(corrected_text="hello", original_text="")
    accepted, _ = filter_candidates([c])
    assert len(accepted) == 1


# ── R2: empty after strip ─────────────────────

def test_r2_whitespace_only_rejected():
    c = make(corrected_text="   \t\n   ")
    _, report = filter_candidates([c])
    assert report.rejections == {R2_EMPTY_AFTER_STRIP: 1}


def test_r2_empty_string_rejected():
    c = make(corrected_text="")
    _, report = filter_candidates([c])
    assert report.rejections == {R2_EMPTY_AFTER_STRIP: 1}


# ── R3: invalid URL ───────────────────────────

def test_r3_http_url_rejected():
    c = make(crop_s3_url="https://evil.example.com/x.png")
    _, report = filter_candidates([c])
    assert report.rejections == {R3_CROP_URL_INVALID: 1}


def test_r3_empty_url_rejected():
    c = make(crop_s3_url="")
    _, report = filter_candidates([c])
    assert report.rejections == {R3_CROP_URL_INVALID: 1}


def test_r3_relative_path_rejected():
    c = make(crop_s3_url="paperless-images/abc/def.png")   # no scheme
    _, report = filter_candidates([c])
    assert report.rejections == {R3_CROP_URL_INVALID: 1}


# ── R4: crop missing in MinIO ─────────────────

class _FakeMinioMissing:
    """stat_object always raises — every crop looks missing."""
    def stat_object(self, bucket, key):
        raise FileNotFoundError(f"NoSuchKey: {bucket}/{key}")


class _FakeMinioPresent:
    """stat_object returns without raising — every crop looks present."""
    def stat_object(self, bucket, key):
        return object()


class _FakeMinioSelective:
    """Rejects any crop whose key contains 'missing'."""
    def stat_object(self, bucket, key):
        if "missing" in key:
            raise FileNotFoundError(f"NoSuchKey: {bucket}/{key}")
        return object()


def test_r4_minio_miss_rejected():
    c = make(crop_s3_url="s3://paperless-images/gone.png")
    _, report = filter_candidates([c], minio_client=_FakeMinioMissing())
    assert report.rejections == {R4_CROP_MISSING_IN_MINIO: 1}


def test_r4_minio_hit_accepted():
    c = make(crop_s3_url="s3://paperless-images/present.png")
    accepted, _ = filter_candidates([c], minio_client=_FakeMinioPresent())
    assert len(accepted) == 1


def test_r4_skipped_when_no_client():
    c = make(crop_s3_url="s3://paperless-images/whatever.png")
    accepted, report = filter_candidates([c])     # no minio_client
    assert len(accepted) == 1
    assert R4_CROP_MISSING_IN_MINIO not in report.rejections


def test_r4_mixed_batch():
    cs = [
        make(correction_id="a", crop_s3_url="s3://b/ok-a.png"),
        make(correction_id="b", crop_s3_url="s3://b/missing-b.png"),
        make(correction_id="c", crop_s3_url="s3://b/ok-c.png"),
    ]
    accepted, report = filter_candidates(cs, minio_client=_FakeMinioSelective())
    assert len(accepted) == 2
    assert report.rejections == {R4_CROP_MISSING_IN_MINIO: 1}


# ── R5: too long ──────────────────────────────

def test_r5_oversized_text_rejected():
    c = make(corrected_text="x" * 10_000)
    _, report = filter_candidates([c])
    assert report.rejections == {R5_CORRECTION_TOO_LONG: 1}


def test_r5_boundary_accepted():
    # 2000 chars is the MAX; must be strictly greater to trigger.
    c = make(corrected_text="x" * 2000)
    accepted, _ = filter_candidates([c])
    assert len(accepted) == 1


# ── Precedence / combined ─────────────────────

def test_empty_and_noop_both_empty_is_r2():
    # Both original and corrected are "" — R2 fires before R1 because R2 is
    # cheaper and R1 requires non-empty original.
    c = make(corrected_text="", original_text="")
    _, report = filter_candidates([c])
    assert report.rejections == {R2_EMPTY_AFTER_STRIP: 1}


def test_multiple_reasons_in_same_batch():
    cs = [
        make(correction_id="good"),
        make(correction_id="empty", corrected_text=""),
        make(correction_id="noop", corrected_text="abc", original_text="abc"),
        make(correction_id="bad_url", crop_s3_url="http://bad.com/x"),
        make(correction_id="too_long", corrected_text="z" * 5000),
    ]
    accepted, report = filter_candidates(cs)
    assert len(accepted) == 1
    assert accepted[0]["correction_id"] == "good"
    assert report.rejections == {
        R2_EMPTY_AFTER_STRIP: 1,
        R1_NO_OP: 1,
        R3_CROP_URL_INVALID: 1,
        R5_CORRECTION_TOO_LONG: 1,
    }


# ── Report structure ──────────────────────────

def test_report_sample_limit():
    # Send 20 bad candidates of the same type; report should sample only 5.
    cs = [make(correction_id=f"c{i}", corrected_text="") for i in range(20)]
    _, report = filter_candidates(cs)
    assert report.rejected == 20
    assert report.rejections == {R2_EMPTY_AFTER_STRIP: 20}
    assert len(report.samples[R2_EMPTY_AFTER_STRIP]) == 5


def test_report_as_dict_serializable():
    import json
    cs = [make(corrected_text=""), make()]
    _, report = filter_candidates(cs)
    d = report.as_dict()
    assert d["accepted"] == 1
    assert d["rejected"] == 1
    # Must round-trip through JSON (this is what goes in the manifest).
    json.dumps(d)


def test_empty_input_is_ok():
    accepted, report = filter_candidates([])
    assert accepted == []
    assert report.total_candidates == 0
    assert report.accepted == 0
    assert report.rejected == 0
