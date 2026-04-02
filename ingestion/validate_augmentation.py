"""
Quality Validation for Synthetic/Augmented Data

Implements best practices from class:
  1. Check for blank/corrupted images
  2. Check for near-duplicates (perceptual hashing)
  3. Check pixel distribution drift between original and augmented
  4. Report quality metrics and flag bad samples

Run after augmentation to validate quality before using for training.
"""

import os
import io
import json
import logging
import hashlib
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from minio import Minio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.getenv("MINIO_BUCKET", "paperless-datalake")
IAM_PREFIX = "warehouse/iam_dataset"

# Quality thresholds
MIN_PIXEL_STD = 5.0        # images with std < this are nearly blank
MAX_BLANK_RATIO = 0.95     # if >95% of pixels are white, flag as blank
MAX_DUPLICATE_RATIO = 0.05 # allow at most 5% near-duplicates
MAX_DRIFT = 0.15           # max allowed shift in mean pixel value


def get_minio_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def perceptual_hash(img_bytes: bytes, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash (average hash) for near-duplicate detection."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.array(img)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    return "".join(["1" if b else "0" for b in bits])


def check_blank(img_bytes: bytes) -> dict:
    """Check if an image is blank or near-blank."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)
    pixel_std = arr.std()
    white_ratio = (arr > 240).sum() / arr.size
    is_blank = pixel_std < MIN_PIXEL_STD or white_ratio > MAX_BLANK_RATIO
    return {
        "pixel_std": round(float(pixel_std), 2),
        "white_ratio": round(float(white_ratio), 4),
        "is_blank": is_blank,
    }


def compute_distribution(img_bytes: bytes) -> dict:
    """Compute pixel distribution stats for drift checking."""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)
    return {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "median": round(float(np.median(arr)), 2),
    }


def list_shards(client: Minio, split: str) -> list[str]:
    prefix = f"{IAM_PREFIX}/{split}/"
    return [obj.object_name for obj in client.list_objects(BUCKET, prefix=prefix)
            if obj.object_name.endswith(".parquet")]


def validate_split(client: Minio, split: str, max_samples: int = 200) -> dict:
    """Validate a split's images. Samples up to max_samples for efficiency."""
    shards = list_shards(client, split)
    if not shards:
        return {"split": split, "status": "empty", "num_shards": 0}

    blank_count = 0
    total_count = 0
    hashes = []
    distributions = []

    for shard_path in shards[:3]:  # check first 3 shards
        response = client.get_object(BUCKET, shard_path)
        table = pq.read_table(io.BytesIO(response.read()))
        response.close()

        for i in range(min(len(table), max_samples // len(shards))):
            img_bytes = table.column("image_png")[i].as_py()
            total_count += 1

            # Check blank
            blank_info = check_blank(img_bytes)
            if blank_info["is_blank"]:
                blank_count += 1

            # Perceptual hash
            phash = perceptual_hash(img_bytes)
            hashes.append(phash)

            # Distribution
            dist = compute_distribution(img_bytes)
            distributions.append(dist)

    # Near-duplicate check
    hash_counts = Counter(hashes)
    duplicate_count = sum(c - 1 for c in hash_counts.values() if c > 1)
    duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0

    # Distribution stats
    means = [d["mean"] for d in distributions]
    avg_mean = np.mean(means) if means else 0
    avg_std = np.mean([d["std"] for d in distributions]) if distributions else 0

    return {
        "split": split,
        "num_shards": len(shards),
        "samples_checked": total_count,
        "blank_images": blank_count,
        "blank_ratio": round(blank_count / total_count, 4) if total_count > 0 else 0,
        "near_duplicates": duplicate_count,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "avg_pixel_mean": round(float(avg_mean), 2),
        "avg_pixel_std": round(float(avg_std), 2),
    }


def check_drift(original_stats: dict, augmented_stats: dict) -> dict:
    """Check if augmented data has drifted too far from original."""
    mean_drift = abs(original_stats["avg_pixel_mean"] - augmented_stats["avg_pixel_mean"])
    std_drift = abs(original_stats["avg_pixel_std"] - augmented_stats["avg_pixel_std"])

    # Normalize drift relative to original range (0-255)
    normalized_drift = mean_drift / 255.0

    return {
        "mean_drift": round(float(mean_drift), 2),
        "std_drift": round(float(std_drift), 2),
        "normalized_drift": round(float(normalized_drift), 4),
        "drift_acceptable": normalized_drift <= MAX_DRIFT,
    }


def main():
    log.info("=" * 60)
    log.info("Synthetic Data Quality Validation")
    log.info("=" * 60)

    client = get_minio_client()

    report = {
        "pipeline": "iam_augmentation_quality_check",
        "thresholds": {
            "min_pixel_std": MIN_PIXEL_STD,
            "max_blank_ratio": MAX_BLANK_RATIO,
            "max_duplicate_ratio": MAX_DUPLICATE_RATIO,
            "max_drift": MAX_DRIFT,
        },
        "splits": {},
        "drift_checks": {},
        "overall_pass": True,
    }

    # Validate original and augmented splits
    for split in ["train", "train_augmented", "validation", "validation_augmented"]:
        log.info(f"\nValidating split: {split}")
        stats = validate_split(client, split)
        report["splits"][split] = stats

        log.info(f"  Shards: {stats.get('num_shards', 0)}")
        log.info(f"  Samples checked: {stats.get('samples_checked', 0)}")
        log.info(f"  Blank images: {stats.get('blank_images', 0)} ({stats.get('blank_ratio', 0):.1%})")
        log.info(f"  Near-duplicates: {stats.get('near_duplicates', 0)} ({stats.get('duplicate_ratio', 0):.1%})")
        log.info(f"  Avg pixel mean: {stats.get('avg_pixel_mean', 0)}")
        log.info(f"  Avg pixel std: {stats.get('avg_pixel_std', 0)}")

        # Quality gates
        if stats.get("blank_ratio", 0) > MAX_BLANK_RATIO:
            log.warning(f"  FAIL: Too many blank images")
            report["overall_pass"] = False
        if stats.get("duplicate_ratio", 0) > MAX_DUPLICATE_RATIO:
            log.warning(f"  FAIL: Too many near-duplicates")
            report["overall_pass"] = False

    # Drift checks between original and augmented
    for base_split in ["train", "validation"]:
        aug_split = f"{base_split}_augmented"
        if base_split in report["splits"] and aug_split in report["splits"]:
            orig = report["splits"][base_split]
            aug = report["splits"][aug_split]
            if orig.get("samples_checked", 0) > 0 and aug.get("samples_checked", 0) > 0:
                drift = check_drift(orig, aug)
                report["drift_checks"][f"{base_split}_vs_{aug_split}"] = drift
                log.info(f"\nDrift check: {base_split} → {aug_split}")
                log.info(f"  Mean drift: {drift['mean_drift']} pixels")
                log.info(f"  Normalized drift: {drift['normalized_drift']:.2%}")
                log.info(f"  Acceptable: {drift['drift_acceptable']}")
                if not drift["drift_acceptable"]:
                    log.warning(f"  FAIL: Drift too high")
                    report["overall_pass"] = False

    # Upload report
    log.info(f"\n{'=' * 60}")
    if report["overall_pass"]:
        log.info("OVERALL: PASS — Augmented data quality is acceptable")
    else:
        log.warning("OVERALL: FAIL — Review flagged issues above")

    buf = io.BytesIO(json.dumps(report, indent=2).encode())
    report_path = f"{IAM_PREFIX}/quality_report.json"
    client.put_object(BUCKET, report_path, buf, length=buf.getbuffer().nbytes)
    log.info(f"Report uploaded to {report_path}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
