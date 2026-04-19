"""
Build the drift-detection reference set + detector.

Run OFFLINE, once, when the HTR reference data changes. Uploads the saved
detector to s3://paperless-datalake/warehouse/drift_reference/htr_v1/cd/
so the drift_monitor service can load it at startup.

Pipeline
--------
1. Read 500 random IAM line crops from the existing Parquet shards at
   s3://paperless-datalake/warehouse/iam_dataset/train/.
2. Preprocess each (grayscale, resize to CROP_HEIGHTxCROP_WIDTH, scale to
   [0,1]). These are the same steps drift_monitor's /drift/check runs on
   live crops, so reference and production go through identical preprocessing.
3. Fit `MMDDriftOnline` with a lightweight convolutional feature extractor
   (no TrOCR dependency — keeps the detector image small and avoids pulling
   half of transformers just to build a reference).
4. `save_detector(cd_online, path)` then upload everything under that path
   to MinIO.

We deliberately do NOT use TrOCR's encoder as the feature extractor, even
though the online eval lab uses the model's own encoder. Reasons:
  - TrOCR's encoder is ~300MB and pulls transformers + tokenizers just
    for this one job.
  - A from-scratch Conv feature extractor is what alibi-detect uses in
    its own MMDDrift image examples — it's good enough for detecting
    distribution shifts between handwriting styles.
  - We are monitoring *input* distribution shift, not model-internal shift,
    so the choice of feature extractor isn't tied to the HTR model at all.

Usage
-----
    # With VM's MinIO exposed to localhost:9000
    MINIO_ENDPOINT=localhost:9000 \
    MINIO_ACCESS_KEY=minioadmin \
    MINIO_SECRET_KEY=minioadmin \
    python scripts/build_drift_reference.py

Environment variables mirror drift_monitor/service.py.
"""

from __future__ import annotations

import io
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from minio import Minio
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_drift_reference")

# ── Config ─────────────────────────────────────

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE     = os.environ.get("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "paperless-datalake")
SOURCE_PREFIX = os.environ.get("SOURCE_PREFIX", "warehouse/iam_dataset/train")
TARGET_BUCKET = os.environ.get("TARGET_BUCKET", "paperless-datalake")
TARGET_PREFIX = os.environ.get("TARGET_PREFIX", "warehouse/drift_reference/htr_v1/cd")

N_REFERENCE   = int(os.environ.get("N_REFERENCE", "500"))
CROP_HEIGHT   = int(os.environ.get("DRIFT_CROP_HEIGHT", "64"))
CROP_WIDTH    = int(os.environ.get("DRIFT_CROP_WIDTH",  "512"))
ERT           = int(os.environ.get("MMD_ERT",          "300"))
WINDOW_SIZE   = int(os.environ.get("MMD_WINDOW_SIZE",  "20"))

RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────

def get_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE,
    )


def load_iam_crops(mc: Minio, n: int) -> np.ndarray:
    """
    Sample `n` IAM crops from the ingested Parquet shards.

    IAM Parquet schema (from ingestion/ingest_iam.py):
        image_bytes  bytes  — PNG encoded
        text         str    — transcription
        split        str
        writer_id    str

    Returns an array of shape (n, 1, H, W), dtype float32, values in [0, 1].
    """
    shards = sorted(
        obj.object_name for obj in
        mc.list_objects(SOURCE_BUCKET, prefix=SOURCE_PREFIX + "/", recursive=True)
        if obj.object_name.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"no IAM shards under s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}")
    log.info("found %d IAM shards; reading enough rows for %d crops", len(shards), n)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shards)

    crops: list[np.ndarray] = []
    for shard in shards:
        if len(crops) >= n:
            break
        resp = mc.get_object(SOURCE_BUCKET, shard)
        try:
            table = pq.read_table(io.BytesIO(resp.read()))
        finally:
            resp.close()
            resp.release_conn()

        for image_bytes in table.column("image_bytes").to_pylist():
            if len(crops) >= n:
                break
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("L")
                img = img.resize((CROP_WIDTH, CROP_HEIGHT))
                crops.append(np.asarray(img, dtype=np.float32) / 255.0)
            except Exception as exc:
                log.debug("skipping bad crop: %s", exc)

    if len(crops) < n:
        raise RuntimeError(f"only {len(crops)}/{n} crops after reading all shards")

    arr = np.stack(crops, axis=0)           # (n, H, W)
    arr = np.expand_dims(arr, axis=1)       # (n, 1, H, W)
    log.info("reference tensor: shape=%s dtype=%s", arr.shape, arr.dtype)
    return arr


def build_detector(x_ref: np.ndarray):
    """Build an online MMD detector with a small CNN feature extractor."""
    import torch
    from torch import nn
    from functools import partial

    from alibi_detect.cd import MMDDriftOnline
    from alibi_detect.cd.pytorch import preprocess_drift

    # Small CNN: (1 → 16 → 32 channels) with stride-2 pools. Output is
    # flattened so MMD operates on a fixed-size feature vector.
    class CropEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 16)),
                nn.Flatten(),
            )

        def forward(self, x):
            return self.net(x)

    encoder = CropEncoder().eval()
    preprocess_fn = partial(preprocess_drift, model=encoder)

    log.info("fitting MMDDriftOnline (ert=%d, window=%d)", ERT, WINDOW_SIZE)
    cd = MMDDriftOnline(
        x_ref=x_ref,
        ert=ERT,
        window_size=WINDOW_SIZE,
        backend="pytorch",
        preprocess_fn=preprocess_fn,
    )
    return cd


def upload_dir(mc: Minio, local: Path, bucket: str, prefix: str) -> int:
    """Recursively upload a directory to MinIO under `prefix/`."""
    count = 0
    for path in local.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local).as_posix()
        key = f"{prefix}/{rel}"
        size = path.stat().st_size
        with open(path, "rb") as fh:
            mc.put_object(bucket, key, fh, length=size)
        log.info("  s3://%s/%s  (%d bytes)", bucket, key, size)
        count += 1
    return count


# ── Main ───────────────────────────────────────

def main() -> int:
    mc = get_minio()
    if not mc.bucket_exists(TARGET_BUCKET):
        raise SystemExit(f"target bucket {TARGET_BUCKET} doesn't exist")

    log.info("building drift reference")
    log.info("  source:       s3://%s/%s", SOURCE_BUCKET, SOURCE_PREFIX)
    log.info("  target:       s3://%s/%s", TARGET_BUCKET, TARGET_PREFIX)
    log.info("  n_reference:  %d", N_REFERENCE)
    log.info("  crop shape:   %sx%s  (H x W)", CROP_HEIGHT, CROP_WIDTH)

    x_ref = load_iam_crops(mc, N_REFERENCE)

    cd = build_detector(x_ref)

    from alibi_detect.saving import save_detector

    tmpdir = Path(tempfile.mkdtemp(prefix="build_drift_"))
    try:
        out = tmpdir / "cd"
        save_detector(cd, str(out))
        log.info("saved detector locally: %s", out)
        n = upload_dir(mc, out, TARGET_BUCKET, TARGET_PREFIX)
        log.info("uploaded %d files; detector ready", n)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
