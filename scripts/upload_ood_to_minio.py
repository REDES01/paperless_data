"""
Upload OOD samples to the VM's MinIO under paperless-images/ood/.

Reads every PNG from OUT_DIR, puts it at s3://paperless-images/ood/<filename>.
Prints the resulting s3:// URLs so you can paste them into the demo script.

Assumes:
  - MinIO is reachable at MINIO_ENDPOINT (default 129.114.109.45:9000)
  - Credentials are minioadmin/minioadmin (paperless-ml's defaults)
  - The paperless-images bucket already exists (created by paperless-ml's
    minio-init container on first compose up)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from minio import Minio

MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "129.114.109.45:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE     = os.environ.get("MINIO_SECURE", "false").lower() == "true"

BUCKET = "paperless-images"
PREFIX = "ood"
SRC    = Path(os.environ.get("SRC_DIR", "ood_samples"))


def main() -> int:
    mc = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

    if not mc.bucket_exists(BUCKET):
        print(f"bucket {BUCKET} doesn't exist — is the stack up?", file=sys.stderr)
        return 1

    files = sorted(SRC.glob("*.png"))
    if not files:
        print(f"no PNGs in {SRC}/", file=sys.stderr)
        return 1

    urls: list[str] = []
    for f in files:
        key = f"{PREFIX}/{f.name}"
        size = f.stat().st_size
        with open(f, "rb") as fh:
            mc.put_object(BUCKET, key, fh, length=size, content_type="image/png")
        url = f"s3://{BUCKET}/{key}"
        urls.append(url)
        print(f"  uploaded  {url}  ({size} bytes)")

    print()
    print(f"done: {len(urls)} OOD samples at s3://{BUCKET}/{PREFIX}/")
    print()
    print("--- paste into drift demo loop ---")
    for u in urls:
        print(u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
