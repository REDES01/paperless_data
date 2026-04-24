"""
IAM handwriting crop pool.

Loads a sample of real handwritten line images from MinIO (the IAM
dataset shards ingested earlier into s3://paperless-datalake/
iam_dataset/train/shard_*.parquet) into memory so the data generator
can paste REAL handwriting into synthetic document pages.

Why this matters:
  The previous generator drew a blue squiggly line to trigger region
  detection. The slicer found the ink, but TrOCR produced gibberish
  because there were no letterforms to transcribe. behavior_emulator
  had nothing to correct toward — all corrections ended up as no-ops.

  Real IAM crops have real letter shapes, so TrOCR produces plausible
  transcriptions (often correct, sometimes with character errors).
  Corrections perturbing those outputs produce real training signal.

Memory footprint: a 500-crop pool is ~25-50MB (most crops are ~30-60KB
grayscale PNGs). We keep them as bytes; each gets PIL-decoded on demand
inside the page compositor.
"""

from __future__ import annotations

import io
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

import pyarrow.parquet as pq
from minio import Minio

log = logging.getLogger(__name__)

IAM_BUCKET = os.environ.get("IAM_BUCKET", "paperless-datalake")
IAM_PREFIX = os.environ.get("IAM_PREFIX", "warehouse/iam_dataset/train/")

# How many crops to keep in memory. 500 is plenty of variety for a demo
# and keeps RAM usage <50MB. The pool never refills — we draw with
# replacement from the initial sample, which is fine for a generator.
POOL_SIZE = int(os.environ.get("IAM_POOL_SIZE", "500"))

# Read at most this many parquet shards to reach POOL_SIZE. Each shard is
# ~40-50MB and holds several thousand rows, so 1-2 shards is usually
# enough. Capping protects against oversized downloads at startup.
MAX_SHARDS = int(os.environ.get("IAM_MAX_SHARDS", "3"))


@dataclass
class IAMCrop:
    image_png: bytes         # raw PNG bytes, ready to paste
    transcription: str       # ground-truth (not used by the generator
                             # today, but kept for future evaluation tooling)


class IAMPool:
    """In-memory pool of IAM handwriting crops.

    Loaded once at generator startup. `sample()` returns a random
    (image_png_bytes, transcription) tuple — caller uses the bytes to
    paste into a page and can optionally log the transcription for
    eval purposes.

    Falls back gracefully if MinIO is unreachable or the prefix is
    empty — `sample()` returns None and the caller should draw a
    squiggle stub instead. No crash-on-startup.
    """

    def __init__(
        self,
        endpoint: str = "minio:9000",
        access_key: str = "admin",
        secret_key: str = "paperless_minio",
        secure: bool = False,
    ):
        self._crops: list[IAMCrop] = []
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def load(self) -> int:
        """Download and decode IAM parquet shards until the pool reaches
        POOL_SIZE or we hit MAX_SHARDS. Returns final pool size."""
        try:
            objects = [
                o for o in self._client.list_objects(
                    IAM_BUCKET, prefix=IAM_PREFIX, recursive=True
                )
                if o.object_name.endswith(".parquet")
            ]
        except Exception as exc:
            log.warning(
                "IAM pool disabled: could not list %s/%s (%s)",
                IAM_BUCKET, IAM_PREFIX, exc,
            )
            return 0

        if not objects:
            log.warning("IAM pool empty: no .parquet shards under %s", IAM_PREFIX)
            return 0

        # Shuffle so successive generator restarts don't load the same
        # first shard — gives variation across runs.
        random.shuffle(objects)

        for obj in objects[:MAX_SHARDS]:
            try:
                resp = self._client.get_object(IAM_BUCKET, obj.object_name)
                data = resp.read()
                resp.close()
                resp.release_conn()
            except Exception as exc:
                log.warning("failed to download %s: %s", obj.object_name, exc)
                continue

            try:
                table = pq.read_table(io.BytesIO(data))
            except Exception as exc:
                log.warning("failed to parse parquet %s: %s", obj.object_name, exc)
                continue

            images = table.column("image_png").to_pylist()
            texts = table.column("transcription").to_pylist()

            for img_bytes, text in zip(images, texts):
                if not img_bytes or not text:
                    continue
                self._crops.append(IAMCrop(image_png=img_bytes, transcription=text))
                if len(self._crops) >= POOL_SIZE:
                    break

            log.info(
                "IAM pool: loaded %d crops from %s (target %d)",
                len(self._crops), obj.object_name, POOL_SIZE,
            )
            if len(self._crops) >= POOL_SIZE:
                break

        log.info("IAM pool ready: %d crops loaded", len(self._crops))
        return len(self._crops)

    def sample(self) -> Optional[IAMCrop]:
        """Return one random crop, or None if the pool is empty."""
        if not self._crops:
            return None
        return random.choice(self._crops)

    def __len__(self) -> int:
        return len(self._crops)
