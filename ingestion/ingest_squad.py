"""
Ingest SQuAD 2.0 → MinIO
Downloads directly from the SQuAD website, formats into (query, passage, label) triplets
for training a bi-encoder retrieval model.
Writes Parquet shards to paperless-datalake/warehouse/squad_dataset/{split}/
"""

import os
import io
import json
import logging
import hashlib
import random
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
BUCKET = "paperless-datalake"
PREFIX = "warehouse/squad_dataset"
SHARD_SIZE = 2000

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def download_squad(url: str) -> list[dict]:
    """Download SQuAD JSON and flatten into a list of {question, context, is_impossible}."""
    log.info(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())

    samples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                samples.append({
                    "question": qa["question"],
                    "context": context,
                    "is_impossible": qa.get("is_impossible", False),
                })
    return samples


def make_triplets(samples: list[dict], split: str) -> list[dict]:
    """
    Convert SQuAD into retrieval triplets:
    - Answerable questions → positive pair (query, context, label=1)
    - Unanswerable questions → hard negative pair (query, context, label=0)
    - Also generate in-batch negatives from mismatched positives
    """
    random.seed(42)

    positives = []
    negatives = []
    all_contexts = []

    for sample in samples:
        query = sample["question"]
        context = sample["context"]

        if not sample["is_impossible"]:
            positives.append({"query": query, "passage": context, "label": 1})
            all_contexts.append(context)
        else:
            negatives.append({"query": query, "passage": context, "label": 0})

    # Generate hard negatives from mismatched positives
    synthetic_negatives = []
    for p in positives[: len(positives) // 2]:
        random_ctx = random.choice(all_contexts)
        if random_ctx != p["passage"]:
            synthetic_negatives.append({
                "query": p["query"],
                "passage": random_ctx,
                "label": 0,
            })

    all_triplets = positives + negatives + synthetic_negatives
    random.shuffle(all_triplets)
    log.info(
        f"  {split}: {len(positives)} pos, {len(negatives)} neg, "
        f"{len(synthetic_negatives)} synthetic neg = {len(all_triplets)} total"
    )
    return all_triplets


def upload_parquet_shard(client: Minio, table: pa.Table, split: str, shard_idx: int):
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    obj_name = f"{PREFIX}/{split}/shard_{shard_idx:04d}.parquet"
    client.put_object(BUCKET, obj_name, buf, length=buf.getbuffer().nbytes)
    log.info(f"Uploaded {obj_name} ({len(table)} rows)")


def ingest_split(client: Minio, samples: list[dict], split: str):
    triplets = make_triplets(samples, split)

    queries, passages, labels, triplet_ids = [], [], [], []
    shard_idx = 0

    for t in triplets:
        tid = hashlib.md5(f"{t['query']}:{t['passage']}".encode()).hexdigest()[:12]
        queries.append(t["query"])
        passages.append(t["passage"])
        labels.append(t["label"])
        triplet_ids.append(tid)

        if len(queries) >= SHARD_SIZE:
            table = pa.table({
                "triplet_id": pa.array(triplet_ids, type=pa.string()),
                "query": pa.array(queries, type=pa.string()),
                "passage": pa.array(passages, type=pa.string()),
                "label": pa.array(labels, type=pa.int32()),
                "split": pa.array([split] * len(queries), type=pa.string()),
            })
            upload_parquet_shard(client, table, split, shard_idx)
            shard_idx += 1
            queries, passages, labels, triplet_ids = [], [], [], []

    if queries:
        table = pa.table({
            "triplet_id": pa.array(triplet_ids, type=pa.string()),
            "query": pa.array(queries, type=pa.string()),
            "passage": pa.array(passages, type=pa.string()),
            "label": pa.array(labels, type=pa.int32()),
            "split": pa.array([split] * len(queries), type=pa.string()),
        })
        upload_parquet_shard(client, table, split, shard_idx)


def upload_metadata(client: Minio):
    meta = {
        "source": "SQuAD 2.0 (rajpurkar.github.io)",
        "description": "SQuAD 2.0 formatted as retrieval triplets (query, passage, label)",
        "schema": ["triplet_id", "query", "passage", "label", "split"],
        "label_mapping": {
            "1": "relevant (answerable)",
            "0": "irrelevant (unanswerable or mismatched)",
        },
    }
    buf = io.BytesIO(json.dumps(meta, indent=2).encode())
    client.put_object(BUCKET, f"{PREFIX}/metadata.json", buf, length=buf.getbuffer().nbytes)
    log.info("Uploaded metadata.json")


def main():
    client = get_minio_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    for split, url in SQUAD_URLS.items():
        samples = download_squad(url)
        log.info(f"Downloaded {split}: {len(samples)} QA pairs")
        ingest_split(client, samples, split)

    upload_metadata(client)
    log.info("SQuAD ingestion complete.")


if __name__ == "__main__":
    main()
