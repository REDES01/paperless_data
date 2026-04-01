"""
Stub API — Simulates Paperless-ngx service endpoints.
Receives synthetic traffic and writes to PostgreSQL, MinIO, and Redpanda.

Endpoints:
  POST /api/upload      — upload a document (image + metadata)
  POST /api/corrections — submit an HTR correction
  POST /api/search      — execute a search query
  POST /api/feedback    — submit search feedback
  GET  /api/health      — health check
"""

import io
import json
import os
import uuid
import logging
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from minio import Minio
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_DSN = os.getenv("DB_DSN", "host=postgres dbname=paperless user=user password=paperless_postgres")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "paperless_minio")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "redpanda:9092")

app = FastAPI(title="Paperless Stub API")

# Register UUID adapter so psycopg2 handles UUID arrays properly
psycopg2.extras.register_uuid()

# ── Clients ────────────────────────────────────
_pg = None
_minio = None
_producer = None


def get_pg():
    global _pg
    if _pg is None or _pg.closed:
        _pg = psycopg2.connect(DB_DSN)
        _pg.autocommit = True
    return _pg


def get_minio():
    global _minio
    if _minio is None:
        _minio = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY, secure=False)
    return _minio


def get_producer():
    global _producer
    if _producer is None:
        _producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    return _producer


def publish(topic: str, event: dict):
    p = get_producer()
    p.produce(topic, json.dumps(event).encode())
    p.flush()


# ── Upload endpoint ────────────────────────────
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form("user_upload"),
    page_count: int = Form(1),
):
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Save image to MinIO
    content = await file.read()
    mc = get_minio()
    bucket = "paperless-images"
    obj_path = f"documents/{doc_id}/page_1.png"
    mc.put_object(bucket, obj_path, io.BytesIO(content), length=len(content))

    # Write to PostgreSQL
    conn = get_pg()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (id, filename, source, page_count, uploaded_at)
               VALUES (%s, %s, %s, %s, NOW())""",
            (doc_id, file.filename, source, page_count),
        )
        page_id = str(uuid.uuid4())
        cur.execute(
            """INSERT INTO document_pages (id, document_id, page_number, image_s3_url)
               VALUES (%s, %s, 1, %s)""",
            (page_id, doc_id, f"s3://{bucket}/{obj_path}"),
        )
        # Create a synthetic handwritten region
        region_id = str(uuid.uuid4())
        crop_path = f"documents/{doc_id}/regions/{region_id}.png"
        mc.put_object(bucket, crop_path, io.BytesIO(content[:1024]), length=min(1024, len(content)))
        cur.execute(
            """INSERT INTO handwritten_regions (id, page_id, crop_s3_url, htr_output, htr_confidence)
               VALUES (%s, %s, %s, %s, %s)""",
            (region_id, page_id, f"s3://{bucket}/{crop_path}", "synthetic htr output", 0.75),
        )

    # Publish event
    publish("paperless.uploads", {
        "document_id": doc_id,
        "page_count": page_count,
        "uploaded_at": now,
        "source": source,
    })

    log.info(f"Uploaded doc {doc_id} ({file.filename})")
    return {"document_id": doc_id, "page_id": page_id, "region_id": region_id}


# ── Correction endpoint ────────────────────────
class CorrectionRequest(BaseModel):
    region_id: str
    corrected_text: str
    user_id: str | None = None
    opted_in: bool = True


@app.post("/api/corrections")
async def submit_correction(req: CorrectionRequest):
    corr_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_pg()
    with conn.cursor() as cur:
        cur.execute("SELECT htr_output FROM handwritten_regions WHERE id = %s", (req.region_id,))
        row = cur.fetchone()
        original = row[0] if row else ""

        cur.execute(
            """INSERT INTO htr_corrections
               (id, region_id, user_id, original_text, corrected_text, opted_in)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (corr_id, req.region_id, req.user_id, original, req.corrected_text, req.opted_in),
        )

    publish("paperless.corrections", {
        "region_id": req.region_id,
        "document_id": None,
        "corrected_text": req.corrected_text,
        "corrected_at": now,
        "opted_in": req.opted_in,
    })

    log.info(f"Correction {corr_id} for region {req.region_id}")
    return {"correction_id": corr_id}


# ── Search endpoint ────────────────────────────
class SearchRequest(BaseModel):
    query_text: str
    user_id: str | None = None
    is_test_account: bool = False


@app.post("/api/search")
async def search_documents(req: SearchRequest):
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_pg()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents ORDER BY RANDOM() LIMIT 5")
        result_ids = [row[0] for row in cur.fetchall()]  # keep as UUID objects

        cur.execute(
            """INSERT INTO query_sessions
               (id, query_text, user_id, is_test_account, result_doc_ids)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, req.query_text, req.user_id, req.is_test_account, result_ids),
        )

    # Convert to strings for JSON response
    result_id_strs = [str(r) for r in result_ids]

    publish("paperless.queries", {
        "session_id": session_id,
        "query_text": req.query_text,
        "queried_at": now,
        "result_doc_ids": result_id_strs,
    })

    log.info(f"Search '{req.query_text}' → {len(result_ids)} results")
    return {"session_id": session_id, "result_doc_ids": result_id_strs}


# ── Feedback endpoint ──────────────────────────
class FeedbackRequest(BaseModel):
    session_id: str
    document_id: str
    feedback_type: str  # 'click', 'thumbs_up', 'thumbs_down'


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    fb_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = get_pg()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO search_feedback (id, session_id, document_id, feedback_type)
               VALUES (%s, %s, %s, %s)""",
            (fb_id, req.session_id, req.document_id, req.feedback_type),
        )

    publish("paperless.feedback", {
        "session_id": req.session_id,
        "document_id": req.document_id,
        "feedback_type": req.feedback_type,
        "created_at": now,
    })

    log.info(f"Feedback {req.feedback_type} on doc {req.document_id}")
    return {"feedback_id": fb_id}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
