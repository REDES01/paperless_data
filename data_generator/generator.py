"""
Data Generator — Sends synthetic traffic to the Paperless stub API.

Simulates realistic user behavior:
  1. Uploads scanned documents (synthetic page images)
  2. Submits HTR corrections on random regions
  3. Executes search queries with realistic text
  4. Provides feedback on search results

Usage:
  python generator.py --api-url http://localhost:8000 --rate 2.0 --duration 300
"""

import argparse
import io
import uuid
import json
import random
import time
import logging
from datetime import datetime

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Synthetic data generators ──────────────────

# Realistic search queries for a document management system
SEARCH_QUERIES = [
    "budget report 2024", "meeting minutes march", "invoice #4521",
    "employee handbook update", "grant proposal NSF", "equipment purchase order",
    "travel reimbursement form", "committee meeting agenda", "course syllabus fall",
    "research paper draft", "lab safety protocol", "student enrollment data",
    "financial audit report", "facilities maintenance request", "IT support ticket",
    "curriculum committee notes", "parking permit application", "building access form",
    "conference registration", "library acquisition list", "departmental budget",
    "hiring committee report", "tenure review documents", "annual performance review",
    "room reservation form", "supply order catalog", "safety inspection checklist",
    "accreditation self-study", "strategic plan 2025", "alumni newsletter draft",
]

# Handwriting-like correction texts
CORRECTION_TEXTS = [
    "Please review and approve by Friday",
    "Budget approved - see attached memo",
    "Meeting rescheduled to 3:00 PM",
    "Forward to department chair",
    "Revised per committee feedback",
    "Signature required on page 3",
    "See handwritten notes in margin",
    "Updated figures for Q3",
    "Action items from today's meeting",
    "Draft - do not distribute",
    "Confidential - internal use only",
    "Approved with minor revisions",
    "Return to sender - wrong department",
    "File under grants and contracts",
    "Urgent - deadline approaching",
]


def generate_synthetic_page() -> bytes:
    """Create a synthetic document page image with text."""
    width, height = 850, 1100  # ~letter size at 100 DPI
    img = Image.new("RGB", (width, height), color=(252, 251, 248))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((50, 40), fake.company(), fill=(30, 30, 30))
    draw.text((50, 70), fake.address().replace("\n", ", "), fill=(80, 80, 80))
    draw.line([(50, 100), (800, 100)], fill=(150, 150, 150), width=2)

    # Body text
    y = 130
    for _ in range(random.randint(8, 20)):
        line = fake.sentence(nb_words=random.randint(6, 14))
        draw.text((50, y), line, fill=(40, 40, 40))
        y += 22
        if y > 900:
            break

    # Handwritten scribbles
    for _ in range(random.randint(1, 3)):
        x_start = random.randint(50, 400)
        y_start = random.randint(200, 800)
        points = [(x_start + i * 8, y_start + random.randint(-3, 3)) for i in range(random.randint(10, 30))]
        draw.line(points, fill=(20, 20, 150), width=2)

    # Margin note
    if random.random() < 0.4:
        note_y = random.randint(200, 700)
        draw.text((650, note_y), random.choice(CORRECTION_TEXTS)[:20], fill=(180, 30, 30))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Pre-generate 10 stable user UUIDs
_USER_IDS = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user-{i}")) for i in range(1, 11)]


def generate_user_id() -> str:
    """Pick from a consistent set of ~10 synthetic users."""
    return random.choice(_USER_IDS)


# ── Traffic generation ─────────────────────────

class TrafficGenerator:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
        self.uploaded_docs = []    # (doc_id, region_id)
        self.search_sessions = []  # (session_id, result_doc_ids)
        self.stats = {"uploads": 0, "corrections": 0, "searches": 0, "feedback": 0, "errors": 0}

    def do_upload(self):
        """Upload a synthetic document."""
        img_bytes = generate_synthetic_page()
        filename = f"{fake.file_name(extension='png')}"

        try:
            resp = self.client.post(
                f"{self.api_url}/api/upload",
                files={"file": (filename, img_bytes, "image/png")},
                data={"source": "synthetic", "page_count": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            self.uploaded_docs.append((data["document_id"], data["region_id"]))
            self.stats["uploads"] += 1
            log.info(f"[UPLOAD] doc={data['document_id'][:8]}... file={filename}")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[UPLOAD ERROR] {e}")

    def do_correction(self):
        """Submit a correction on a random uploaded document's region."""
        if not self.uploaded_docs:
            return

        doc_id, region_id = random.choice(self.uploaded_docs)
        text = random.choice(CORRECTION_TEXTS)
        user_id = generate_user_id()

        try:
            resp = self.client.post(
                f"{self.api_url}/api/corrections",
                json={
                    "region_id": region_id,
                    "corrected_text": text,
                    "user_id": user_id,
                    "opted_in": random.random() > 0.1,  # 90% opt in
                },
            )
            resp.raise_for_status()
            self.stats["corrections"] += 1
            log.info(f"[CORRECTION] region={region_id[:8]}... text='{text[:30]}...'")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[CORRECTION ERROR] {e}")

    def do_search(self):
        """Execute a search query."""
        query = random.choice(SEARCH_QUERIES)
        user_id = generate_user_id()

        try:
            resp = self.client.post(
                f"{self.api_url}/api/search",
                json={
                    "query_text": query,
                    "user_id": user_id,
                    "is_test_account": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data["result_doc_ids"]:
                self.search_sessions.append((data["session_id"], data["result_doc_ids"]))
            self.stats["searches"] += 1
            log.info(f"[SEARCH] '{query}' → {len(data['result_doc_ids'])} results")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[SEARCH ERROR] {e}")

    def do_feedback(self):
        """Submit feedback on a search result."""
        if not self.search_sessions:
            return

        session_id, result_doc_ids = random.choice(self.search_sessions)
        doc_id = random.choice(result_doc_ids)
        feedback_type = random.choices(
            ["click", "thumbs_up", "thumbs_down"],
            weights=[0.6, 0.25, 0.15],
        )[0]

        try:
            resp = self.client.post(
                f"{self.api_url}/api/feedback",
                json={
                    "session_id": session_id,
                    "document_id": doc_id,
                    "feedback_type": feedback_type,
                },
            )
            resp.raise_for_status()
            self.stats["feedback"] += 1
            log.info(f"[FEEDBACK] {feedback_type} on doc={doc_id[:8]}...")
        except Exception as e:
            self.stats["errors"] += 1
            log.error(f"[FEEDBACK ERROR] {e}")

    def run(self, rate: float, duration: int):
        """
        Run the traffic generator.

        rate: average events per second
        duration: total runtime in seconds

        Traffic pattern:
        - First 20% of time: mostly uploads (building up documents)
        - Middle 60%: mixed traffic
        - Last 20%: more searches and feedback
        """
        log.info(f"Starting traffic generator: rate={rate}/s, duration={duration}s")
        log.info(f"Target API: {self.api_url}")

        start = time.time()
        elapsed = 0

        while elapsed < duration:
            progress = elapsed / duration

            if progress < 0.2:
                weights = [0.7, 0.1, 0.15, 0.05]
            elif progress < 0.8:
                weights = [0.25, 0.2, 0.35, 0.2]
            else:
                weights = [0.1, 0.15, 0.4, 0.35]

            action = random.choices(
                [self.do_upload, self.do_correction, self.do_search, self.do_feedback],
                weights=weights,
            )[0]

            action()

            sleep_time = max(0.1, (1.0 / rate) + random.uniform(-0.2, 0.2))
            time.sleep(sleep_time)
            elapsed = time.time() - start

        log.info(f"Generator finished. Stats: {json.dumps(self.stats)}")
        log.info(
            f"  Uploaded {self.stats['uploads']} docs, "
            f"{self.stats['corrections']} corrections, "
            f"{self.stats['searches']} searches, "
            f"{self.stats['feedback']} feedback events, "
            f"{self.stats['errors']} errors"
        )


def main():
    parser = argparse.ArgumentParser(description="Paperless data traffic generator")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Stub API base URL")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Average events per second")
    parser.add_argument("--duration", type=int, default=300,
                        help="Total runtime in seconds")
    args = parser.parse_args()

    gen = TrafficGenerator(args.api_url)
    gen.run(args.rate, args.duration)


if __name__ == "__main__":
    main()
