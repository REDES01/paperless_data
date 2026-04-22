"""
Generate OOD samples for the drift dashboard demo.

Produces N typed-text strips sized like IAM line crops (64 tall, up to 512
wide). Typed text is OOD vs IAM cursive in every way the MMD detector
cares about (stroke thickness, character spacing, edge statistics).

Writes PNG files to ./ood_samples/ locally. Upload them to MinIO yourself
with `mc cp ood_samples/*.png local/paperless-images/ood/` or use the
`upload_ood_to_minio.py` companion.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from faker import Faker

N = int(os.environ.get("N_SAMPLES", "10"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "ood_samples"))
HEIGHT = 64
WIDTH = 512

fake = Faker()
Faker.seed(42)
random.seed(42)


def _load_font():
    # Try a few common fonts; fall back to PIL default.
    candidates = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, 32)
    return ImageFont.load_default()


def make_sample(i: int, font) -> Image.Image:
    img = Image.new("L", (WIDTH, HEIGHT), color=245)
    draw = ImageDraw.Draw(img)
    text = fake.sentence(nb_words=random.randint(4, 8))[:60]
    draw.text((10, 15), text, fill=15, font=font)
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = _load_font()
    for i in range(N):
        img = make_sample(i, font)
        path = OUT_DIR / f"ood_{i:02d}.png"
        img.save(path, format="PNG")
        print(f"  wrote {path}")
    print(f"done: {N} OOD samples in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
