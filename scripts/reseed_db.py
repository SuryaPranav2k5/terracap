"""Re-seeds GeoCaption SQLite database with genuine PyTorch model outputs."""

import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import load_model_and_vocab, run_inference_on_pil

DB_PATH = REPO_ROOT / "web" / "db" / "custom.db"
IMAGES_DIR = REPO_ROOT / "web" / "public" / "images"

SEEDS = [
    {"file": "harbor.png", "label": "Harbor"},
    {"file": "stadium.png", "label": "Stadium"},
    {"file": "agriculture.png", "label": "Farmland"},
    {"file": "industrial.png", "label": "Industrial"},
    {"file": "residential.png", "label": "Residential"},
    {"file": "overpass.png", "label": "Overpass"},
    {"file": "beach.png", "label": "Beach"},
    {"file": "airport.png", "label": "Airport"},
]


def reseed():
    print(f"Connecting to database at {DB_PATH} ...")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Clear old fake reference items
    cur.execute("DELETE FROM Generation WHERE source = 'reference'")
    conn.commit()
    print("Cleared existing reference records.")

    print("Pre-warming model for batch inference...")
    load_model_and_vocab()

    now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    for idx, item in enumerate(SEEDS):
        img_path = IMAGES_DIR / item["file"]
        if not img_path.exists():
            print(f"Warning: {img_path} not found. Skipping.")
            continue

        print(f"Generating genuine caption for {item['label']} ({item['file']}) ...")
        pil_img = Image.open(img_path).convert("RGB")
        out = run_inference_on_pil(pil_img, beam_size=3)

        gen_id = f"ref-{idx+1:02d}-{item['file'].split('.')[0]}"
        image_src = f"/images/{item['file']}"
        caption = out["caption"]
        keywords_json = json.dumps(out["keywords"])
        confidence = float(out["confidence"])
        gen_time_ms = int(out["genTimeMs"])
        tokens_json = json.dumps(out["tokens"])
        att_json = json.dumps(out["attention_steps"])

        cur.execute("""
            INSERT INTO Generation (id, imageSrc, imageName, label, caption, keywords, confidence, genTimeMs, source, createdAt, tokens, attentionSteps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reference', ?, ?, ?)
        """, (gen_id, image_src, item["file"], item["label"], caption, keywords_json, confidence, gen_time_ms, now, tokens_json, att_json))
        conn.commit()
        print(f"  -> '{caption}' (conf: {confidence}, time: {gen_time_ms}ms, {len(out['tokens'])} tokens)")

    conn.close()
    print("Database re-seeding completed successfully!")


if __name__ == "__main__":
    reseed()

