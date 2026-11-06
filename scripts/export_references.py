import sys
from pathlib import Path
import json
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer import load_model_and_vocab, run_inference_on_pil

IMAGES_DIR = REPO_ROOT / "web" / "public" / "images"

seeds = [
    ("airport.png", "Airport"),
    ("harbor.png", "Harbor"),
    ("stadium.png", "Stadium"),
    ("agriculture.png", "Farmland"),
    ("industrial.png", "Industrial"),
    ("residential.png", "Residential"),
    ("overpass.png", "Overpass"),
    ("beach.png", "Beach"),
]

load_model_and_vocab()
results = []
for idx, (filename, label) in enumerate(seeds):
    img_path = IMAGES_DIR / filename
    img = Image.open(img_path).convert("RGB")
    out = run_inference_on_pil(img, beam_size=3)
    stem = filename.split(".")[0]
    results.append({
        "id": f"ref-{idx+1:02d}-{stem}",
        "imageSrc": f"/images/{filename}",
        "imageName": filename,
        "label": label,
        "caption": out["caption"],
        "tokens": out["tokens"],
        "keywords": out["keywords"],
        "confidence": out["confidence"],
        "attentionSteps": out["attention_steps"],
        "genTimeMs": out["genTimeMs"],
        "source": "reference",
        "createdAt": "2026-10-30T12:00:00.000Z"
    })
    print(f"Processed {filename}: '{out['caption']}'")

out_path = REPO_ROOT / "web" / "src" / "lib" / "reference-data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Successfully generated genuine reference data at", out_path)

