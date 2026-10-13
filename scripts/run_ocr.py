"""
Step 1 of the HUD-mask project: run OCR (text detection + recognition)
on a folder of game frames and save what it finds.

Usage (from the project root, with the conda env active):
    python scripts/run_ocr.py

Input:  data/frames/*.png (or .jpg)
Output: outputs/ocr/images/  -> frames with detected text boxes drawn on them
        outputs/ocr/json/    -> raw results per frame (boxes, text, confidence)
        outputs/ocr/summary.csv -> one row per detected text region
"""

import csv
import os
import time
from pathlib import Path

# Skip the "checking connectivity to model hosters" step on every run.
# Must be set before paddleocr is imported.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

# --- Paths -------------------------------------------------------------------
# Path(__file__) is this script; .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "outputs" / "ocr"
IMG_OUT = OUT_DIR / "images"
JSON_OUT = OUT_DIR / "json"

# Only keep text the model is reasonably sure about (0.0 - 1.0).
MIN_CONFIDENCE = 0.5


def main():
    # Make output folders if they don't exist yet.
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        p for p in FRAMES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        print(f"No frames found in {FRAMES_DIR}. Run scripts/extract_frames.py first.")
        return
    print(f"Found {len(frames)} frames.")

    # Create the OCR pipeline. The first run downloads the models (~100 MB).
    # We switch off the document-specific extras: game frames are not
    # rotated or warped scans, and the extras only cost time.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # Turn off oneDNN (Intel CPU acceleration). Recent PaddlePaddle CPU
        # builds on Windows crash with "ConvertPirAttribute2RuntimeAttribute
        # not support" when it is on. Slightly slower, but it works.
        enable_mkldnn=False,
    )

    rows = []
    total_time = 0.0

    for i, frame_path in enumerate(frames, start=1):
        start = time.perf_counter()
        results = ocr.predict(str(frame_path))
        elapsed = time.perf_counter() - start
        total_time += elapsed

        for res in results:
            # Save a picture with boxes drawn, plus the raw JSON.
            res.save_to_img(str(IMG_OUT))
            res.save_to_json(str(JSON_OUT))

            data = res.json.get("res", res.json)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("rec_polys", [])

            for text, score, poly in zip(texts, scores, polys):
                if score < MIN_CONFIDENCE:
                    continue
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                rows.append({
                    "frame": frame_path.name,
                    "text": text,
                    "confidence": round(float(score), 3),
                    "x_min": int(min(xs)), "y_min": int(min(ys)),
                    "x_max": int(max(xs)), "y_max": int(max(ys)),
                })

        print(f"[{i}/{len(frames)}] {frame_path.name}: {elapsed * 1000:.0f} ms")

    # Write everything into one table you can open in Excel.
    summary_path = OUT_DIR / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame", "text", "confidence", "x_min", "y_min", "x_max", "y_max"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Done. {len(rows)} text regions above confidence {MIN_CONFIDENCE}.")
    print(f"Average time per frame: {total_time / len(frames) * 1000:.0f} ms")
    print(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
