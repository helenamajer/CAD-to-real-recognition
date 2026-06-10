import os
import json
import faiss
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import tempfile

# ── Config ────────────────────────────────────────────────────────────────────
# update these paths if your folder structure is different
YOLO_WEIGHTS = "runs/segment/models/yolov8_parts_singleclass/weights/best.pt"
INDEX_PATH   = "embeddings/index.faiss"
META_PATH    = "embeddings/metadata.json"

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models...")

yolo      = YOLO(YOLO_WEIGHTS)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino      = AutoModel.from_pretrained("facebook/dinov2-base")
dino.eval()
index     = faiss.read_index(INDEX_PATH)
metadata  = json.load(open(META_PATH))

print("Models loaded. Server ready.\n")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/identify", methods=["POST"])
def identify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        print(f"Image received: {os.path.getsize(tmp_path)} bytes")

        # ── Step 1: YOLO ──────────────────────────────────────────────────────
        results = yolo(tmp_path, conf=0.05)
        result  = results[0]
        print(f"Detections: {len(result.boxes)}")

        if len(result.boxes) == 0:
            print("No detection.")
            return jsonify({"detected": False})

        best_idx = result.boxes.conf.argmax().item()
        box      = result.boxes[best_idx].xyxy[0].tolist()
        conf     = result.boxes[best_idx].conf.item()
        print(f"Confidence: {conf:.1%}")

        # ── Step 2: Crop ──────────────────────────────────────────────────────
        img  = Image.open(tmp_path).convert("RGB")
        w, h = img.size
        pad  = 20
        crop = img.crop((
            max(0, int(box[0]) - pad),
            max(0, int(box[1]) - pad),
            min(w, int(box[2]) + pad),
            min(h, int(box[3]) + pad),
        ))

        # ── Step 3: DINOv2 ────────────────────────────────────────────────────
        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            embedding = dino(**inputs).last_hidden_state.mean(dim=1).numpy()

        # ── Step 4: FAISS ─────────────────────────────────────────────────────
        distances, indices = index.search(embedding, k=1)
        match = metadata[indices[0][0]]
        print(f"Match: {match['part_name']} — {match['part_number']}")

        return jsonify({
            "detected":    True,
            "part_name":   match["part_name"],
            "part_number": match["part_number"],
            "confidence":  round(conf * 100, 1),
            "render_path": match["render_path"],
        })

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.route("/render")
def render_image():
    path = request.args.get("path")
    if not path or not Path(path).exists():
        print(f"[ERROR] Render not found: {path}")
        return "Image not found", 404
    return send_file(path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)