import torch
import faiss
import json
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
import argparse
import sys

YOLO_WEIGHTS = "../Instance_Recognition_App/Instance-Recognition/runs/segment/models/yolov8_parts_singleclass/weights/best.pt"
INDEX_PATH   = "embeddings/index.faiss"
META_PATH    = "embeddings/metadata.json"

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True,
    help="Path to test photo e.g. test_photos/photo_001.jpg")
parser.add_argument("--conf", type=float, default=0.25,
    help="YOLO detection confidence threshold (default: 0.25)")
parser.add_argument("--top_k", type=int, default=1,
    help="Number of closest matches to return from FAISS (default: 1)")
args = parser.parse_args()

# ── Validate inputs ───────────────────────────────────────────────────────────
if not Path(args.image).exists():
    print(f"[ERROR] Image not found: {args.image}")
    sys.exit(1)

if not Path(INDEX_PATH).exists():
    print("[ERROR] FAISS index not found. Run 3_build_index.py first.")
    sys.exit(1)

# ── Load everything ───────────────────────────────────────────────────────────
print("Loading models...")

yolo      = YOLO(YOLO_WEIGHTS)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino      = AutoModel.from_pretrained("facebook/dinov2-base")
dino.eval()
index     = faiss.read_index(INDEX_PATH)
metadata  = json.load(open(META_PATH))

print("Models loaded.\n")

# ── Step 1: YOLO detects and crops the part ───────────────────────────────────
# YOLO finds where the part is in the image and returns a bounding box
print(f"Running detection on {args.image}...")
results = yolo(args.image, conf=args.conf)
result  = results[0]

if len(result.boxes) == 0:
    print(f"\n[NO DETECTION] No part found at conf={args.conf}.")
    print("Try lowering --conf e.g. --conf 0.1")
    sys.exit(0)

# if multiple detections, take the highest confidence one
best_box_idx = result.boxes.conf.argmax().item()
box          = result.boxes[best_box_idx].xyxy[0].tolist()  # [x1, y1, x2, y2]
conf         = result.boxes[best_box_idx].conf.item()

print(f"Part detected — YOLO confidence: {conf:.1%}")

# ── Step 2: Crop the detected part ────────────────────────────────────────────
# crop the bounding box region from the original image
# add small padding around the crop for better DINOv2 context
img     = Image.open(args.image).convert("RGB")
w, h    = img.size
padding = 20

x1 = max(0, int(box[0]) - padding)
y1 = max(0, int(box[1]) - padding)
x2 = min(w, int(box[2]) + padding)
y2 = min(h, int(box[3]) + padding)

crop = img.crop((x1, y1, x2, y2))

# ── Step 3: DINOv2 embeds the crop ────────────────────────────────────────────
# DINOv2 converts the cropped part image into a 768-dimensional fingerprint
inputs = processor(images=crop, return_tensors="pt")
with torch.no_grad():
    output    = dino(**inputs)
    embedding = output.last_hidden_state.mean(dim=1).numpy()

# ── Step 4: FAISS finds the closest match ─────────────────────────────────────
# searches the index for the render whose fingerprint most closely matches
# the embedding of the real photo crop
distances, indices = index.search(embedding, k=args.top_k)

# ── Step 5: Print results ─────────────────────────────────────────────────────
print("\n── Result ───────────────────────────────────────")

for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    match = metadata[idx]

    # lower L2 distance = closer match
    # convert to a rough similarity score for readability
    similarity = 1 / (1 + dist)

    print(f"  Rank {rank + 1}:")
    print(f"  Part Name:   {match['part_name']}")
    print(f"  Part Number: {match['part_number']}")
    print(f"  Similarity:  {similarity:.4f}  (L2 distance: {dist:.2f})")
    print(f"  Matched render: {match['render_path']}")
    if rank < args.top_k - 1:
        print()

print("─────────────────────────────────────────────────\n")

# ── Step 6: Show matched render ───────────────────────────────────────────────
top_match = metadata[indices[0][0]]
render    = Image.open(top_match["render_path"])
render.show()