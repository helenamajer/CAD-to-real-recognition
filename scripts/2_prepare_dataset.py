import os
import shutil
import random
import json
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

RAW         = Path("/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/data/raw")
OUT         = Path("/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/data/yolo_dataset")
BACKGROUNDS = Path("/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/backgrounds")

# ── Validate background images folder ────────────────────────────────────────
bg_paths = list(BACKGROUNDS.glob("*.jpg")) + list(BACKGROUNDS.glob("*.png"))
if not bg_paths:
    raise FileNotFoundError(
        f"No background images found in '{BACKGROUNDS.resolve()}'. "
        "Add .jpg or .png images to the backgrounds/ folder."
    )
print(f"Found {len(bg_paths)} background images.")

# ── Create YOLO dataset folder structure ──────────────────────────────────────
for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ── Compositing with random position + scale ──────────────────────────────────
def composite_onto_background(render_path, label_path, out_img_path, out_label_path):
    """
    Paste the render onto a random background at a random scale and position.
    Updates annotation coordinates to match new position and scale.
    Forces all class IDs to 0 — identification is handled by DINOv2 + FAISS.
    Returns False if compositing failed.
    """
    part = Image.open(render_path).convert("RGBA")
    orig_w, orig_h = part.size

    # pick a random background
    bg = Image.open(random.choice(bg_paths)).convert("RGB")
    bg = bg.resize((orig_w, orig_h))
    bg_w, bg_h = bg.size

    # ── Random scale ──────────────────────────────────────────────────────────
    scale = random.uniform(0.9, 1.0)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    part  = part.resize((new_w, new_h), Image.LANCZOS)

    # ── Random position ───────────────────────────────────────────────────────
    # ensure part stays fully within the frame
    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    # ── Composite ─────────────────────────────────────────────────────────────
    bg.paste(part, (pos_x, pos_y), part)
    bg.convert("RGB").save(out_img_path)

    # ── Update annotation coordinates + force class_id to 0 ──────────────────
    # original annotation was generated at orig_w x orig_h with part centred
    # transform coordinates to match new scale and position
    # class_id is forced to 0 because YOLO only needs to detect "part" vs background
    # the specific part identity is handled downstream by DINOv2 + FAISS
    lines     = open(label_path).read().strip().splitlines()
    new_lines = []

    for line in lines:
        parts = line.strip().split()

        # need at least class_id + 3 polygon points (7 values minimum)
        if len(parts) < 7:
            continue

        coords = [float(v) for v in parts[1:]]
        new_coords = []

        for i in range(0, len(coords), 2):
            x_norm = coords[i]
            y_norm = coords[i + 1]

            # convert to pixel in original render space
            x_px   = x_norm * orig_w
            y_px   = y_norm * orig_h

            # apply scale and translation offset
            x_new  = x_px * scale + pos_x
            y_new  = y_px * scale + pos_y

            # normalize to background dimensions and clamp to 0-1
            x_new_norm = max(0.0, min(1.0, x_new / bg_w))
            y_new_norm = max(0.0, min(1.0, y_new / bg_h))

            new_coords.extend([x_new_norm, y_new_norm])

        if not new_coords:
            continue

        coord_str = " ".join(f"{v:.6f}" for v in new_coords)

        # force class_id to 0 — all parts are one class: "part"
        new_lines.append(f"0 {coord_str}")

    if not new_lines:
        return False

    with open(out_label_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    return True

# ── One pass over every part folder in data/raw/ ──────────────────────────────
for class_id, part_folder in enumerate(sorted(RAW.iterdir())):
    if not part_folder.is_dir():
        continue

    # skip empty folders
    if not any(part_folder.iterdir()):
        print(f"[WARNING] {part_folder.name} is empty, skipping.")
        continue

    # skip if metadata.json is missing
    if not (part_folder / "metadata.json").exists():
        print(f"[WARNING] No metadata.json in {part_folder.name}, skipping.")
        continue

    meta   = json.load(open(part_folder / "metadata.json"))
    images = sorted(part_folder.glob("*.png"))
    random.shuffle(images)

    # 80% train, 20% val
    split_idx  = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    print(f"{meta['name']}: {len(train_imgs)} train, {len(val_imgs)} val")

    skipped = 0
    for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in img_list:
            label_path = img_path.with_suffix(".txt")

            if not label_path.exists():
                print(f"  [WARNING] No annotation for {img_path.name}, skipping.")
                skipped += 1
                continue

            out_img   = OUT / "images" / split / img_path.name
            out_label = OUT / "labels" / split / label_path.name

            success = composite_onto_background(
                img_path, label_path, out_img, out_label)

            if not success:
                print(f"  [WARNING] Compositing failed for {img_path.name}, skipping.")
                skipped += 1

    if skipped:
        print(f"  {skipped} images skipped for {meta['name']}")

# ── Write dataset.yaml ────────────────────────────────────────────────────────
# single class: "part" — YOLO only needs to detect that a part exists
# identification of which specific part is handled by DINOv2 + FAISS
yaml_data = {
    "path":  str(OUT.resolve()),
    "train": "images/train",
    "val":   "images/val",
    "nc":    1,
    "names": ["part"],
}

with open(OUT / "dataset.yaml", "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print(f"\ndataset.yaml written — 1 class: ['part']")
print("All annotations forced to class_id 0.")
print("Ready to train.")