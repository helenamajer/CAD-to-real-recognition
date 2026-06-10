import torch
import faiss
import json
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

RAW        = Path("data/raw")
EMBEDDINGS = Path("embeddings")
EMBEDDINGS.mkdir(exist_ok=True)

INDEX_PATH    = EMBEDDINGS / "index.faiss"
META_PATH     = EMBEDDINGS / "metadata.json"
PROCESSED_PATH = EMBEDDINGS / "processed.json"  # tracks which renders are already indexed

# ── Load existing index and metadata if they exist ────────────────────────────
# this allows incremental updates — only new renders are processed
if INDEX_PATH.exists() and META_PATH.exists() and PROCESSED_PATH.exists():
    print("Existing index found — loading...")
    index    = faiss.read_index(str(INDEX_PATH))
    metadata = json.load(open(META_PATH))
    processed = set(json.load(open(PROCESSED_PATH)))  # set of already-indexed render paths
    print(f"  {len(metadata)} renders already indexed.")
    print(f"  Only new renders will be processed.\n")
else:
    print("No existing index found — building from scratch.\n")
    index     = faiss.IndexFlatL2(768)  # 768 = DINOv2-base embedding dimension
    metadata  = []
    processed = set()

# ── Load DINOv2 ───────────────────────────────────────────────────────────────
print("Loading DINOv2...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()
print("DINOv2 loaded.\n")

new_embeddings = []
new_metadata   = []
new_processed  = []
skipped        = 0

# ── Loop over every part folder ───────────────────────────────────────────────
for part_folder in sorted(RAW.iterdir()):
    if not part_folder.is_dir():
        continue

    if not any(part_folder.iterdir()):
        print(f"[WARNING] {part_folder.name} is empty, skipping.")
        continue

    meta_path = part_folder / "metadata.json"
    if not meta_path.exists():
        print(f"[WARNING] No metadata.json in {part_folder.name}, skipping.")
        continue

    meta   = json.load(open(meta_path))
    images = sorted(part_folder.glob("*.png"))

    if not images:
        print(f"[WARNING] No images in {part_folder.name}, skipping.")
        continue

    # find renders not yet in the index
    new_images = [p for p in images if str(p) not in processed]

    if not new_images:
        print(f"  {meta['name']} — already fully indexed, skipping.")
        continue

    print(f"Indexing {meta['name']} — {len(new_images)} new renders "
          f"({len(images) - len(new_images)} already indexed)...")

    for img_path in new_images:
        try:
            image  = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                output    = model(**inputs)
                embedding = output.last_hidden_state.mean(dim=1).numpy()

            new_embeddings.append(embedding)
            new_metadata.append({
                "part_name":   meta["name"],
                "part_number": meta["part_number"],
                "render_path": str(img_path),
            })
            new_processed.append(str(img_path))

        except Exception as e:
            print(f"  [WARNING] Could not process {img_path.name}: {e}")
            skipped += 1

    print(f"  done.")

# ── Append new embeddings to index ────────────────────────────────────────────
if new_embeddings:
    print(f"\nAdding {len(new_embeddings)} new embeddings to index...")
    index.add(np.vstack(new_embeddings))

    # update metadata and processed list
    metadata.extend(new_metadata)
    processed.update(new_processed)

    # save everything
    faiss.write_index(index, str(INDEX_PATH))
    json.dump(metadata,         open(META_PATH,       "w"), indent=4)
    json.dump(list(processed),  open(PROCESSED_PATH,  "w"), indent=4)

    print(f"\nIndex updated and saved to embeddings/")
    print(f"  New renders added     : {len(new_embeddings)}")
    print(f"  Total renders indexed : {len(metadata)}")
    print(f"  Total renders skipped : {skipped}")
    print(f"  Parts in index        : {len(set(m['part_name'] for m in metadata))}")
else:
    print("\nNo new renders found — index is already up to date.")

print("\nReady to run pipeline.")