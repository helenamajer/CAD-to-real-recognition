from pathlib import Path

IMAGES = Path("data/yolo_dataset/images/train")
LABELS = Path("data/yolo_dataset/labels/train")

bad = [
    "Bracket-53141411_0105",
    "Lamp-Bracket-54563962_0012",
    "Bracket-53141411_0168",
]

for name in bad:
    for folder, ext in [(LABELS, ".txt"), (IMAGES, ".png")]:
        f = folder / (name + ext)
        if f.exists():
            f.unlink()
            print(f"Deleted: {f.name}")