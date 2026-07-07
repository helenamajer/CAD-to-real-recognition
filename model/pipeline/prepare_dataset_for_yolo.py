"""Prepare the dataset for training YOLOv8.
- split dataset into training/validation batch using 80/20 ratio.
- paste the component renders onto background images.
- change class ID for all component classes to 0 so all components belong to one class.
- generate a .yaml file for paths, class, and name.
"""

import os
import json
import yaml
import random
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv


def load_env_vars() -> tuple[Path, Path, Path]:
    """Load environment variables and return key dataset paths."""
    load_dotenv()

    def require(name: str) -> Path:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"{name} environment variable is required.")
        return Path(value).expanduser().resolve()

    raw = require("RAW_DATA_DIR")
    out = require("OUTPUT_DATA_DIR")
    backgrounds = require("BACKGROUNDS_DIR")

    return raw, out, backgrounds


def validate_background_images_folder(backgrounds: Path) -> list[Path]:
    """Ensure background folder contains images with valid formats."""
    images = list(backgrounds.glob("*.jpg")) + list(backgrounds.glob("*.png"))

    if not images:
        raise FileNotFoundError(f"No .jpg or .png images found in {backgrounds}")

    print(f"Found {len(images)} background images.")
    return images


def create_yolo_dataset_file_structure(out: Path) -> None:
    """Create YOLO folder structure."""
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> Image.Image:
    """Open an image"""
    return Image.open(path).convert("RGBA")


def choose_background(bg_paths: list[Path]) -> Image.Image:
    """Select random image of background images pool"""
    return Image.open(random.choice(bg_paths)).convert("RGB")


def apply_random_scale_of_part_render(part: Image.Image, scale_range=(0.9, 1.0)):
    """
    Randomly scales a component render image.

    Returns:
        - resized image
        - scale factor used (for label adjustment later)
    """
    scale = random.uniform(*scale_range)
    new_size = (
        int(part.width * scale),
        int(part.height * scale),
    )
    return part.resize(new_size, Image.LANCZOS), scale


def random_position_of_part_render(bg_size, part_size):
    """
    Generates a random (x, y) position where the part
    fully fits inside the background image.

    Ensures the component does not go out of bounds.
    """
    bg_w, bg_h = bg_size
    part_w, part_h = part_size

    max_x = max(0, bg_w - part_w)
    max_y = max(0, bg_h - part_h)

    return random.randint(0, max_x), random.randint(0, max_y)


def transform_labels(label_path: Path, orig_size, scale, offset, bg_size) -> list[str]:
    """
    Convert original annotation coordinates to new coordinates after compositing with random scale and position.
    Forces class_id = 0.
    """
    orig_w, orig_h = orig_size  # size of original render
    bg_w, bg_h = bg_size  # size of backgrounf image
    pos_x, pos_y = offset  # top-left corner x,y coordinate of the render image

    # read original annotation file (YOLO-style polygon format)
    lines = label_path.read_text().strip().splitlines()
    new_lines = []

    # the annotation must be a minimum of 7 values
    # 1 class number, and at least 3 x,y points (6 values)
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        # convert all coordinate values (skip class_id)
        coords = list(map(float, parts[1:]))
        new_coords = []

        # process coordinates as (x, y) pairs
        for i in range(0, len(coords), 2):

            # convert normalized coords (0–1) → pixel coords in original image
            x = coords[i] * orig_w
            y = coords[i + 1] * orig_h

            # apply scaling from random resize
            x = x * scale + pos_x
            y = y * scale + pos_y

            x = max(0.0, min(1.0, x / bg_w))
            y = max(0.0, min(1.0, y / bg_h))

            new_coords.extend([x, y])

        # skip if something went wrong and no valid coords were produced
        if new_coords:
            coord_str = " ".join(f"{v:.6f}" for v in new_coords)
            new_lines.append(f"0 {coord_str}")

    return new_lines


def composite_render(
    render_path: Path,
    label_path: Path,
    out_img_path: Path,
    out_label_path: Path,
    bg_paths: list[Path],
) -> bool:
    """Paste render onto random background and update labels."""

    part = load_image(render_path)
    orig_size = part.size

    bg = choose_background(bg_paths)
    bg_size = bg.size

    part, scale = apply_random_scale_of_part_render(part)
    pos = random_position_of_part_render(bg_size, part.size)

    bg.paste(part, pos, part)
    bg.convert("RGB").save(out_img_path)

    new_labels = transform_labels(
        label_path,
        orig_size,
        scale,
        pos,
        bg_size,
    )

    if not new_labels:
        return False

    out_label_path.write_text("\n".join(new_labels) + "\n")
    return True


def generate_yaml(out: Path) -> None:
    """Create a .yaml file with paths to the trian/val split, class number and name"""
    yaml_data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["part"],
    }

    with open(out / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


def main():
    """Run the dataset generation pipeline"""
    raw, out, backgrounds = load_env_vars()

    # converts background folder into a list of usable image paths
    bg_paths = validate_background_images_folder(backgrounds)

    # creates:
    #   out/images/train
    #   out/images/val
    #   out/labels/train
    #   out/labels/val
    create_yolo_dataset_file_structure(out)

    # iterate over each part category folde
    for part_folder in sorted(raw.iterdir()):
        if not part_folder.is_dir():
            continue

        # require metadata.json to ensure dataset is valid
        meta_file = part_folder / "metadata.json"
        if not meta_file.exists():
            print(f"Skipping {part_folder.name}: no metadata.json")
            continue

        # load all component render images for this part
        images = list(part_folder.glob("*.png"))
        if not images:
            continue

        # shuffle to randomize train/val split
        random.shuffle(images)

        # split dataset into 80% train / 20% validation
        split = int(len(images) * 0.8)
        train, val = images[:split], images[split:]

        print(f"{part_folder.name}: {len(train)} train, {len(val)} val")

        # process both train and validation splits
        for split_name, dataset in [("train", train), ("val", val)]:
            for img_path in dataset:
                # corresponding YOLO annotations text file
                label_path = img_path.with_suffix(".txt")

                if not label_path.exists():
                    continue

                # define output paths for image + label
                out_img = out / "images" / split_name / img_path.name
                out_label = out / "labels" / split_name / label_path.name

                # composite render onto random background and update labels
                ok = composite_render(
                    img_path,
                    label_path,
                    out_img,
                    out_label,
                    bg_paths,
                )

                if not ok:
                    print(f"Failed: {img_path.name}")
    # generate YOLO dataset configuration file
    generate_yaml(out)

    print("\nDataset ready.")


if __name__ == "__main__":
    main()
