"""Uses blender .OBJ files to generate a synthetic dataset."""

import json
import math
from os import environ, path, remove
from pathlib import Path
import random
from typing import cast
from blender.events import Events
from blender.render_settings import RenderSettings
import bpy
import cv2
from dotenv import load_dotenv
from mathutils import Vector


def load_env_vars():
    """Loads a .env (if there is one) from the pipeline directory, before
    checking if the necessary environment variables are set."""
    load_dotenv()
    num_renders = int(environ.get("MODEL_DATA_NUM_RENDERS", 200))
    root_dir = environ.get("MODEL_DATA_DIR")
    if root_dir is None:
        raise FileNotFoundError("`MODEL_DATA_DIR` is not set.")
    root_dir = Path(root_dir)
    return num_renders, root_dir / "obj", root_dir / "raw"


NUM_RENDERS, OBJ_DIR, RAW_DIR = load_env_vars()

OBJECT_FRAME_PADDING_MIN = 1.3  # INFLUENCES THE CAMERA'S DISTANCE
OBJECT_FRAME_PADDING_MAX = 1.8  # INFLUENCES THE CAMERA'S DISTANCE


def ensure_dirs():
    """Ensures the DXF and OBJ directories are on the system."""
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


@staticmethod
def __generate_random_location(min_radius: float, max_radius: float):
    """Generates a random location."""
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(10), math.radians(89))
    r = random.uniform(min_radius, max_radius)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return Vector((x, y, z))


@staticmethod
def __mask_to_yolo_polygon(
    class_id: int, input_file: Path, resolution_x: int, resolution_y: int
):
    """Converts a binary mask PNG to a normalized YOLO segmentation polygon.
    TODO: REFACTOR/cleanup"""
    mask = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Threshold to pure black/white - pixels > 127 becomes 255, rest become 0
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # RETR_TREE captures the full contour hierarchy
    # Outer boundary of the part AND all interior holes
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    segments = []

    for _, contour in enumerate(contours):
        # Skip tiny contours - rendering artifacts and noise
        # 100px area theshold filters specs without removing real bolt holes
        if cv2.contourArea(contour) < 100:
            continue

            # Simplify polygon — 0.002 = 0.2% of perimeter tolerance
        # Keeps enough points for accuracy without thousands of coordinates
        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Normalize coordinates to 0-1 range relative to image dimensions
        # YOLO requires all coordinates normalized between 0 and 1
        points = simplified.reshape(-1, 2)
        normalized = [(x / resolution_x, y / resolution_y) for x, y in points]
        flat = [val for point in normalized for val in point]

        segments.append(" ".join(f"{v:.6f}" for v in flat))

    if not segments:
        return None

    # YOLO segmentation format with multiple polygons:
    # Class_id [outer boundary coords] [hole 1 coords] [hole 2 coords] etc...
    # YOLO correctly interprets multiple polygon segments as one annotation
    return f"{class_id} " + " ".join(segments)


def handle_data_generation(
    input_file: Path, output_dir: Path, class_id: int, part_name: str, part_number: str
):
    """Generates synthetic data by using the .OBJ file."""
    # bpy.ops.wm.obj_import(filepath=str(input_file))
    Events.Object.import_native_object(input_file)

    obj = cast(bpy.types.Object, Events.Object.get_active_object())

    bbox_size = Events.Object.get_bbox_size(obj)
    min_radius = bbox_size * OBJECT_FRAME_PADDING_MIN
    max_radius = bbox_size * OBJECT_FRAME_PADDING_MAX
    print(f"Object bbox size: {bbox_size:.3f}")
    print(f"Camera distance range: {min_radius:.3f} — {max_radius:.3f}")

    part_material = Events.Material.get_principled_bsdf(obj)

    render_settings = RenderSettings()
    Events.Scene.set_render_settings(render_settings)

    cam = Events.Object.get_object(name="Camera")
    assert cam

    main_light = Events.Object.get_object(name="MainLight")
    assert main_light

    fill_light = Events.Object.get_object(name="FillLight")
    assert fill_light

    mask_material = Events.Material.get_segmentation_mask_material()
    original_materials = Events.Material.get_object_materials(obj)
    for frame_idx in range(1, NUM_RENDERS + 1):
        # TODO: Relocate grouped logic
        # TODO: Move expensive logic outside the loop
        # TODO: Create functions for exporting files
        # TODO: Keep image buffer in memory instead of exporting and then reimporting?
        Events.Object.set_object_location(
            cam, __generate_random_location(min_radius, max_radius)
        )
        Events.Scene.point_camera(cam, obj)

        main_light_loc = Vector(
            (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(2, 8))
        )
        main_light_energy = random.uniform(300, 1500)
        main_light_color = Vector(
            (
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0),
                random.uniform(0.8, 1.0),
            )
        )
        Events.Object.set_object_location(main_light, main_light_loc)
        Events.Object.set_light_energy(main_light, main_light_energy)
        Events.Object.set_light_color(main_light, main_light_color)

        fill_light_loc = Vector(
            (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(1, 4))
        )
        fill_light_energy = random.uniform(50, 300)
        Events.Object.set_object_location(fill_light, fill_light_loc)
        Events.Object.set_light_energy(fill_light, fill_light_energy)

        part_material.inputs["Base Color"].default_value = (
            random.uniform(0.0, 1.0),   # R
            random.uniform(0.0, 1.0),   # G
            random.uniform(0.0, 1.0),   # B
            1.0 # Alpha
        )

        part_material.inputs["Roughness"].default_value = random.uniform(0.1, 0.9)  # type: ignore
        part_material.inputs["Metallic"].default_value = random.uniform(0.0, 1.0)  # type: ignore

        frame_as_str = f"{frame_idx:04d}"
        file_prefix: str = f"{input_file.with_suffix("").name}_{frame_as_str}"

        rgb_output_path = (output_dir / file_prefix).with_suffix(".png")
        Events.Scene.set_render_filepath(str(rgb_output_path))
        Events.Scene.set_cycle_sample_count(64)

        Events.Scene.write_render()

        for slot in obj.material_slots:
            slot.material = mask_material

        mask_output_path = (output_dir / f"{file_prefix}_mask").with_suffix(".png")
        Events.Scene.set_render_filepath(str(mask_output_path))
        Events.Scene.set_cycle_sample_count(1)

        Events.Scene.write_render()

        # Swap to white material
        for slot, material in zip(obj.material_slots, original_materials):
            slot.material = material

        yolo_line = __mask_to_yolo_polygon(
            class_id,
            mask_output_path,
            render_settings.image.resolution_x,
            render_settings.image.resolution_y,
        )

        txt_output_path = (output_dir / file_prefix).with_suffix(".txt")
        if yolo_line:
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(yolo_line + "\n")
        else:
            print(
                f"[WARNING] No contour found for frame {frame_idx} - skipping annotation"
            )
        if path.exists(mask_output_path):
            remove(mask_output_path)
        print(f"[{frame_idx}/{NUM_RENDERS}] {file_prefix}.txt done")
    metadata = {"name": part_name, "part_number": part_number, "class_id": class_id}
    metadata_output_path = (output_dir / "metadata").with_suffix(".json")
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(
        f"Dataset generation complete - {NUM_RENDERS} renders written to {output_dir}"
    )


def main():
    """Runs the script.
    TODO: Find a way to make class_id dynamically assigned.
    Iteration will not work if looking to generate one object on its own.
    """
    ensure_dirs()
    files = list(OBJ_DIR.glob("*.obj"))
    if not files:
        return
    first_time = False
    class_id = 0
    for file in files:
        if first_time is False:
            Events.Scene.clear_scene(remove_light_and_camera=True)
            Events.Scene.add_camera("Camera")
            Events.Scene.add_light("MainLight")
            # Weaker, simulaties ambience
            Events.Scene.add_light("FillLight")
            first_time = True
        else:
            Events.Scene.clear_scene(remove_light_and_camera=False)
        part_name, part_number = file.with_suffix("").name.rsplit("-", 1)
        output_dir = RAW_DIR / file.with_suffix("").name
        handle_data_generation(file, output_dir, class_id, part_name, part_number)
        class_id += 1


if __name__ == "__main__":
    main()
