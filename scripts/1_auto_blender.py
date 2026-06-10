# script to generate annotated dataset
import bpy
import json
import math
import numpy as np
import cv2
from mathutils import Vector
import random
import os

# Paths #

# Path to .OBJ file *removing hardcoded file paths soon*
file_path = "/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/obj/Mounting-Plate_16870479.obj"
output_path = "/Users/helenamajer/Git-Repos/Instance_Recognition_App/Instance-Recognition/data/raw/Mounting-Plate_16870479"
part_name = "Mounting-Plate_16870479"
part_number = "16870479"
# Increment this per part (0, 1, 2, and so on)
class_id = 5

# Settings
num_renders = 500 # 1-10 for testing, 500 for final dataset
 
os.makedirs(output_path, exist_ok=True)
 
# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
 
# Import OBJ
before = set(bpy.context.scene.objects)
bpy.ops.wm.obj_import(filepath=file_path)
bpy.context.view_layer.update()
 
after = set(bpy.context.scene.objects)
new_objects = list(after - before)
 
bpy.ops.object.select_all(action='DESELECT')
for o in new_objects:
    o.select_set(True)
if new_objects:
    bpy.context.view_layer.objects.active = new_objects[0]
 
# Normalize: center at origin
obj = new_objects[0]
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Bounding box 
# Auto camera distance based on object size to avoid having to edit min_radius max_radius every time you import
# Calculate the diagonal size of the objects bounding box
bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
bbox_min     = Vector((min(c.x for c in bbox_corners),
                        min(c.y for c in bbox_corners),
                        min(c.z for c in bbox_corners)))
bbox_max     = Vector((max(c.x for c in bbox_corners),
                        max(c.y for c in bbox_corners),
                        max(c.z for c in bbox_corners)))
# Diagonal size of the object
bbox_size = (bbox_max - bbox_min).length

# Camera distance = object size × padding multiplier
# Padding_min/max controls how much of the frame the object fills
padding_min = 1.3
padding_max = 1.8
min_radius = bbox_size * padding_min
max_radius = bbox_size * padding_max

print(f"Object bbox size: {bbox_size:.3f}")
print(f"Camera distance range: {min_radius:.3f} — {max_radius:.3f}")
 
# Material setup

# Create and assign material if object has none
if len(obj.material_slots) == 0:
    mat = bpy.data.materials.new(name="PartMaterial")
    obj.data.materials.append(mat)
 
# Grab material and set up Principled BSDF (shading node) from scratch
mat = obj.material_slots[0].material
# Enable node based material
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
 
principled = nodes.new("ShaderNodeBsdfPrincipled")
out_node = nodes.new("ShaderNodeOutputMaterial")
links.new(principled.outputs["BSDF"], out_node.inputs["Surface"])
 
# Store reference to Principled BSDF for randomisation in render loop
part_material = principled
 
# Camera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
 
# Main light
light_data = bpy.data.lights.new(name="MainLight", type='AREA')
light = bpy.data.objects.new(name="MainLight", object_data=light_data)
bpy.context.collection.objects.link(light)
 
# Fill light (weaker, simulates ambient)
fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_data)
bpy.context.collection.objects.link(fill_light)
 
# Render settings
scene = bpy.context.scene
# Use Cycles for ray tracing
scene.render.engine = 'CYCLES'
# Enable Metal GPU backend (macOS)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"
prefs.get_devices()
 
# Enable GPU only, disable CPU
for device in prefs.devices:
    if device.type == "METAL":
        device.use = True
        print(f"Enabled: {device.name}")
    else:
        device.use = False
        print(f"Disabled: {device.name}")
 
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.cycles.use_denoising = True
 
# Image format and resolution
scene.render.image_settings.file_format = 'PNG'
# RGBA for transparent background
scene.render.image_settings.color_mode = 'RGBA'
scene.render.resolution_x = 800
scene.render.resolution_y = 800
# Transparent background
scene.render.film_transparent = True
 
# Mask material: flat white emission, no shadin
# Strategy: render RGB normally, swap to flat white material, render mask, swap back
mask_mat = bpy.data.materials.new(name="MaskMaterial")
mask_mat.use_nodes = True
 
mask_nodes = mask_mat.node_tree.nodes
mask_links = mask_mat.node_tree.links
mask_nodes.clear()
 
# Emission shader — ignores all scene lighting, renders solid white
emission = mask_nodes.new("ShaderNodeEmission")
# Solid white
emission.inputs["Color"].default_value = (1, 1, 1, 1)
emission.inputs["Strength"].default_value = 1.0
 
mask_output = mask_nodes.new("ShaderNodeOutputMaterial")
mask_links.new(emission.outputs["Emission"], mask_output.inputs["Surface"])
 
# Store original materials to restore after mask render
original_materials = [slot.material for slot in obj.material_slots]
 
# Helpers
def look_at(cam, target):
    # Point camera towards target location
    direction = target - cam.location
    rotation_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rotation_quat.to_euler()
 
def random_camera_position():
    # Place camera at random point on a sphere around the object
    # Full 360 degree spin
    theta = random.uniform(0, 2 * math.pi)
    # Near-flat to near top-down
    phi = random.uniform(math.radians(10), math.radians(89))
    # Random distance
    r = random.uniform(min_radius, max_radius)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return Vector((x, y, z))
 
def mask_to_yolo_polygon(mask_path, img_w, img_h):
    # Convert a binary mask PNG to a normalized YOLO segmentation polygon
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
 
    # Threshold to pure black/white — pixels > 127 become 255, rest become 0
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
 
    # RETR_TREE captures the full contour hierarchy:
    # Outer boundary of the part AND all interior holes
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None
    
    # Unwrap from extra dimension
    hierarchy = hierarchy[0]
    segments = []

    for idx, contour in enumerate(contours):
        # Skip tiny contours — rendering artifacts and noise
        # 100px area threshold filters specs without removing real bolt holes
        if cv2.contourArea(contour) < 100:
            continue

        # Simplify polygon — 0.002 = 0.2% of perimeter tolerance
        # Keeps enough points for accuracy without thousands of coordinates
        epsilon = 0.002 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        # Normalize coordinates to 0-1 range relative to image dimensions
        # YOLO requires all coordinates normalized between 0 and 1
        points = simplified.reshape(-1, 2)
        normalized = [(x / img_w, y / img_h) for x, y in points]
        flat = [val for point in normalized for val in point]

        segments.append(" ".join(f"{v:.6f}" for v in flat))

    if not segments:
        return None

    # YOLO segmentation format with multiple polygons:
    # Class_id [outer boundary coords] [hole 1 coords] [hole 2 coords] etc...
    # YOLO correctly interprets multiple polygon segments as one annotation
    return f"{class_id} " + " ".join(segments)
 
base_name = os.path.splitext(os.path.basename(file_path))[0]
 
# Render loop
for i in range(num_renders):
    frame_str = f"{i:04d}"
 
    # Camera randomization
    cam.location = random_camera_position()
    look_at(cam, obj.location)
 
    # Main light randomization
    light.location = (
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(2, 8),
    )
    light.data.energy = random.uniform(300, 1500)
    # Randomize light colour: warm yellow to cool white
    light.data.color = (
        random.uniform(0.8, 1.0),# R
        random.uniform(0.8, 1.0),# G
        random.uniform(0.6, 1.0),# B
    )
 
    # Fill light randomisation (weaker ambient)
    fill_light.location = (
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(1, 4),
    )
    fill_light.data.energy = random.uniform(50, 300)
 
    # Colour + surface randomization
    part_material.inputs["Base Color"].default_value = (
        random.uniform(0.0, 1.0),# R
        random.uniform(0.0, 1.0),# G
        random.uniform(0.0, 1.0),# B
        1.0# A
    )
    # Roughness: 0.0 = mirror finish, 1.0 = fully matte
    part_material.inputs["Roughness"].default_value = random.uniform(0.1, 0.9)
    # Metallic: 0.0 = plastic/painted surface, 1.0 = bare metal
    part_material.inputs["Metallic"].default_value  = random.uniform(0.0, 1.0)
 
    # Render 1: RGB image
    scene.render.filepath = os.path.join(output_path, f"{base_name}_{frame_str}.png")
    scene.cycles.samples = 64
    bpy.ops.render.render(write_still=True)
 
    # Render 2: segmentation mask
    # Swap to flat white material — 1 sample is enough, no lighting needed
    for slot in obj.material_slots:
        slot.material = mask_mat
 
    mask_path = os.path.join(output_path, f"{base_name}_{frame_str}_mask.png")
    scene.render.filepath = mask_path
    scene.cycles.samples = 1
    bpy.ops.render.render(write_still=True)
 
    # Restore original material
    for slot, mat in zip(obj.material_slots, original_materials):
        slot.material = mat
 
    # Reset sample count for next RGB render
    scene.cycles.samples = 64
 
    # Convert mask to YOLO annotation (.txt)
    yolo_line = mask_to_yolo_polygon(
        mask_path,
        scene.render.resolution_x,
        scene.render.resolution_y,
    )
 
    txt_path = os.path.join(output_path, f"{base_name}_{frame_str}.txt")
    if yolo_line:
        with open(txt_path, "w") as f:
            f.write(yolo_line + "\n")
    else:
        print(f"[WARNING] No contour found for frame {i} — skipping annotation.")
 
    # Delete intermediate mask file — only the .txt annotation is kept
    if os.path.exists(mask_path):
        os.remove(mask_path)
 
    print(f"[{i+1}/{num_renders}] {base_name}_{frame_str}.png + .txt done")
 
# Write one metadata.json per part folder
metadata = {
    "name": part_name,
    "part_number": part_number,
    "class_id": class_id,
}
with open(os.path.join(output_path, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# Confirmation message
print(f"Dataset generation complete — {num_renders} renders written to {output_path}")
 