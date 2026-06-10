from ultralytics import YOLO
from pathlib import Path
import torch

# Train YOLOv8 for instance recognition with prepared dataset.

print(torch.backends.mps.is_available()) 

DATASET = Path("data/yolo_dataset/dataset.yaml")
OUTPUT  = Path("models")

# Load the pretrained YOLOv8 segmentation model.
model = YOLO("yolov8s-seg.pt")

# Training parameters.
model.train(
    data=str(DATASET),
    # number of times the model cycles through each image in the dataset.
    epochs=15,
    patience=5,
    # match the render resolution of raw dataset.
    imgsz=640,
    # Use batch=-1 to tigger AutoBatch.
    # Or set batch size (e.g. 16, 32, 64)
    batch=16,
    # use Apply Silicon GPU.
    device="mps",
    cache="ram",
    project=str(OUTPUT),
    name="yolov8_parts_singleclass",
    exist_ok=True,
)

print("Training complete.")
print(f"Weights saved to: models/yolov8_parts/weights/best.pt")