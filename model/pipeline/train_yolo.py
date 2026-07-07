"""Train YOLOv8s-seg for component detection using the prepared dataset."""

import os
from pathlib import Path
import torch
from ultralytics import YOLO
from dotenv import load_dotenv


def load_env_vars() -> Path:
    """
    Load environment variables from a .env file and validate
    required configuration.
    """
    load_dotenv()

    model_data_dir = os.getenv("MODEL_DATA_DIR")
    if not model_data_dir:
        raise RuntimeError("MODEL_DATA_DIR environment variable is required.")

    return Path(model_data_dir).expanduser().resolve()


MODEL_DATA_DIR = load_env_vars()
DATASET = MODEL_DATA_DIR / "dataset"
OUTPUT = MODEL_DATA_DIR / "models"

if __name__ == "__main__":
    # Load the pretrained YOLOv8 segmentation model.
    # Find documentation about the model here: https://docs.ultralytics.com/models/yolov8#overview
    print(torch.backends.mps.is_available())
    model = YOLO("yolov8s-seg.pt")

    # Train YOLOv8 for instance recognition with prepared dataset.
    model.train(
        data=str(DATASET),
        # number of times the model cycles through each image in the dataset.
        epochs=20,
        patience=5,
        # match the render resolution of raw dataset.
        imgsz=640,
        # Use batch=-1 to trigger AutoBatch.
        # Or set batch size (e.g. 8, 16, 32, 64)
        batch=16,
        cache="ram",
        project=str(OUTPUT),
        name="yolov8_singleclass",
        exist_ok=True,
    )

    # The result of training is the best-performing model checkpoint saved to best.pt.
    # best.pt is the trained model used for inference.
    # It contains the weights that performed best on the validation data during training.
    print("Training complete.")
    print(
        f"Weights saved to: "
        f"{OUTPUT / 'yolov8_singleclass' / 'weights' / 'best.pt'}"
)
