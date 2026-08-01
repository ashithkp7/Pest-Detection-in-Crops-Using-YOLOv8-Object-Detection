"""
Pest Detection in Crops Using YOLOv8 Object Detection
Training Pipeline

Downloads the pest dataset from Roboflow and fine-tunes a pre-trained
YOLOv8n model (transfer learning from COCO) to detect crop pests.
"""

import os
from roboflow import Roboflow
from ultralytics import YOLO

# --- Dataset Acquisition ---
# NOTE: Set your Roboflow API key as an environment variable rather than
# hardcoding it, e.g.: export ROBOFLOW_API_KEY="your_key_here"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("yolo-dv5qa").project("pest-uruhn-yfh5t")
version = project.version(1)
dataset = version.download("yolov8")
print(f"Dataset downloaded to: {dataset.location}")

# --- Model Initialization & Training ---
data_yaml_path = os.path.join(dataset.location, "data.yaml")

model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8n (COCO weights)

results = model.train(
    data=data_yaml_path,
    epochs=25,
    imgsz=640,
    project="pest-detection",
    name="run1",
)

# Best-performing weights are automatically saved to:
# pest-detection/run1/weights/best.pt
