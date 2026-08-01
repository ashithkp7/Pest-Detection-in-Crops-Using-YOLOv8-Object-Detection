"""
Pest Detection in Crops Using YOLOv8 Object Detection
Inference Pipeline

Loads the trained YOLOv8 model (best.pt) and runs pest detection on a
validation image, displaying bounding boxes, class labels, and
confidence scores.
"""

import os
from ultralytics import YOLO

# --- Load Validation Image ---
# `dataset` comes from having run train.py in the same session/directory
validation_image_dir = os.path.join(dataset.location, "valid/images")
test_images = os.listdir(validation_image_dir)

if test_images:
    test_image_path = os.path.join(validation_image_dir, test_images[1])
    print(f"Using image for inference: {test_image_path}")

    # --- Load Trained Model ---
    trained_model_path = "pest-detection/run1/weights/best.pt"
    model = YOLO(trained_model_path)

    # --- Run Inference ---
    results = model(test_image_path)

    # --- Visualize Results ---
    # Draws bounding boxes + class labels + confidence scores onto the image
    for r in results:
        r.show()
else:
    print("No validation images found to test inference.")
