import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on the image.
    """
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def run_yolo(weights_path: str, image_path: str, output_path: str):
    """
    Runs inference using a YOLOv8 model.
    """
    model = YOLO(weights_path)

    print(f"\n🔍 Running YOLOv8 inference on: {image_path}")

    # Run inference
    results = model(image_path)[0]

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at: {image_path}")

    # Extract bounding boxes
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    print(f"📦 Detected {len(boxes)} objects")

    # Draw predictions
    img_out = draw_boxes(img, boxes)

    # Save output
    cv2.imwrite(output_path, img_out)
    print(f" Saved annotated image to: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on an X-ray image")
    
    parser.add_argument(
        "--model-type",
        choices=["yolov8"],
        default="yolov8",
        help="Select which model type to run inference with"
    )

    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="prediction_output.png", help="Path to save output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "yolov8":
        run_yolo(args.weights, args.image, args.output)
    else:
        raise NotImplementedError("Only YOLOv8 inference is currently implemented.")
