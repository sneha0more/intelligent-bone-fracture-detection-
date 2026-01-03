import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for bone fracture detection")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base YOLOv8 model to start from (yolov8n.pt, yolov8s.pt, etc.)"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/yolov8_data.yaml",
        help="Path to YOLOv8 dataset YAML config"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Training image size"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/fracture_yolov8",
        help="Directory where YOLOv8 will save training runs"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Name for this training experiment"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    print(f"\n🚀 Starting YOLOv8 training...")
    print(f"Model: {args.model}")
    print(f"Dataset config: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Project directory: {args.project}/{args.name}\n")

    # Load YOLO model
    model = YOLO(args.model)

    # Train model
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("\n Training complete! Check the runs folder for results.\n")


if __name__ == "__main__":
    main()
