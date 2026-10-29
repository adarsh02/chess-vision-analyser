import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8s to locate chessboards.")
    parser.add_argument("--data", type=Path, default=Path("data/boards/dataset.yaml"))
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/detector"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="cpu or a GPU index such as 0.")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Dataset YAML not found. Run training.generate_data first.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    if args.output.exists():
        parser.error("Choose a new output directory to avoid replacing a previous run.")
    model = YOLO(args.weights)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch_size,
        project=str(args.output.resolve().parent),
        name=args.output.name,
        device=args.device,
        seed=args.seed,
        mosaic=0.0,
    )


if __name__ == "__main__":
    main()
