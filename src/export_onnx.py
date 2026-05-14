import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DEFAULT_IMG_SIZE, DEFAULT_RUN_NAME, default_best_weights


def add_args(parser):
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    return parser


def run(args):
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = default_best_weights(args.run_name)

    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    model = YOLO(str(model_path))
    model.export(format="onnx", imgsz=args.imgsz, half=args.half, simplify=not args.no_simplify)

    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"Exported: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
