import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from config import (
    DATA_YAML,
    DEFAULT_BATCH,
    DEFAULT_IMG_SIZE,
    DEFAULT_RUN_NAME,
    RUNS_DIR,
    default_best_weights,
)
from data_yaml import resolve_data_yaml


def add_args(parser):
    parser.add_argument("--data-yaml", default=str(DATA_YAML))
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=None)
    half_group = parser.add_mutually_exclusive_group()
    half_group.add_argument("--half", action="store_true")
    half_group.add_argument("--no-half", action="store_true")
    return parser


def run(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = default_best_weights(args.run_name)

    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    if args.half:
        half = True
    elif args.no_half:
        half = False
    else:
        half = torch.cuda.is_available()

    resolved_yaml = resolve_data_yaml(data_yaml, RUNS_DIR / "resolved_data.yaml")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(resolved_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        half=half,
        device=device,
        plots=True,
    )

    print("Test metrics")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    ap50 = getattr(metrics.box, "ap50", None)
    if ap50 is not None:
        print("Per class AP50")
        for idx, name in model.names.items():
            if idx < len(ap50):
                print(f"  {name}: {ap50[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
