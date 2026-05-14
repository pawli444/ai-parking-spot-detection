import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from config import (
    DATA_YAML,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKERS,
    PROJECT_DIR,
    RUNS_DIR,
    default_model_path,
)
from data_yaml import resolve_data_yaml


def add_args(parser):
    parser.add_argument("--data-yaml", default=str(DATA_YAML))
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", choices=["ram", "disk", "none"], default=None)

    half_group = parser.add_mutually_exclusive_group()
    half_group.add_argument("--half", action="store_true")
    half_group.add_argument("--no-half", action="store_true")
    return parser


def run(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    model_path = args.model or default_model_path()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    if args.half:
        half = True
    elif args.no_half:
        half = False
    else:
        half = torch.cuda.is_available()

    if args.cache == "none":
        cache = False
    elif args.cache:
        cache = args.cache
    else:
        cache = "ram" if torch.cuda.is_available() else False

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    resolved_yaml = resolve_data_yaml(data_yaml, RUNS_DIR / "resolved_data.yaml")

    model = YOLO(model_path)
    results = model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        half=half,
        cache=cache,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        erasing=0.2,
        scale=0.3,
        translate=0.1,
        weight_decay=0.0005,
        dropout=0.0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        val=True,
        plots=True,
        save_period=10,
        verbose=True,
    )

    best = project_dir / args.name / "weights" / "best.pt"
    print(f"Best model: {best}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
