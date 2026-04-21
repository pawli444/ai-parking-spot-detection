from pathlib import Path
import os
import torch
from ultralytics import YOLO


def load_env(env_path: Path):
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def main():

    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    data_path = os.environ.get("DATA_PATH", str(root / "dataset" / "data.yaml"))
    data_dir = os.environ.get("DATA_DIR", str(root / "dataset"))

    print(f"Using DATA_PATH={data_path}")
    print(f"Using DATA_DIR={data_dir}")


    try:
        epochs = int(os.environ.get("EPOCHS", "10"))
    except ValueError:
        epochs = 10
    print(f"Using EPOCHS={epochs}")

    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} (torch.cuda.is_available()={torch.cuda.is_available()}, count={torch.cuda.device_count()})")

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        device=device,
        workers=4,
        batch=16
    )


if __name__ == '__main__':
    main()