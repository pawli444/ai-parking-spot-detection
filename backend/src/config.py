from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.environ.get("DATASET_DIR", str(REPO_ROOT / "dataset")))
DATA_YAML = Path(os.environ.get("DATA_YAML", str(DATASET_DIR / "data.yaml")))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(REPO_ROOT / "runs")))
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", str(RUNS_DIR / "detect")))

DEFAULT_RUN_NAME = os.environ.get("RUN_NAME", "yolov8s_parking_v1")
DEFAULT_EPOCHS = int(os.environ.get("EPOCHS", "100"))
DEFAULT_PATIENCE = int(os.environ.get("PATIENCE", "20"))
DEFAULT_IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
DEFAULT_BATCH = int(os.environ.get("BATCH", "32"))
DEFAULT_WORKERS = int(os.environ.get("WORKERS", "4"))
DEFAULT_CONF = float(os.environ.get("CONF", "0.25"))


def default_model_path():
    env_model = os.environ.get("YOLO_MODEL")
    if env_model:
        return env_model
    for name in ("yolov8s.pt", "yolov8n.pt", "yolov8m.pt"):
        candidate = REPO_ROOT / name
        if candidate.exists():
            return str(candidate)
    return "yolov8s.pt"


def default_best_weights(run_name=DEFAULT_RUN_NAME):
    return PROJECT_DIR / run_name / "weights" / "klasyfikator.pt"