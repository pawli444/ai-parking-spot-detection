from pathlib import Path

import yaml


def resolve_data_yaml(data_yaml: Path, out_path: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    cfg["path"] = str(base)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_path
