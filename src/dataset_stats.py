import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from config import DATA_YAML, RUNS_DIR


def add_args(parser):
    parser.add_argument("--data-yaml", default=str(DATA_YAML))
    parser.add_argument("--out", default=str(RUNS_DIR / "dataset_stats.png"))
    return parser


def run(args):
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    class_names = cfg.get("names", [])
    colors = ["#2ecc71", "#e74c3c"]

    splits = ["train", "val", "test"]
    for split in splits:
        lbl_dir = base / "labels" / split
        if not lbl_dir.exists():
            continue
        counts = Counter()
        for lbl in lbl_dir.glob("*.txt"):
            lines = lbl.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if line:
                    counts[int(line.split()[0])] += 1
        total = sum(counts.values())
        print(f"{split:6s}: {sum(1 for _ in lbl_dir.glob('*.txt')):5d} images", end="")
        for i, name in enumerate(class_names):
            pct = 100 * counts[i] / max(total, 1)
            print(f" | {name}={counts[i]:6d} ({pct:.1f}%)", end="")
        print()

    fig, axes = plt.subplots(1, len(splits), figsize=(12, 4))
    for ax, split in zip(axes, splits):
        lbl_dir = base / "labels" / split
        if not lbl_dir.exists():
            ax.set_visible(False)
            continue
        counts = Counter()
        for lbl in lbl_dir.glob("*.txt"):
            lines = lbl.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if line:
                    counts[int(line.split()[0])] += 1
        vals = [counts[i] for i in range(len(class_names))]
        ax.bar(class_names, vals, color=colors[: len(class_names)])
        ax.set_title(f"{split} ({sum(vals):,} inst.)")
        ax.set_ylabel("Instances")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01, f"{v:,}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
