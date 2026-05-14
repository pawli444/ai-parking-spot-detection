import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, DEFAULT_IMG_SIZE, DEFAULT_RUN_NAME, RUNS_DIR, default_best_weights


def add_args(parser):
    parser.add_argument("--video-input", required=True)
    parser.add_argument("--video-output", default=str(RUNS_DIR / "predict" / "parking_output.mp4"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--device", default=None)
    return parser


def run(args):
    video_input = Path(args.video_input)
    if not video_input.exists():
        raise SystemExit(f"video not found: {video_input}")

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = default_best_weights(args.run_name)

    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    video_output = Path(args.video_output)
    video_output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_input))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        raise SystemExit("failed to open video")

    print(f"Video: {width}x{height} @ {fps}fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))

    colors = {0: (50, 205, 50), 1: (60, 20, 220)}
    labels = {0: "FREE", 1: "OCCUPIED"}

    frame_idx = 0
    stats_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]

        counts = {0: 0, 1: 0}
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            counts[cls] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{labels.get(cls, str(cls))} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        total_spots = counts[0] + counts[1]
        stats_history.append(counts.copy())

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"FREE:  {counts[0]:3d}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 205, 50), 2)
        cv2.putText(frame, f"OCCUPIED: {counts[1]:3d}", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 20, 220), 2)

        if total_spots > 0:
            occupancy = 100 * counts[1] / total_spots
            bar_w = int(250 * counts[1] / total_spots)
            cv2.rectangle(frame, (15, 108), (265, 125), (60, 60, 60), -1)
            cv2.rectangle(frame, (15, 108), (15 + bar_w, 125), (60, 20, 220) if occupancy > 70 else (50, 180, 50), -1)
            cv2.putText(frame, f"Occupancy: {occupancy:.0f}%", (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        out.write(frame)
        frame_idx += 1
        if fps > 0 and frame_idx % (fps * 5) == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"Frame {frame_idx}/{total} ({pct:.0f}%) - free={counts[0]} occupied={counts[1]}")

    cap.release()
    out.release()
    print(f"Saved video: {video_output}")

    if stats_history:
        avg_free = np.mean([s[0] for s in stats_history])
        avg_occ = np.mean([s[1] for s in stats_history])
        occ_pct = 100 * avg_occ / max(avg_free + avg_occ, 1e-6)
        print(f"Avg free={avg_free:.1f}, occupied={avg_occ:.1f}")
        print(f"Avg occupancy: {occ_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run video inference")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
