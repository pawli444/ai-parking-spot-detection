import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json

from config import DEFAULT_CONF, DEFAULT_IMG_SIZE, DEFAULT_RUN_NAME, RUNS_DIR, default_best_weights


def add_args(parser):
    parser.add_argument("--video-input", required=True)
    parser.add_argument("--video-output", default=str(RUNS_DIR / "predict" / "parking_output.mp4"))
    parser.add_argument("--output-dir", default=None, help="Output directory for artifacts")
    parser.add_argument("--fourcc", default=None, help="Output codec (e.g., mp4v, MJPG, XVID)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lock-spots", action="store_true", help="Calibrate spots once, then only classify occupancy")
    parser.add_argument("--calib-frames", type=int, default=30, help="Frames used to build spot map")
    parser.add_argument("--conf-calib", type=float, default=None, help="Confidence during calibration")
    parser.add_argument("--imgsz-calib", type=int, default=None, help="Image size during calibration")
    parser.add_argument("--tile-calib", action="store_true", help="Use tiled inference during calibration")
    parser.add_argument("--tile-infer", action="store_true", help="Use tiled inference for all frames")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for calibration")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap ratio during calibration")
    parser.add_argument("--tile-nms-iou", type=float, default=0.5, help="NMS IOU for tiled calibration")
    parser.add_argument("--spot-iou", type=float, default=0.5, help="IOU threshold for matching spots")
    parser.add_argument(
        "--spot-match",
        choices=["iou", "ioa", "center"],
        default=None,
        help="Spot matching method: iou, ioa (overlap/spot area), or center",
    )
    parser.add_argument("--spot-pad", type=int, default=0, help="Pad spot boxes by N pixels")
    parser.add_argument("--spot-min-hits", type=int, default=5, help="Min hits to keep a spot after calibration")
    parser.add_argument(
        "--unknown-policy",
        choices=["free", "occupied", "hold", "unknown"],
        default="free",
        help="Behavior when a spot has no match",
    )
    parser.add_argument("--state-ttl", type=int, default=15, help="Frames to keep last state when not matched")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debug)")
    parser.add_argument("--spots-file", default=None, help="JSON/CSV file with fixed spots (x1,y1,x2,y2)")
    return parser


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union


def match_box_to_spots(box, spots, iou_thresh):
    return match_box_to_spots_mode(box, spots, iou_thresh, "iou")


def spot_match_score(spot, det, mode):
    ax1, ay1, ax2, ay2 = spot
    bx1, by1, bx2, by2 = det
    if mode == "center":
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if (ax1 <= cx <= ax2) and (ay1 <= cy <= ay2):
            return 1.0
        return 0.0
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if mode == "ioa":
        return inter_area / max(area_a, 1e-6)
    union = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union


def match_box_to_spots_mode(box, spots, iou_thresh, match_mode):
    best_idx = -1
    best_score = 0.0
    for idx, spot in enumerate(spots):
        score = spot_match_score(spot, box, match_mode)
        if score > best_score:
            best_score = score
            best_idx = idx
    if match_mode == "center":
        return (best_idx, best_score) if best_score > 0.0 else (-1, best_score)
    if best_score >= iou_thresh:
        return best_idx, best_score
    return -1, best_score


def run_model(frame, model, conf, imgsz, device):
    results = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        detections.append((np.array([x1, y1, x2, y2], dtype=float), cls, conf_score))
    return detections


def nms_detections(detections, iou_thresh):
    if not detections:
        return []
    by_class = {}
    for det in detections:
        by_class.setdefault(det[1], []).append(det)

    kept = []
    for cls, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d[2], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if box_iou(best[0], d[0]) < iou_thresh]
    return kept


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def run_tiled_detections(frame, model, conf, imgsz, device, tile_size, overlap, nms_iou):
    height, width = frame.shape[:2]
    min_dim = min(width, height)
    if tile_size >= min_dim:
        tile_size = max(128, min_dim // 2)
        tile_size = min(tile_size, min_dim)
    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    overlap = min(max(overlap, 0.0), 0.9)
    stride_x = max(1, int(tile_w * (1.0 - overlap)))
    stride_y = max(1, int(tile_h * (1.0 - overlap)))

    detections = []
    for y in tile_positions(height, tile_h, stride_y):
        for x in tile_positions(width, tile_w, stride_x):
            tile = frame[y : y + tile_h, x : x + tile_w]
            tile_dets = run_model(tile, model, conf, imgsz, device)
            for det_box, cls, conf_score in tile_dets:
                det_box[0] += x
                det_box[2] += x
                det_box[1] += y
                det_box[3] += y
                detections.append((det_box, cls, conf_score))

    return nms_detections(detections, nms_iou)


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

    default_output = str(RUNS_DIR / "predict" / "parking_output.mp4")
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.video_output == default_output:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = RUNS_DIR / "predict" / f"{video_input.stem}_{args.run_name}_{timestamp}"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_output = output_dir / "parking_output.mp4"
    else:
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

    fourcc_str = args.fourcc
    if fourcc_str is None:
        suffix = video_output.suffix.lower()
        if suffix == ".avi":
            fourcc_str = "MJPG"
        elif suffix in (".mp4", ".m4v"):
            fourcc_str = "mp4v"
        else:
            fourcc_str = "mp4v"

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
    if not out.isOpened():
        raise SystemExit(f"failed to open video writer (fourcc={fourcc_str})")

    print(f"Writer: {fourcc_str} -> {video_output}")

    colors = {0: (50, 205, 50), 1: (60, 20, 220)}
    labels = {0: "FREE", 1: "OCCUPIED"}
    unknown_color = (180, 180, 180)

    lock_spots = args.lock_spots
    spot_match = args.spot_match or "iou"
    if args.spots_file:
        # Auto-tune defaults for custom spot models when user didn't override
        if args.conf == DEFAULT_CONF:
            args.conf = 0.3
        if args.imgsz == DEFAULT_IMG_SIZE:
            args.imgsz = 640
        if args.spot_iou == 0.5:
            args.spot_iou = 0.3
        if args.spot_match is None:
            spot_match = "center"
        print(
            f"Auto-tune: conf={args.conf}, imgsz={args.imgsz}, "
            f"spot_iou={args.spot_iou}, spot_match={spot_match}"
        )
    calib_frames = max(1, args.calib_frames) if lock_spots else 0
    calib_conf = args.conf_calib if args.conf_calib is not None else args.conf
    calib_imgsz = args.imgsz_calib if args.imgsz_calib is not None else args.imgsz
    unknown_policy = args.unknown_policy
    state_ttl = max(0, args.state_ttl)
    # If a spots file is provided, load fixed spots and enable lock_spots
    if args.spots_file:
        spots_path = Path(args.spots_file)
        if not spots_path.exists():
            raise SystemExit(f"spots file not found: {spots_path}")
        with open(spots_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        spot_pad = max(0, int(args.spot_pad))
        spots = []
        for b in data:
            # support dicts {x1,y1,x2,y2} or list/tuple
            if isinstance(b, dict):
                x1 = b.get("x1")
                y1 = b.get("y1")
                x2 = b.get("x2")
                y2 = b.get("y2")
            else:
                x1, y1, x2, y2 = b
            x1 = max(0.0, float(x1) - spot_pad)
            y1 = max(0.0, float(y1) - spot_pad)
            x2 = min(float(width - 1), float(x2) + spot_pad)
            y2 = min(float(height - 1), float(y2) + spot_pad)
            spots.append(np.array([float(x1), float(y1), float(x2), float(y2)], dtype=float))
        # initialize auxiliary spot arrays
        lock_spots = True
        calib_frames = 0
        if args.unknown_policy in ("free", "hold"):
            default_state = 0
        elif args.unknown_policy == "occupied":
            default_state = 1
        else:
            default_state = None
        spot_hits = [1] * len(spots)
        spot_states = [default_state] * len(spots)
        spot_confs = [0.0] * len(spots)
        spot_last_seen = [-10**9] * len(spots)
    else:
        spots = []
        spot_hits = []
        spot_states = []
        spot_confs = []
        spot_last_seen = []

    frame_idx = 0
    stats_history = []

    while cap.isOpened():
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        calib_mode = lock_spots and frame_idx < calib_frames
        if calib_mode:
            if args.tile_calib:
                detections = run_tiled_detections(
                    frame,
                    model,
                    calib_conf,
                    calib_imgsz,
                    device,
                    args.tile_size,
                    args.tile_overlap,
                    args.tile_nms_iou,
                )
            else:
                detections = run_model(frame, model, calib_conf, calib_imgsz, device)
        else:
            detections = run_model(frame, model, args.conf, args.imgsz, device)

        counts = {0: 0, 1: 0}
        unknown_count = 0

        if lock_spots and frame_idx < calib_frames:
            matched = set()
            for det_box, cls, conf in detections:
                idx, _ = match_box_to_spots_mode(det_box, spots, args.spot_iou, spot_match)
                if idx >= 0 and idx not in matched:
                    hits = spot_hits[idx]
                    spots[idx] = (spots[idx] * hits + det_box) / (hits + 1)
                    spot_hits[idx] = hits + 1
                    matched.add(idx)
                else:
                    spots.append(det_box.copy())
                    spot_hits.append(1)

            for det_box, cls, conf in detections:
                x1, y1, x2, y2 = map(int, det_box)
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{labels.get(cls, str(cls))} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if frame_idx + 1 == calib_frames:
                keep = [i for i, hits in enumerate(spot_hits) if hits >= args.spot_min_hits]
                spots = [spots[i] for i in keep]
                spot_hits = [spot_hits[i] for i in keep]
                if unknown_policy in ("free", "hold"):
                    default_state = 0
                elif unknown_policy == "occupied":
                    default_state = 1
                else:
                    default_state = None
                spot_states = [default_state] * len(spots)
                spot_confs = [0.0] * len(spots)
                spot_last_seen = [-10**9] * len(spots)
                print(f"Locked {len(spots)} spots after calibration")

            total_spots = len(spots)
            stats_history.append({0: 0, 1: 0})
        elif lock_spots and spots:
            for idx, spot in enumerate(spots):
                best_score = 0.0
                best_cls = None
                best_conf = 0.0
                for det_box, cls, conf in detections:
                    score = spot_match_score(spot, det_box, spot_match)
                    if score > best_score:
                        best_score = score
                        best_cls = cls
                        best_conf = conf
                if spot_match == "center":
                    matched = best_score > 0.0
                else:
                    matched = best_score >= args.spot_iou
                if matched and best_cls is not None:
                    spot_states[idx] = best_cls
                    spot_confs[idx] = best_conf
                    spot_last_seen[idx] = frame_idx
                elif unknown_policy != "hold":
                    stale = True
                    if state_ttl > 0 and spot_last_seen:
                        last_seen = spot_last_seen[idx]
                        stale = (frame_idx - last_seen) > state_ttl
                    if stale:
                        if unknown_policy == "free":
                            spot_states[idx] = 0
                        elif unknown_policy == "occupied":
                            spot_states[idx] = 1
                        else:
                            spot_states[idx] = None
                        spot_confs[idx] = 0.0

            for idx, spot in enumerate(spots):
                state = spot_states[idx]
                if state is None:
                    unknown_count += 1
                    label = "UNKNOWN"
                    color = unknown_color
                    conf_text = ""
                else:
                    counts[state] += 1
                    label = labels.get(state, str(state))
                    color = colors.get(state, (255, 255, 255))
                    conf_text = f" {spot_confs[idx]:.2f}" if spot_confs[idx] > 1e-3 else ""

                x1, y1, x2, y2 = map(int, spot)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label}{conf_text}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            total_spots = counts[0] + counts[1]
            stats_history.append(counts.copy())
        else:
            for det_box, cls, conf in detections:
                counts[cls] += 1

                x1, y1, x2, y2 = map(int, det_box)
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{labels.get(cls, str(cls))} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            total_spots = counts[0] + counts[1]
            stats_history.append(counts.copy())

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"FREE:  {counts[0]:3d}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 205, 50), 2)
        cv2.putText(frame, f"OCCUPIED: {counts[1]:3d}", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 20, 220), 2)

        if lock_spots and frame_idx < calib_frames:
            cv2.putText(frame, f"CALIBRATING: {frame_idx + 1}/{calib_frames}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        elif lock_spots and unknown_policy == "unknown":
            cv2.putText(frame, f"UNKNOWN: {unknown_count:3d}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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
