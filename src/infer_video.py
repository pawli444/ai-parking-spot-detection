"""
Parking spot detector – video inference
Improvements over v1:
  - Temporal state smoothing: each spot votes over last N frames → no flickering
  - match_spot: IOU + center-distance fallback → catches spots even when boxes shift
  - Better calibration: tracks class votes per spot, smarter min-hits default
  - Improved visuals: semi-transparent fill on locked spots + dot marker
  - Tighter NMS defaults (overlap 0.35, iou 0.35)
"""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import DEFAULT_CONF, DEFAULT_IMG_SIZE, DEFAULT_RUN_NAME, RUNS_DIR, default_best_weights


def add_args(parser):
    parser.add_argument("--video-input", required=True)
    parser.add_argument("--video-output", default=str(RUNS_DIR / "predict" / "parking_output.mp4"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fourcc", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--device", default=None)

    # ── Tiling (ALL frames) ───────────────────────────────────────────────────
    parser.add_argument("--tile", action="store_true",
                        help="Tiled inference on every frame — use this for wide/hi-res video")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tile size px (default 640)")
    parser.add_argument("--tile-overlap", type=float, default=0.35,
                        help="Tile overlap 0-0.9 (default 0.35)")
    parser.add_argument("--tile-nms-iou", type=float, default=0.35,
                        help="NMS IOU for merging tile detections (default 0.35)")

    # ── Lock-spots mode ───────────────────────────────────────────────────────
    parser.add_argument("--lock-spots", action="store_true",
                        help="Calibrate spot positions once, then only classify occupancy. "
                             "Eliminates flickering and double-counting.")
    parser.add_argument("--calib-frames", type=int, default=45,
                        help="Frames to use for spot calibration (default 45)")
    parser.add_argument("--conf-calib", type=float, default=None,
                        help="Confidence during calibration (default: same as --conf)")
    parser.add_argument("--imgsz-calib", type=int, default=None)
    parser.add_argument("--tile-calib", action="store_true",
                        help="Tile inference during calibration (overridden by --tile)")
    parser.add_argument("--spot-iou", type=float, default=0.3,
                        help="IOU threshold for matching detections to spots (default 0.3)")
    parser.add_argument("--spot-center-dist", type=float, default=0.6,
                        help="Center-distance fallback: fraction of avg box size (default 0.6). "
                             "Used when IOU=0 but boxes are close.")
    parser.add_argument("--spot-min-hits", type=int, default=3,
                        help="Min detection hits to keep a spot after calibration (default 3)")
    parser.add_argument("--smooth-frames", type=int, default=7,
                        help="Temporal smoothing window: majority vote over last N frames (default 7). "
                             "Higher = less flickering, more lag.")
    parser.add_argument("--unknown-policy",
                        choices=["free", "occupied", "hold", "unknown"],
                        default="hold",
                        help="What to show for unmatched spots: free/occupied/hold/unknown (default hold)")
    parser.add_argument("--state-ttl", type=int, default=20,
                        help="Frames before unmatched spot reverts to unknown-policy (default 20)")

    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only first N frames (debug)")
    return parser


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1);  ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(ua + ub - inter, 1e-6)


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_avg_size(b):
    return ((b[2] - b[0]) + (b[3] - b[1])) / 2.0


def nms_detections(detections, iou_thresh):
    """Per-class greedy NMS. detections = list of (box_np, cls, conf)."""
    if not detections:
        return []
    by_class = {}
    for d in detections:
        by_class.setdefault(d[1], []).append(d)
    kept = []
    for _, dets in by_class.items():
        dets = sorted(dets, key=lambda d: d[2], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if box_iou(best[0], d[0]) < iou_thresh]
    return kept


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_spot(det_box, spots, iou_thresh, center_dist_frac=0.6):
    """
    Match det_box to the best spot.
    Primary: IOU >= iou_thresh
    Fallback: center distance < center_dist_frac * avg_size  (catches shifted / smaller boxes)
    Returns (best_idx, score) or (-1, 0.0)
    """
    best_idx = -1
    best_score = 0.0

    dc = box_center(det_box)
    ds = box_avg_size(det_box)

    for i, spot in enumerate(spots):
        iou = box_iou(det_box, spot)
        if iou >= iou_thresh:
            if iou > best_score:
                best_score = iou
                best_idx   = i
        elif best_idx < 0:
            # Fallback: normalised center distance
            sc = box_center(spot)
            ss = box_avg_size(spot)
            avg_size = max((ds + ss) / 2.0, 1.0)
            dist = ((dc[0] - sc[0])**2 + (dc[1] - sc[1])**2) ** 0.5
            if dist < center_dist_frac * avg_size:
                norm_score = 1.0 - dist / (center_dist_frac * avg_size)
                if norm_score > best_score:
                    best_score = norm_score
                    best_idx   = i

    return (best_idx, best_score) if best_idx >= 0 else (-1, 0.0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(frame, model, conf, imgsz, device):
    results = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls = int(box.cls[0])
        cs  = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        dets.append((np.array([x1, y1, x2, y2], dtype=float), cls, cs))
    return dets


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def run_tiled(frame, model, conf, imgsz, device, tile_size, overlap, nms_iou):
    h, w = frame.shape[:2]
    tile_size = max(64, min(tile_size, w, h))
    overlap   = min(max(overlap, 0.0), 0.9)
    stride    = max(1, int(tile_size * (1.0 - overlap)))

    all_dets = []
    for y in tile_positions(h, tile_size, stride):
        for x in tile_positions(w, tile_size, stride):
            tile = frame[y: y + tile_size, x: x + tile_size]
            for det_box, cls, cs in run_model(tile, model, conf, imgsz, device):
                det_box[0] += x;  det_box[2] += x
                det_box[1] += y;  det_box[3] += y
                all_dets.append((det_box, cls, cs))

    return nms_detections(all_dets, nms_iou)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# Fill alpha for locked-spot boxes
_FILL_ALPHA = 0.18

def draw_locked_spot(frame, x1, y1, x2, y2, state, conf_val, unknown_color):
    """Draw a locked spot with semi-transparent fill + coloured border."""
    color_map = {0: (50, 205, 50), 1: (60, 20, 220)}
    color = color_map.get(state, unknown_color) if state is not None else unknown_color

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, _FILL_ALPHA, frame, 1 - _FILL_ALPHA, 0, frame)

    # Border
    border_thick = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thick)

    # Label
    if state is None:
        text = "?"
    else:
        bw = x2 - x1
        if bw < 32:
            text = "F" if state == 0 else "O"
        elif conf_val > 1e-3:
            text = f"{'F' if state == 0 else 'O'} {conf_val:.2f}"
        else:
            text = "FREE" if state == 0 else "OCC"

    _put_label(frame, x1, y1, x2, text, color)


def draw_free_run_box(frame, x1, y1, x2, y2, cls, conf_val):
    """Draw a free-running detection box (no fill, just border + label)."""
    color_map = {0: (50, 205, 50), 1: (60, 20, 220)}
    color = color_map.get(cls, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    bw = x2 - x1
    if bw < 32:
        text = "F" if cls == 0 else "O"
    else:
        text = f"{'FREE' if cls == 0 else 'OCC'} {conf_val:.2f}"
    _put_label(frame, x1, y1, x2, text, color)


def _put_label(frame, x1, y1, x2, text, color):
    """Draw label above the box if there's room, otherwise inside top."""
    bw   = max(x2 - x1, 1)
    fscl = max(0.28, min(0.40, bw / 110.0))
    th_  = 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fscl, th_)
    pad  = 2

    if y1 - th - pad * 2 >= 0:
        bg1, bg2, ty = y1 - th - pad * 2, y1, y1 - pad
    else:
        bg1, bg2, ty = y1, y1 + th + pad * 2, y1 + th + pad

    lx2 = min(x1 + tw + pad * 2, frame.shape[1] - 1)
    cv2.rectangle(frame, (x1, bg1), (lx2, bg2), color, -1)
    cv2.putText(frame, text, (x1 + pad, ty),
                cv2.FONT_HERSHEY_SIMPLEX, fscl, (255, 255, 255), th_, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def draw_hud(frame, counts, unknown_count, total_spots,
             in_calib, frame_idx, calib_frames, unknown_policy, lock_spots):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (295, 162), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FREE:     {counts[0]:3d}", (20, 44),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (50, 205, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OCCUPIED: {counts[1]:3d}", (20, 82),
                cv2.FONT_HERSHEY_DUPLEX, 0.82, (60, 20, 220), 2, cv2.LINE_AA)

    if in_calib:
        prog = int(250 * frame_idx / max(calib_frames - 1, 1))
        cv2.rectangle(frame, (15, 92), (265, 104), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 92), (15 + prog, 104), (220, 180, 40), -1)
        cv2.putText(frame, f"CALIBRATING {frame_idx + 1}/{calib_frames}", (20, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 60), 1, cv2.LINE_AA)
    elif lock_spots and unknown_policy == "unknown" and unknown_count:
        cv2.putText(frame, f"UNKNOWN:  {unknown_count:3d}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

    if total_spots > 0:
        occ    = 100 * counts[1] / total_spots
        bar_w  = int(250 * counts[1] / total_spots)
        cv2.rectangle(frame, (15, 122), (265, 136), (60, 60, 60), -1)
        bar_c  = (60, 20, 220) if occ > 70 else (50, 180, 50)
        cv2.rectangle(frame, (15, 122), (15 + bar_w, 136), bar_c, -1)
        cv2.putText(frame, f"Occupancy: {occ:.0f}%", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args):
    video_input = Path(args.video_input)
    if not video_input.exists():
        raise SystemExit(f"video not found: {video_input}")

    model_path = Path(args.model) if args.model else default_best_weights(args.run_name)
    if not model_path.exists():
        raise SystemExit(
            f"model not found: {model_path}\n"
            f"Hint: --model path/to/best.pt  OR  --run-name <folder-in-runs/detect>"
        )

    device = args.device or (0 if torch.cuda.is_available() else "cpu")

    default_out = str(RUNS_DIR / "predict" / "parking_output.mp4")
    output_dir  = Path(args.output_dir) if args.output_dir else None
    if output_dir is None and args.video_output == default_out:
        ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = RUNS_DIR / "predict" / f"{video_input.stem}_{args.run_name}_{ts}"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_output = output_dir / "parking_output.mp4"
    else:
        video_output = Path(args.video_output)
        video_output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    print(f"Model  : {model_path}")
    print(f"Names  : {model.names}")
    print(f"Conf   : {args.conf}   imgsz: {args.imgsz}   device: {device}")
    print(f"Tiling : {'ON  size=' + str(args.tile_size) + ' overlap=' + str(args.tile_overlap) if args.tile else 'OFF'}")
    if args.lock_spots:
        print(f"Mode   : LOCK-SPOTS  calib_frames={args.calib_frames}  smooth={args.smooth_frames}")
    else:
        print(f"Mode   : FREE-RUNNING  (add --lock-spots for stable output)")

    cap = cv2.VideoCapture(str(video_input))
    if not cap.isOpened():
        raise SystemExit("failed to open video")
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video  : {width}x{height} @ {fps}fps, {total} frames")

    fourcc_str = args.fourcc or ("MJPG" if video_output.suffix.lower() == ".avi" else "mp4v")
    writer = cv2.VideoWriter(str(video_output),
                             cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"failed to open VideoWriter (fourcc={fourcc_str})")
    print(f"Writer : {fourcc_str} -> {video_output}")

    unknown_color  = (160, 160, 160)
    lock_spots     = args.lock_spots
    calib_frames   = max(1, args.calib_frames) if lock_spots else 0
    calib_conf     = args.conf_calib  if args.conf_calib  is not None else max(0.1, args.conf - 0.05)
    calib_imgsz    = args.imgsz_calib if args.imgsz_calib is not None else args.imgsz
    use_tile_calib = args.tile or args.tile_calib
    smooth_n       = max(1, args.smooth_frames)

    # Spot data
    spots          = []   # np.array([x1,y1,x2,y2]) averaged position
    spot_hits      = []   # int: total detection count during calib
    spot_cls_votes = []   # list of int: cls seen during calib (to init state)
    spot_states    = []   # current smoothed state (0/1/None)
    spot_confs     = []   # latest conf for display
    spot_last_seen = []   # frame_idx of last match
    spot_history   = []   # deque of recent raw states for smoothing

    frame_idx     = 0
    stats_history = []

    def detect(frame, conf, imgsz, use_tile):
        if use_tile:
            return run_tiled(frame, model, conf, imgsz, device,
                             args.tile_size, args.tile_overlap, args.tile_nms_iou)
        return run_model(frame, model, conf, imgsz, device)

    while cap.isOpened():
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        in_calib   = lock_spots and frame_idx < calib_frames
        detections = (detect(frame, calib_conf, calib_imgsz, use_tile_calib)
                      if in_calib
                      else detect(frame, args.conf, args.imgsz, args.tile))

        if frame_idx < 2:
            sample = (f"cls={detections[0][1]} conf={detections[0][2]:.3f}"
                      if detections else "none")
            print(f"  [frame {frame_idx}] detections={len(detections)} ({sample})")

        counts        = {0: 0, 1: 0}
        unknown_count = 0

        # ── CALIBRATION ───────────────────────────────────────────────────────
        if in_calib:
            matched = set()
            for det_box, cls, conf in detections:
                idx, _ = match_spot(det_box, spots,
                                    args.spot_iou, args.spot_center_dist)
                if idx >= 0 and idx not in matched:
                    h = spot_hits[idx]
                    spots[idx] = (spots[idx] * h + det_box) / (h + 1)
                    spot_hits[idx] = h + 1
                    spot_cls_votes[idx].append(cls)
                    matched.add(idx)
                else:
                    spots.append(det_box.copy())
                    spot_hits.append(1)
                    spot_cls_votes.append([cls])

            # Draw calibration boxes
            for det_box, cls, conf in detections:
                x1, y1, x2, y2 = map(int, det_box)
                draw_free_run_box(frame, x1, y1, x2, y2, cls, conf)

            # Finalize calibration
            if frame_idx + 1 == calib_frames:
                keep = [i for i, h in enumerate(spot_hits) if h >= args.spot_min_hits]
                spots          = [spots[i]          for i in keep]
                spot_hits      = [spot_hits[i]      for i in keep]
                spot_cls_votes = [spot_cls_votes[i] for i in keep]

                # Init state from majority class vote during calibration
                def majority(votes):
                    if not votes:
                        return 0
                    return max(set(votes), key=votes.count)

                spot_states    = [majority(v) for v in spot_cls_votes]
                spot_confs     = [0.0]        * len(spots)
                spot_last_seen = [-10**9]      * len(spots)
                spot_history   = [deque([majority(v)] * smooth_n, maxlen=smooth_n)
                                  for v in spot_cls_votes]
                print(f"Locked {len(spots)} spots after calibration")

            stats_history.append({0: 0, 1: 0})

        # ── LOCKED-SPOTS INFERENCE ────────────────────────────────────────────
        elif lock_spots and spots:
            for idx, spot in enumerate(spots):
                best_score, best_cls, best_conf = 0.0, None, 0.0
                for det_box, cls, conf in detections:
                    iou = box_iou(spot, det_box)
                    # Also try center-distance for boxes that shifted a bit
                    dc = box_center(det_box)
                    sc = box_center(spot)
                    avg_sz = max(box_avg_size(spot), 1.0)
                    dist = ((dc[0]-sc[0])**2 + (dc[1]-sc[1])**2)**0.5
                    score = iou if iou >= args.spot_iou else (
                        (1.0 - dist / (args.spot_center_dist * avg_sz))
                        if dist < args.spot_center_dist * avg_sz else 0.0
                    )
                    if score > best_score:
                        best_score, best_cls, best_conf = score, cls, conf

                if best_score > 0 and best_cls is not None:
                    spot_history[idx].append(best_cls)
                    spot_confs[idx]     = best_conf
                    spot_last_seen[idx] = frame_idx
                else:
                    stale = ((frame_idx - spot_last_seen[idx]) > args.state_ttl
                             if args.state_ttl > 0 else True)
                    if stale and args.unknown_policy != "hold":
                        fallback = (0 if args.unknown_policy == "free"
                                    else (1 if args.unknown_policy == "occupied" else None))
                        spot_history[idx].append(fallback)
                        spot_confs[idx] = 0.0

                # Smoothed state = majority vote over recent history
                hist = list(spot_history[idx])
                valid = [s for s in hist if s is not None]
                if valid:
                    spot_states[idx] = max(set(valid), key=valid.count)
                else:
                    spot_states[idx] = None

            # Draw locked spots
            for idx, spot in enumerate(spots):
                state = spot_states[idx]
                x1, y1, x2, y2 = map(int, spot)
                if state is None:
                    unknown_count += 1
                draw_locked_spot(frame, x1, y1, x2, y2,
                                 state, spot_confs[idx], unknown_color)
                if state is not None:
                    counts[state] += 1

            stats_history.append(counts.copy())

        # ── FREE-RUNNING MODE ─────────────────────────────────────────────────
        else:
            for det_box, cls, conf in detections:
                counts[cls] = counts.get(cls, 0) + 1
                x1, y1, x2, y2 = map(int, det_box)
                draw_free_run_box(frame, x1, y1, x2, y2, cls, conf)
            stats_history.append(counts.copy())

        total_spots = counts[0] + counts[1]
        draw_hud(frame, counts, unknown_count, total_spots,
                 in_calib, frame_idx, calib_frames, args.unknown_policy, lock_spots)

        writer.write(frame)
        frame_idx += 1
        if fps > 0 and frame_idx % (fps * 5) == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"Frame {frame_idx}/{total} ({pct:.0f}%) "
                  f"free={counts[0]} occ={counts[1]}")

    cap.release()
    writer.release()
    print(f"Saved: {video_output}")

    if stats_history:
        avg_free = np.mean([s[0] for s in stats_history])
        avg_occ  = np.mean([s[1] for s in stats_history])
        occ_pct  = 100 * avg_occ / max(avg_free + avg_occ, 1e-6)
        print(f"Avg free={avg_free:.1f}  occ={avg_occ:.1f}  occupancy={occ_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Parking video inference")
    add_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()