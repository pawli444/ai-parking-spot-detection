"""
parking_pipeline.py - gotowy pipeline: model + spots JSON -> filmik
Użycie:
    python parking_pipeline.py \
        --model src/runs/detect/train-9/weights/best.pt \
        --video video_dwa.mp4 \
        --spots spotsVIDEO_DWA_manual.json \
        --output wynik.mp4
 
Opcje:
    --conf       próg pewności modelu (domyślnie 0.15)
    --upscale    współczynnik upscale przed inferencją (domyślnie 2)
    --iou        próg IoU do matchowania detekcji ze spotami (domyślnie 0.15)
    --match      metoda matchowania: iou, ioa, center (domyślnie ioa)
    --ttl        ile klatek pamiętać ostatni stan spotu (domyślnie 10)
"""
 
import argparse
import json
from pathlib import Path
 
import cv2
import numpy as np
from ultralytics import YOLO
 
 
# ── kolory i etykiety ──────────────────────────────────────────────────────────
COLOR_FREE     = (50, 205, 50)    # zielony
COLOR_OCCUPIED = (0, 0, 220)      # czerwony
COLOR_UNKNOWN  = (160, 160, 160)  # szary
 
 
# ── geometria ──────────────────────────────────────────────────────────────────
def match_score(spot, det, mode):
    ax1, ay1, ax2, ay2 = spot
    bx1, by1, bx2, by2 = det
    if mode == "center":
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        return 1.0 if (ax1 <= cx <= ax2 and ay1 <= cy <= ay2) else 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if mode == "ioa":
        return inter / max(area_a, 1e-6)
    return inter / max(area_a + area_b - inter, 1e-6)
 
 
# ── inferencja ─────────────────────────────────────────────────────────────────
def detect(frame, model, conf, upscale, orig_w, orig_h):
    if upscale != 1:
        h, w = frame.shape[:2]
        frame_up = cv2.resize(frame, (w * upscale, h * upscale),
                              interpolation=cv2.INTER_LINEAR)
    else:
        frame_up = frame
 
    results = model(frame_up, conf=conf, imgsz=640, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls        = int(box.cls[0])
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        if upscale != 1:
            x1, x2 = x1 / upscale, x2 / upscale
            y1, y2 = y1 / upscale, y2 / upscale
        dets.append((np.array([x1, y1, x2, y2]), cls, conf_score))
    return dets
 
 
# ── rysowanie ──────────────────────────────────────────────────────────────────
def draw_spot(frame, spot, state, conf_score):
    x1, y1, x2, y2 = map(int, spot)
    if state == 0:
        color, label = COLOR_FREE, "FREE"
    elif state == 1:
        color, label = COLOR_OCCUPIED, "OCC"
    else:
        color, label = COLOR_UNKNOWN, "?"
 
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf_score:.2f}" if conf_score > 0.01 else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x1, y1 - th - 3), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
 
 
def draw_hud(frame, free, occupied, frame_idx, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (260, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
 
    cv2.putText(frame, f"FREE:     {free:3d}", (18, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_FREE, 2)
    cv2.putText(frame, f"OCCUPIED: {occupied:3d}", (18, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_OCCUPIED, 2)
 
    total = free + occupied
    if total > 0:
        pct = 100 * occupied / total
        bar = int(230 * occupied / total)
        cv2.rectangle(frame, (18, 85), (248, 100), (60, 60, 60), -1)
        bar_color = COLOR_OCCUPIED if pct > 70 else (50, 180, 50)
        cv2.rectangle(frame, (18, 85), (18 + bar, 100), bar_color, -1)
        cv2.putText(frame, f"Occupancy: {pct:.0f}%", (18, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
 
    secs = frame_idx / max(fps, 1)
    cv2.putText(frame, f"t={secs:.1f}s  #{frame_idx}", (18, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
 
 
# ── główna pętla ───────────────────────────────────────────────────────────────
def run(args):
    # model
    model = YOLO(str(args.model))
    print(f"Model: {args.model}")
    print(f"Klasy: {model.names}")
 
    # spoty
    with open(args.spots, "r", encoding="utf-8") as f:
        raw = json.load(f)
    spots = []
    for b in raw:
        if isinstance(b, dict):
            spots.append(np.array([b['x1'], b['y1'], b['x2'], b['y2']], dtype=float))
        else:
            spots.append(np.array(b, dtype=float))
    print(f"Spotów: {len(spots)}")
 
    # wideo
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Nie można otworzyć: {args.video}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Wideo: {width}x{height} @ {fps:.0f}fps, {total} klatek")
 
    # writer
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Nie można otworzyć writera: {out_path}")
 
    # stan spotów
    n = len(spots)
    states     = [0] * n   # 0=free domyślnie
    confs      = [0.0] * n
    last_seen  = [-999] * n
 
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        # detekcja
        dets = detect(frame, model, args.conf, args.upscale, width, height)
 
        # matchowanie detekcji → spoty
        for idx in range(n):
            best_score = 0.0
            best_cls   = None
            best_conf  = 0.0
            for det_box, cls, conf_score in dets:
                score = match_score(spots[idx], det_box, args.match)
                if score > best_score:
                    best_score = score
                    best_cls   = cls
                    best_conf  = conf_score
 
            threshold = 0.0 if args.match == "center" else args.iou
            if best_score > threshold and best_cls is not None:
                states[idx]    = best_cls
                confs[idx]     = best_conf
                last_seen[idx] = frame_idx
            else:
                # TTL — jeśli dawno nie widziany, ustaw free
                if (frame_idx - last_seen[idx]) > args.ttl:
                    states[idx] = 0
                    confs[idx]  = 0.0
 
        # zlicz
        free     = sum(1 for s in states if s == 0)
        occupied = sum(1 for s in states if s == 1)
 
        # rysuj spoty
        for idx in range(n):
            draw_spot(frame, spots[idx], states[idx], confs[idx])
 
        # HUD
        draw_hud(frame, free, occupied, frame_idx, fps)
 
        writer.write(frame)
        frame_idx += 1
 
        if fps > 0 and frame_idx % int(fps * 5) == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  klatka {frame_idx}/{total} ({pct:.0f}%) "
                  f"free={free} occupied={occupied} dets={len(dets)}")
 
    cap.release()
    writer.release()
    print(f"\nGotowe! Zapisano: {out_path}")
    print(f"Łączna liczba klatek: {frame_idx}")
 
 
# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Parking pipeline: model + spots → filmik")
    p.add_argument("--model",   default="src/runs/detect/train-9/weights/best.pt")
    p.add_argument("--video",   default="video_dwa.mp4")
    p.add_argument("--spots",   default="spotsVIDEO_DWA_manual.json")
    p.add_argument("--output",  default="wynik_parking.mp4")
    p.add_argument("--conf",    type=float, default=0.15,
                   help="próg pewności modelu")
    p.add_argument("--upscale", type=int,   default=2,
                   help="upscale klatki przed inferencją (1=brak, 2=2x, 3=3x)")
    p.add_argument("--iou",     type=float, default=0.15,
                   help="próg IoU/IoA do matchowania")
    p.add_argument("--match",   default="ioa",
                   choices=["iou", "ioa", "center"],
                   help="metoda matchowania detekcji do spotów")
    p.add_argument("--ttl",     type=int,   default=10,
                   help="ile klatek pamiętać ostatni stan spotu")
    args = p.parse_args()
    run(args)
 
 
if __name__ == "__main__":
    main()