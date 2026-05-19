import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from collections import deque

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

parser = argparse.ArgumentParser(description='Simple occupancy from spots with forced initial occupied count')
parser.add_argument('--video', default='video_dwa.mp4')
parser.add_argument('--spots', default='spotsVIDEO_DWA_manual.json')
parser.add_argument('--out', default='NEWVIDEO_cv_forced5.mp4')
parser.add_argument('--force-n', type=int, default=5, help='Force at least N occupied at start')
parser.add_argument('--frames-check', type=int, default=5, help='Frames to average for initial calibration')
parser.add_argument('--thresh', type=float, default=0.6, help='Threshold on normalized darkness for occupied')
parser.add_argument('--max_frames', type=int, default=None)
parser.add_argument('--use-model', action='store_true', help='Use YOLO model for per-spot detections instead of simple brightness test')
parser.add_argument('--weights', default='src/runs/detect/train-9/weights/best.pt', help='Path to YOLO weights')
parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for model')
parser.add_argument('--imgsz', type=int, default=640, help='Image size for model inference')
parser.add_argument('--tiles', type=int, default=1, help='Split frame into NxN tiles for inference')
parser.add_argument('--overlap', type=float, default=0.1, help='Tile overlap fraction (0-0.5)')
parser.add_argument('--match-iou', type=float, default=0.2, help='Minimum IoU between detection and spot to count as occupied')
parser.add_argument('--min-area', type=int, default=300, help='Minimum detection area in px to consider')
parser.add_argument('--nms-iou', type=float, default=0.45, help='NMS IoU for merging tile detections')
parser.add_argument('--per-roi', action='store_true', help='Run model on each ROI crop (recommended when spots JSON available)')
parser.add_argument('--smooth-window', type=int, default=3, help='Temporal smoothing window (frames)')
parser.add_argument('--smooth-min', type=int, default=2, help='Minimum positives in window to consider occupied')
parser.add_argument('--roi-conf', type=float, default=0.05, help='Confidence threshold for per-ROI classification')
parser.add_argument('--edge-thresh', type=float, default=0.02, help='Minimum edge density inside ROI to accept occupied')
args = parser.parse_args()

video_path = Path(args.video)
spots_path = Path(args.spots)

if not video_path.exists():
    raise SystemExit(f'video not found: {video_path}')
if not spots_path.exists():
    raise SystemExit(f'spots file not found: {spots_path}')

with open(spots_path, 'r', encoding='utf-8') as f:
    spots = json.load(f)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit('failed to open video')

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

# Read first frames_check frames and compute mean intensity per spot
frames_check = args.frames_check
frame_idx = 0
agg_vals = [0.0] * len(spots)
count_frames = 0

while count_frames < frames_check:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i, s in enumerate(spots):
        x1, y1, x2, y2 = int(s['x1']), int(s['y1']), int(s['x2']), int(s['y2'])
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            val = 255.0
        else:
            val = float(crop.mean())
        agg_vals[i] += val
    count_frames += 1

if count_frames == 0:
    raise SystemExit('no frames to analyze')

avg_vals = [v / count_frames for v in agg_vals]
# Normalize (0 dark -> occupied, 255 bright -> free)
norm = [(255.0 - v) / 255.0 for v in avg_vals]

# Determine initial occupied by threshold
occupied_init = [1 if n >= args.thresh else 0 for n in norm]
num_occ = sum(occupied_init)
if num_occ < args.force_n:
    # force top-N by norm value
    idxs = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
    for j in range(args.force_n):
        occupied_init[idxs[j]] = 1

print('Initial occupancy forced/estimated count:', sum(occupied_init))

# If using model, load it and prepare mapping
model = None
class_name_to_idx = {}
if args.use_model:
    if YOLO is None:
        raise SystemExit('Ultralytics YOLO not available in this environment')
    model = YOLO(args.weights)
    # get names mapping if available
    try:
        names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else None
    except Exception:
        names = None
    if names is None:
        # fallback: assume binary: 0 empty, 1 occupied
        names = {0: 'space-empty', 1: 'space-occupied'}
    for k, v in names.items():
        class_name_to_idx[v] = k


def letterbox(img, new_size):
    h0, w0 = img.shape[:2]
    r = min(new_size / w0, new_size / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    img_resized = cv2.resize(img, (new_w, new_h))
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, r, pad_left, pad_top


def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def nms_numpy(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou_vals = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou_vals <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# Rewind video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0
max_frames = args.max_frames

# Temporal smoothing histories (pre-fill with initial forced state)
histories = [deque(maxlen=args.smooth_window) for _ in spots]
for i, v in enumerate(occupied_init):
    for _ in range(args.smooth_window):
        histories[i].append(v)

while True:
    if max_frames is not None and frame_idx >= max_frames:
        break
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counts = {0:0, 1:0}
    detections = []
    if args.use_model and model is not None:
        # If per-roi mode, we will run model per-ROI later; otherwise do tiled/global inference
        if args.per_roi:
            detections = None
        else:
            # tiled inference
            tiles = max(1, args.tiles)
            h_frame, w_frame = frame.shape[:2]
            if tiles == 1:
                try:
                    res = model(frame, conf=args.conf, imgsz=args.imgsz)
                    if len(res) > 0:
                        r0 = res[0]
                        boxes = getattr(r0, 'boxes', None)
                        if boxes is not None:
                            for b in boxes:
                                try:
                                    xyxy = b.xyxy.cpu().numpy()[0]
                                except Exception:
                                    xyxy = np.array(b.xyxy).astype(float)
                                try:
                                    confv = float(b.conf.cpu().numpy()[0])
                                except Exception:
                                    confv = float(getattr(b, 'conf', 0.0))
                                try:
                                    cls = int(b.cls.cpu().numpy()[0])
                                except Exception:
                                    cls = int(getattr(b, 'cls', 0))
                                detections.append((xyxy, confv, cls))
                except Exception:
                    detections = []
            else:
                # compute tile coords with overlap
                ov = float(min(max(0.0, args.overlap), 0.5))
                step_x = int(w_frame / tiles * (1 - ov))
                step_y = int(h_frame / tiles * (1 - ov))
                tw = int(w_frame / tiles) + 1
                th = int(h_frame / tiles) + 1
                all_boxes = []
                all_scores = []
                all_cls = []
                for y0 in range(0, max(1, h_frame - 1), step_y):
                    for x0 in range(0, max(1, w_frame - 1), step_x):
                        x1 = x0
                        y1 = y0
                        x2 = min(w_frame, x0 + tw)
                        y2 = min(h_frame, y0 + th)
                        tile = frame[y1:y2, x1:x2]
                        if tile.size == 0:
                            continue
                        try:
                            res = model(tile, conf=args.conf, imgsz=args.imgsz)
                        except Exception:
                            continue
                        if len(res) == 0:
                            continue
                        r0 = res[0]
                        boxes = getattr(r0, 'boxes', None)
                        if boxes is None:
                            continue
                        for b in boxes:
                            try:
                                xyxy = b.xyxy.cpu().numpy()[0]
                            except Exception:
                                xyxy = np.array(b.xyxy).astype(float)
                            # shift coords
                            xyxy[0] += x1
                            xyxy[1] += y1
                            xyxy[2] += x1
                            xyxy[3] += y1
                            try:
                                confv = float(b.conf.cpu().numpy()[0])
                            except Exception:
                                confv = float(getattr(b, 'conf', 0.0))
                            try:
                                cls = int(b.cls.cpu().numpy()[0])
                            except Exception:
                                cls = int(getattr(b, 'cls', 0))
                            area = max(0, (xyxy[2]-xyxy[0])) * max(0, (xyxy[3]-xyxy[1]))
                            if area < args.min_area:
                                continue
                        all_boxes.append(xyxy)
                        all_scores.append(confv)
                        all_cls.append(cls)
            # apply NMS
            keep = nms_numpy(all_boxes, all_scores, args.nms_iou)
            for k in keep:
                detections.append((all_boxes[k], all_scores[k], all_cls[k]))

    for i, s in enumerate(spots):
        x1, y1, x2, y2 = int(s['x1']), int(s['y1']), int(s['x2']), int(s['y2'])
        occ = occupied_init[i]
        if args.use_model and args.per_roi and model is not None:
            # run model on ROI crop, letterbox to imgsz
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                occ_raw = occupied_init[i]
            else:
                pad_img, scale, pad_x, pad_y = letterbox(crop, args.imgsz)
                try:
                    res_roi = model(pad_img, conf=args.roi_conf, imgsz=args.imgsz)
                except Exception:
                    res_roi = []
                occ_raw = 0
                if len(res_roi) > 0:
                    r0 = res_roi[0]
                    boxes = getattr(r0, 'boxes', None)
                    if boxes is not None and len(boxes) > 0:
                        # pick best by conf
                        best_conf = 0.0
                        best_name = None
                        for b in boxes:
                            try:
                                confv = float(b.conf.cpu().numpy()[0])
                            except Exception:
                                confv = float(getattr(b, 'conf', 0.0))
                            try:
                                cls = int(b.cls.cpu().numpy()[0])
                            except Exception:
                                cls = int(getattr(b, 'cls', 0))
                            try:
                                name = model.model.names.get(cls, None)
                            except Exception:
                                name = None
                            if confv >= best_conf:
                                best_conf = confv
                                best_name = name
                        if best_conf >= args.roi_conf:
                            # additionally check edge density to avoid shadows
                            try:
                                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                edges = cv2.Canny(crop_gray, 50, 150)
                                edge_density = float(np.count_nonzero(edges)) / max(1, crop_gray.size)
                            except Exception:
                                edge_density = 0.0
                            if edge_density >= args.edge_thresh:
                                if best_name is None:
                                    occ_raw = 1
                                elif 'occup' in best_name or 'occupied' in best_name:
                                    occ_raw = 1
                                else:
                                    occ_raw = 1
                            else:
                                # low edge density — likely shadow/texture => ignore unless very confident
                                if best_conf >= max(0.25, args.roi_conf * 4):
                                    occ_raw = 1
                                else:
                                    occ_raw = 0
                # fallback to brightness if no strong detection
                if occ_raw == 0 and frame_idx != 0:
                    crop_gray = gray[y1:y2, x1:x2]
                    if crop_gray.size == 0:
                        occ_raw = occupied_init[i]
                    else:
                        val = float(crop_gray.mean())
                        normv = (255.0 - val) / 255.0
                        occ_raw = 1 if normv >= args.thresh else 0
            # update history and smooth
            histories[i].append(occ_raw)
            occ = 1 if sum(histories[i]) >= args.smooth_min else 0
        elif args.use_model and len(detections) > 0:
            # match detections to spot by IoU
            best_conf = 0.0
            best_match = None
            for (xyxy, confv, cls) in detections:
                # filter by class name if possible: prefer occupied-like labels
                cls_name = None
                try:
                    cls_name = model.model.names.get(cls, None)
                except Exception:
                    cls_name = None
                # If class label indicates empty/occupied, prefer occupied
                if cls_name is not None and ('empty' in cls_name and 'occup' not in cls_name):
                    # likely empty class, skip for occupancy
                    pass
                # compute IoU between detection and spot
                det_box = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
                spot_box = [x1, y1, x2, y2]
                iouv = iou(det_box, spot_box)
                if iouv >= args.match_iou and confv >= best_conf:
                    best_conf = confv
                    best_match = (det_box, confv, cls, cls_name)
            if best_match is not None:
                # assign occupied if detection matched (and class not explicitly empty)
                _, confv, cls, cls_name = best_match
                if cls_name is not None and 'empty' in cls_name:
                    occ = 0
                else:
                    occ = 1
            else:
                # no matching detection: keep brightness fallback (except for first frame where we keep forced init)
                if frame_idx == 0:
                    occ = occupied_init[i]
                else:
                    crop = gray[y1:y2, x1:x2]
                    if crop.size == 0:
                        occ = occupied_init[i]
                    else:
                        val = float(crop.mean())
                        normv = (255.0 - val) / 255.0
                        occ = 1 if normv >= args.thresh else 0
        else:
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                occ = occupied_init[i]
            else:
                val = float(crop.mean())
                normv = (255.0 - val) / 255.0
                occ = 1 if normv >= args.thresh else 0
            if frame_idx == 0:
                occ = occupied_init[i]
        counts[occ] += 1
        color = (0,255,0) if occ==0 else (60,20,220)
        label = 'FREE' if occ==0 else 'OCC'
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f'{label}', (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)

    cv2.putText(frame, f'FREE: {counts[0]} OCC: {counts[1]}', (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200),2)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print('Saved annotated video to', args.out)
print('Initial forced occupied count:', sum(occupied_init))
