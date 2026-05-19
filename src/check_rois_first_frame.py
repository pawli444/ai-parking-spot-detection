import argparse
import json
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def letterbox_image(img, size=640):
    h0, w0 = img.shape[:2]
    r = min(size / w0, size / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    img_resized = cv2.resize(img, (new_w, new_h))
    pad_w = size - new_w
    pad_h = size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, r, pad_left, pad_top


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='video_dwa.mp4')
    p.add_argument('--spots', default='spotsVIDEO_DWA_manual.json')
    p.add_argument('--weights', default='src/runs/detect/train-9/weights/best.pt')
    p.add_argument('--conf', type=float, default=0.01)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--out-dir', default='runs/sanity_check/rois')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if YOLO is None:
        raise SystemExit('Ultralytics not installed')

    spots_path = Path(args.spots)
    if not spots_path.exists():
        raise SystemExit('spots file not found')
    with open(spots_path, 'r', encoding='utf-8') as f:
        spots = json.load(f)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit('failed to open video')
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit('failed to read frame')

    model = YOLO(args.weights)
    names = {}
    try:
        names = model.model.names
    except Exception:
        names = {}

    report = []
    for i, s in enumerate(spots):
        x1, y1, x2, y2 = int(s['x1']), int(s['y1']), int(s['x2']), int(s['y2'])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            report.append({'idx': i, 'status': 'empty_crop'})
            continue
        padded, scale, pad_x, pad_y = letterbox_image(crop, args.imgsz)
        res = model(padded, conf=args.conf, imgsz=args.imgsz)
        dets = []
        if len(res) > 0:
            r0 = res[0]
            boxes = getattr(r0, 'boxes', None)
            if boxes is not None:
                for b in boxes:
                    try:
                        xy = b.xyxy.cpu().numpy()[0]
                    except Exception:
                        xy = np.array(b.xyxy).astype(float)
                    try:
                        confv = float(b.conf.cpu().numpy()[0])
                    except Exception:
                        confv = float(getattr(b, 'conf', 0.0))
                    try:
                        cls = int(b.cls.cpu().numpy()[0])
                    except Exception:
                        cls = int(getattr(b, 'cls', 0))
                    # map back approx to crop coords
                    x1b = int((xy[0] - pad_x) / scale)
                    y1b = int((xy[1] - pad_y) / scale)
                    x2b = int((xy[2] - pad_x) / scale)
                    y2b = int((xy[3] - pad_y) / scale)
                    dets.append({'box': [x1b, y1b, x2b, y2b], 'conf': confv, 'cls': cls, 'name': names.get(cls, str(cls))})
        # decide occupancy: if any det with name containing 'occup' or cls==1
        occ = False
        best = None
        for d in dets:
            if best is None or d['conf'] > best['conf']:
                best = d
            if 'occup' in d['name'] or 'occupied' in d['name']:
                occ = True
        # save annotated crop
        ann = crop.copy()
        if best is not None:
            bx = best['box']
            cv2.rectangle(ann, (bx[0], bx[1]), (bx[2], bx[3]), (0,0,255), 2)
            cv2.putText(ann, f"{best['name']} {best['conf']:.2f}", (max(0,bx[0]), max(12,bx[1]-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        outp = out_dir / f'roi_{i}_{"occ" if occ else "free"}.png'
        cv2.imwrite(str(outp), ann)
        report.append({'idx': i, 'occupied': occ, 'best': best})

    rep_path = out_dir / 'rois_report.json'
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Saved', rep_path)


if __name__ == '__main__':
    main()
