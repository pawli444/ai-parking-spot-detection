import argparse
from pathlib import Path
import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def run_on_tile(model, tile_img, x_off, y_off, conf, imgsz):
    dets = []
    res = model(tile_img, conf=conf, imgsz=imgsz)
    if len(res) == 0:
        return dets
    r0 = res[0]
    boxes = getattr(r0, 'boxes', None)
    if boxes is None:
        return dets
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
        # shift back to original coords
        xyxy[0] += x_off
        xyxy[1] += y_off
        xyxy[2] += x_off
        xyxy[3] += y_off
        dets.append((xyxy, confv, cls))
    return dets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='video_dwa.mp4')
    p.add_argument('--weights', default='src/runs/detect/train-9/weights/best.pt')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.01)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--tiles', type=int, default=1, help='split frame into NxN tiles')
    p.add_argument('--pad', action='store_true', help='Pad/letterbox tiles or frame to square imgsz before inference')
    p.add_argument('--out-img', default='debug_detect_out.jpg')
    args = p.parse_args()

    if YOLO is None:
        raise SystemExit('Ultralytics not available')

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('failed to open video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit('failed to read frame')

    h, w = frame.shape[:2]
    dets = []
    tiles = max(1, args.tiles)
    def letterbox(img, new_size):
        h0, w0 = img.shape[:2]
        r = min(new_size / w0, new_size / h0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        img_resized = cv2.resize(img, (new_w, new_h))
        pad_w = new_size - new_w
        pad_h = new_size - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114,114,114))
        return img_padded, r, pad_left, pad_top

    if tiles == 1:
        try:
            if args.pad:
                img_pad, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
                res = model(img_pad, conf=args.conf, imgsz=args.imgsz)
                if len(res) > 0:
                    r0 = res[0]
                    boxes = getattr(r0, 'boxes', None)
                    if boxes is not None:
                        for b in boxes:
                            try:
                                xyxy = b.xyxy.cpu().numpy()[0]
                            except Exception:
                                xyxy = np.array(b.xyxy).astype(float)
                            # map back to original frame coords
                            xyxy[0] = (xyxy[0] - pad_x) / scale
                            xyxy[1] = (xyxy[1] - pad_y) / scale
                            xyxy[2] = (xyxy[2] - pad_x) / scale
                            xyxy[3] = (xyxy[3] - pad_y) / scale
                            try:
                                confv = float(b.conf.cpu().numpy()[0])
                            except Exception:
                                confv = float(getattr(b, 'conf', 0.0))
                            try:
                                cls = int(b.cls.cpu().numpy()[0])
                            except Exception:
                                cls = int(getattr(b, 'cls', 0))
                            dets.append((xyxy, confv, cls))
            else:
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
                            dets.append((xyxy, confv, cls))
        except Exception as e:
            print('model inference error:', e)
    else:
        # split into NxN tiles without overlap
        h_frame, w_frame = h, w
        tw = (w_frame + tiles - 1) // tiles
        th = (h_frame + tiles - 1) // tiles
        for yi in range(tiles):
            for xi in range(tiles):
                x1 = xi * tw
                y1 = yi * th
                x2 = min(w_frame, (xi + 1) * tw)
                y2 = min(h_frame, (yi + 1) * th)
                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                try:
                    if args.pad:
                        tile_pad, scale, pad_x, pad_y = letterbox(tile, args.imgsz)
                        res = model(tile_pad, conf=args.conf, imgsz=args.imgsz)
                    else:
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
                    if args.pad:
                        # map from padded coords back to tile coords
                        xyxy[0] = (xyxy[0] - pad_x) / scale
                        xyxy[1] = (xyxy[1] - pad_y) / scale
                        xyxy[2] = (xyxy[2] - pad_x) / scale
                        xyxy[3] = (xyxy[3] - pad_y) / scale
                    # shift to frame coords
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
                    dets.append((xyxy, confv, cls))

    out = frame.copy()
    names = None
    try:
        names = model.model.names
    except Exception:
        names = {}

    print('Total detections:', len(dets))
    for i, (xyxy, confv, cls) in enumerate(dets):
        x1, y1, x2, y2 = map(int, xyxy[:4])
        label = f'{names.get(cls,cls)} {confv:.2f}'
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(out, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        print(i, 'cls', cls, 'conf', confv, 'box', x1, y1, x2, y2)

    cv2.imwrite(args.out_img, out)
    print('Saved', args.out_img)


if __name__ == '__main__':
    main()
