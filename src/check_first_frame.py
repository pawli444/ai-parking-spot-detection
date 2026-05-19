import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='video_dwa.mp4')
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--weights', default='src/runs/detect/train-9/weights/best.pt')
    p.add_argument('--conf', type=float, default=0.01)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--out-dir', default='runs/sanity_check')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / 'first_frame_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)

    if YOLO is None:
        raise SystemExit('Ultralytics not available')

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'failed to open video {args.video}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit('failed to read frame')

    first_png = out_dir / 'first_frame.png'
    cv2.imwrite(str(first_png), frame)
    print('Saved first frame to', first_png)

    model = YOLO(args.weights)
    res = model(frame, conf=args.conf, imgsz=args.imgsz)
    if len(res) == 0:
        print('No results from model')
        return
    r0 = res[0]
    # annotated image
    try:
        ann = r0.plot()
        ann_path = out_dir / 'first_frame_annotated.png'
        cv2.imwrite(str(ann_path), ann)
        print('Saved annotated image to', ann_path)
    except Exception as e:
        print('Failed to save annotated image', e)

    boxes = getattr(r0, 'boxes', None)
    if boxes is None:
        print('No boxes attribute in results')
        return
    count = 0
    for i, b in enumerate(boxes):
        try:
            conf = float(b.conf.cpu().numpy()[0])
            xy = b.xyxy.cpu().numpy()[0]
        except Exception:
            conf = float(getattr(b, 'conf', 0.0))
            xy = np.array(b.xyxy).astype(float)
        if conf < 0.1:
            continue
        x1, y1, x2, y2 = map(int, xy[:4])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        outcrop = crops_dir / f'crop_{i}_{conf:.2f}.png'
        cv2.imwrite(str(outcrop), crop)
        count += 1
    print(f'Saved {count} high-confidence crops to', crops_dir)


if __name__ == '__main__':
    main()
