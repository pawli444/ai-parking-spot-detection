import argparse
from pathlib import Path
import cv2
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Run model on video and save annotated output and crops')
parser.add_argument('--model', default='src/runs/detect/train-9/weights/best.pt')
parser.add_argument('--video', default='video_dwa.mp4')
parser.add_argument('--out', default='NEWVIDEO_simple.mp4')
parser.add_argument('--out_dir', default='detection_frames')
parser.add_argument('--conf', type=float, default=0.01)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--max_frames', type=int, default=None)
args = parser.parse_args()

model_path = Path(args.model)
video_path = Path(args.video)

print('Model:', model_path)
print('Video:', video_path)

if not model_path.exists():
    raise SystemExit(f'model not found: {model_path}')
if not video_path.exists():
    raise SystemExit(f'video not found: {video_path}')

model = YOLO(str(model_path))
print('model.names:', model.names)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit('failed to open video')

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

os.makedirs(args.out_dir, exist_ok=True)

frame_idx = 0
total_dets = 0
max_frames = args.max_frames

while True:
    if max_frames is not None and frame_idx >= max_frames:
        break
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=args.conf, imgsz=args.imgsz)[0]
    dets = results.boxes
    n = len(dets)
    print(f'Frame {frame_idx}: {n} detections')
    for j, box in enumerate(dets):
        try:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        except Exception:
            continue
        label = model.names.get(cls, str(cls)) if isinstance(model.names, dict) else (model.names[cls] if cls < len(model.names) else str(cls))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # save crop
        crop = frame[y1:y2, x1:x2]
        if crop.size != 0:
            crop_path = Path(args.out_dir) / f'frame{frame_idx:04d}_det{j}.jpg'
            cv2.imwrite(str(crop_path), crop)
    total_dets += n
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f'Processed {frame_idx} frames, total detections: {total_dets}')
print('Annotated video saved to', args.out)
print('Crops saved to', args.out_dir)
