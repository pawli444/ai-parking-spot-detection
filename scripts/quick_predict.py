import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('src/runs/detect/train-9/weights/best.pt')
video_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('NEWVIDEO.mp4')
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 50

print('Model path:', model_path)
if not model_path.exists():
    print('Model file not found:', model_path)
    raise SystemExit(1)

model = YOLO(str(model_path))
print('model.names:', model.names)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print('Failed to open video:', video_path)
    raise SystemExit(2)

frame_idx = 0
total_dets = 0
while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=conf, imgsz=640, device='cpu')[0]
    dets = len(results.boxes)
    print(f'Frame {frame_idx}: {dets} detections')
    total_dets += dets
    frame_idx += 1

cap.release()
print(f'Total detections in {frame_idx} frames: {total_dets}')
