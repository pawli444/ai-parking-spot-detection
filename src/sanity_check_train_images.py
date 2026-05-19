import random
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

ROOT = Path(__file__).resolve().parents[1]
img_dir = ROOT / 'dataset' / 'train' / 'images'
out_dir = ROOT / 'runs' / 'sanity_check'
out_dir.mkdir(parents=True, exist_ok=True)

weights = str(ROOT / 'src' / 'runs' / 'detect' / 'train-9' / 'weights' / 'best.pt')
if YOLO is None:
    raise SystemExit('Ultralytics not installed')

model = YOLO(weights)

imgs = list(img_dir.glob('**/*'))
imgs = [p for p in imgs if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
if len(imgs) == 0:
    raise SystemExit(f'no images in {img_dir}')

sample = random.sample(imgs, min(3, len(imgs)))
print('Selected images:', sample)

for p in sample:
    img = cv2.imread(str(p))
    if img is None:
        print('failed read', p)
        continue
    try:
        res = model(img, conf=0.01, imgsz=1280)
    except Exception as e:
        print('model inference error on', p, e)
        continue
    if len(res) == 0:
        print(p.name, '=> no results')
        continue
    r0 = res[0]
    boxes = getattr(r0, 'boxes', None)
    n = 0 if boxes is None else len(boxes)
    print(p.name, 'detections:', n)
    try:
        out_img = r0.plot()  # annotated image
        out_path = out_dir / f'annot_{p.name}'
        cv2.imwrite(str(out_path), out_img)
    except Exception as e:
        print('failed to save annotated image', e)
    # save crops of high-confidence detections
    if boxes is not None:
        for i, b in enumerate(boxes):
            try:
                conf = float(b.conf.cpu().numpy()[0])
                cls = int(b.cls.cpu().numpy()[0])
                xy = b.xyxy.cpu().numpy()[0]
            except Exception:
                try:
                    conf = float(getattr(b, 'conf', 0.0))
                    cls = int(getattr(b, 'cls', 0))
                    xy = np.array(b.xyxy).astype(float)
                except Exception:
                    continue
            if conf < 0.1:
                continue
            x1, y1, x2, y2 = map(int, xy[:4])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cv2.imwrite(str(out_dir / f'crop_{p.stem}_{i}_{conf:.2f}.jpg'), crop)

print('Saved annotated images and crops to', out_dir)
