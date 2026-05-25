from ultralytics import YOLO
import os

model_path = os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'train-9', 'weights', 'best.pt')
img_path = os.path.join(os.path.dirname(__file__), 'moj_test.jpg')

print('Model path:', model_path)
print('Image path:', img_path)
print('Model file exists:', os.path.exists(model_path))
print('Image exists:', os.path.exists(img_path))

model = YOLO(model_path)
print('Model names:', model.names)

if not os.path.exists(img_path):
    raise SystemExit('Image missing')

# read image to get shape
import cv2
img = cv2.imread(img_path)
print('Image shape (H,W,C):', None if img is None else img.shape)

print('Running inference with conf=0.1, imgsz=640')
results = model(img_path, conf=0.1, imgsz=640)

for i, r in enumerate(results):
    print('Result', i)
    boxes = getattr(r, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        print('  No detections')
    else:
        xyxy = boxes.xyxy.tolist()
        confs = getattr(boxes, 'conf', None)
        clss = getattr(boxes, 'cls', None)
        for j, box in enumerate(xyxy):
            print(f'  Box {j}: cls={int(clss[j]) if clss is not None else None} conf={confs[j]:.3f} bbox={box}')

print('Saved plotted image to src/moj_test_pred_debug.jpg')
out = os.path.join(os.path.dirname(__file__), 'moj_test_pred_debug.jpg')
cv2.imwrite(out, results[0].plot())
