import os
import json
import cv2
from ultralytics import YOLO


def process_video(input_path, spots_json_path=None, model_path=None, output_path=None, conf=0.4, margin=15, progress_callback=None):
    if spots_json_path is None:
        spots_json_path = os.path.join(os.path.dirname(__file__), 'miejsca_parkingowe.json')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if model_path is None:
        model_path = os.path.join(project_root, 'yolov8n.pt')

    if output_path is None:
        base, _ = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.path.dirname(input_path), base + '_out.mp4')

    if not os.path.exists(spots_json_path):
        raise FileNotFoundError(f"Spots JSON not found: {spots_json_path}")

    with open(spots_json_path, 'r', encoding='utf-8') as f:
        spots = json.load(f)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for spot in spots:
            if 15 <= spot.get('id', -1) <= 22:
                continue

            x = int(spot['x'])
            y = int(spot['y'])
            w = int(spot['w'])
            h = int(spot['h'])

            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            results = model.predict(source=crop, conf=conf, imgsz=640, verbose=False)

            is_occupied = False
            for r in results:
                for box in r.boxes:
                    klasa_id = int(box.cls[0])
                    nazwa_klasy = model.names.get(klasa_id, str(klasa_id))
                    if nazwa_klasy == 'occupied':
                        is_occupied = True
                        break
                if is_occupied:
                    break

            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            x_end = x + w
            y_end = y + h
            cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)
            cv2.rectangle(frame, (x, y - 20), (x + 50, y), color, -1)
            cv2.putText(frame, f"ID: {spot['id']}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        writer.write(frame)

        frame_idx += 1
        # report progress if callback provided
        if progress_callback is not None:
            try:
                if total_frames > 0:
                    pct = int((frame_idx / total_frames) * 100)
                else:
                    # unknown total, report periodic progress
                    pct = min(100, frame_idx)
                progress_callback(min(pct, 100))
            except Exception:
                pass

    cap.release()
    writer.release()

    return output_path
