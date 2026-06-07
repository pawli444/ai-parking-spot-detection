import cv2
import json
import os
import random
from ultralytics import YOLO

# ==========================================
# 1. NAPRAWA ŚCIEŻEK (Wbite na sztywno!)
# ==========================================
VIDEO_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\src\video_dwa.mp4"
JSON_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\miejsca_parkingowe.json"
MODEL_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\backend\src\runs\detect\doszkoleniev2\weights\best.pt"

# Tutaj powstaną foldery z pociętymi zdjęciami
base_dir = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\src\dataset_classifier"

print("Ładowanie modelu YOLO...")
model = YOLO(MODEL_PATH)

# Zabezpieczenie przed brakiem JSON-a
try:
    with open(JSON_PATH, "r") as f:
        spots = json.load(f)
except FileNotFoundError:
    print(f"❌ BŁĄD: Nie znaleziono pliku JSON dokładnie tutaj: {JSON_PATH}")
    exit()

# Tworzymy foldery
for split in ["train", "val"]:
    for cls in ["car", "not_car"]:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

# Zabezpieczenie przed brakiem WIDEO
if not cap.isOpened():
    print(f"❌ BŁĄD: OpenCV nie potrafi otworzyć wideo dokładnie stąd: {VIDEO_PATH}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

frame_count = 0
zapisane_zdjecia = 0

print(f"Znaleziono wideo. Zaczynamy ciąć klatki... (Zapis do folderu: {base_dir})")

# ==========================================
# 2. GŁÓWNA PĘTLA
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bierzemy klatkę co 1 sekundę
    if frame_count % fps == 0:
        for spot in spots:
            x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]

            margin = 15
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            if w > h:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            results = model.predict(source=crop, conf=0.40, verbose=False)

            is_occupied = False
            for r in results:
                for box in r.boxes:
                    if model.names[int(box.cls[0])] == 'occupied':
                        is_occupied = True
                        break

            split = "train" if random.random() < 0.8 else "val"
            katalog = "car" if is_occupied else "not_car"

            sciezka_zapisu = os.path.join(base_dir, split, katalog, f"spot_{spot['id']}_frame_{frame_count}.jpg")
            cv2.imwrite(sciezka_zapisu, crop)
            zapisane_zdjecia += 1

    frame_count += 1

cap.release()
print(f"✅ Gotowe! Wygenerowano {zapisane_zdjecia} zdjęć.")
print(f"Twoje zdjęcia leżą w: {base_dir}")
print("Teraz wejdź w folder 'train/car' i ręcznie usuń cienie i śmieci (tylko klawisz Delete!).")