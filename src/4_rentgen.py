import cv2
import json
from ultralytics import YOLO

# =========================================
# 1. Konfiguracja Ścieżek
# =========================================
VIDEO_PATH = "video_dwa.mp4"
JSON_PATH = "miejsca_parkingowe.json"

# Podpinamy Twój nowy, odporny na cienie model V4
MODEL_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\src\runs\detect\doszkoleniev4\weights\best.pt"

print("Ładowanie modelu (best.pt z V4)...")
model = YOLO(MODEL_PATH)

try:
    with open(JSON_PATH, "r") as f:
        spots = json.load(f)
except FileNotFoundError:
    print(f"❌ Błąd: Nie znaleziono {JSON_PATH}. Wyklikaj najpierw miejsca!")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

print("Wideo działa. Wciśnij klawisz 'q' w oknie z filmem, aby wyłączyć.")

# =========================================
# 2. Główna pętla wideo
# =========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Koniec wideo.")
        break

    for spot in spots:
        x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]

        # Marginesy dla kontekstu
        margin = 15
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # --- HACK INŻYNIERYJNY: Obracamy poziome miejsca o 90 stopni ---
        # Dzięki temu Twój model V4 dostaje auto w pionie (tak jak lubi najbardziej)
        if w > h:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        # Puszczamy predykcję (conf=0.25 bo model V4 lepiej radzi sobie z cieniami)
        results = model.predict(source=crop, conf=0.25, imgsz=640, verbose=False)

        # --- ZASADA NAJWYŻSZEJ PEWNOŚCI ---
        najwyzsza_pewnosc = 0.0
        najlepsza_klasa = 'empty' # Domyślnie zakładamy, że wolne

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                klasa_id = int(box.cls[0])
                nazwa_klasy = model.names[klasa_id]

                # Bierzemy to, czego model jest najbardziej pewny
                if conf > najwyzsza_pewnosc:
                    najwyzsza_pewnosc = conf
                    najlepsza_klasa = nazwa_klasy

        # Miejsce jest czerwone tylko, gdy wygrała klasa 'occupied'
        is_occupied = (najlepsza_klasa == 'occupied')

        # =========================================
        # 4. Rysowanie wyników
        # =========================================
        color = (0, 0, 255) if is_occupied else (0, 255, 0)

        x_end = x + w
        y_end = y + h
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)

        cv2.rectangle(frame, (x, y - 20), (x + 50, y), color, -1)
        cv2.putText(frame, f"ID: {spot['id']}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("System Monitorowania Parkingu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()