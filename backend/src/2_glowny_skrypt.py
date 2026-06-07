import cv2
import json
from ultralytics import YOLO

# =========================================
# 1. Konfiguracja Ścieżek
# =========================================
VIDEO_PATH = "video_dwa.mp4"
JSON_PATH = "miejsca_parkingowe.json"

MODEL_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\backend\src\runs\detect\doszkoleniev2\weights\best.pt"

print("Ładowanie modelu (klasyfikator.pt)...")
model = YOLO(MODEL_PATH)

try:
    with open(JSON_PATH, "r") as f:
        spots = json.load(f)
except FileNotFoundError:
    print(f"")
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
        # Ignorujemy poziome auta z prawej strony (ID 15-22), bo dataset ich nie ogarnia
        # Ignorujemy TYLKO miejsca od 15 do 22 (poziome z prawej)
        # if 15 <= spot["id"] <= 22:
        #     continue

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

        # Puszczamy predykcję
        results = model.predict(source=crop, conf=0.40, imgsz=640, verbose=False)

        is_occupied = False

        # --- KLUCZOWA ZMIANA: Sprawdzamy CO dokładnie model znalazł ---
        for r in results:
            for box in r.boxes:
                # Pobieramy ID klasy (np. 0 lub 1)
                klasa_id = int(box.cls[0])
                # Tłumaczymy ID na nazwę (np. 'empty' albo 'occupied')
                nazwa_klasy = model.names[klasa_id]

                # Reagujemy tylko na klasę 'occupied'!
                if nazwa_klasy == 'occupied':
                    is_occupied = True
                    break

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