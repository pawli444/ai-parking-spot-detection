import cv2
import json
from ultralytics import YOLO

# =========================================
# KONFIGURACJA
# =========================================
VIDEO_PATH = "video_dwa.mp4"
JSON_PATH = "miejsca_parkingowe.json"
MODEL_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\src\runs\detect\doszkoleniev3\weights\best.pt"

print("Ładowanie modelu w trybie DEBUG...")
model = YOLO(MODEL_PATH)

try:
    with open(JSON_PATH, "r") as f:
        spots = json.load(f)
except FileNotFoundError:
    print("Brak pliku JSON.")
    exit()

# Bierzemy tylko JEDNĄ klatkę do analizy
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Błąd ładowania wideo.")
    exit()

print("Analizowanie miejsc... (to może chwilę potrwać)")

for spot in spots:
    x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]

    margin = 25
    y1 = max(0, y - margin)
    y2 = min(frame.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(frame.shape[1], x + w + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: continue

    # UWAGA: Ustawiamy conf na 1% (0.01), żeby model wypluł dosłownie WSZYSTKO co myśli
    results = model.predict(source=crop, conf=0.01, imgsz=640, verbose=False)

    # Szukamy NAJWYŻSZEGO wyniku z tego konkretnego wycinka
    max_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf

    # Kolorujemy dynamicznie:
    # > 50% = Czerwony
    # 20% - 50% = Pomarańczowy (strefa niepewności)
    # < 20% = Zielony
    if max_conf > 0.50:
        color = (0, 0, 255)
    elif max_conf > 0.20:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Dopisujemy dokładny wynik pewności (np. "0.34") na obrazku
    tekst = f"ID:{spot['id']} conf:{max_conf:.2f}"

    cv2.rectangle(frame, (x, y - 20), (x + 110, y), color, -1)
    cv2.putText(frame, tekst, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

# Zmniejszamy okno, żeby zmieściło się na ekranie (opcjonalne)
frame_resized = cv2.resize(frame, (1280, 720))

cv2.imshow("DEBUG MODE - Rentgen Modelu", frame_resized)
print("Gotowe! Wciśnij dowolny klawisz, aby zamknąć.")
cv2.waitKey(0)
cv2.destroyAllWindows()