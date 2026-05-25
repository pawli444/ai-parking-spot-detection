import cv2
import json

VIDEO_PATH = "video_dwa.mp4"
JSON_PATH = "miejsca_parkingowe.json"

# Wczytujemy pierwszą klatkę z wideo, żebyś mógł na niej zaznaczyć miejsca
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"❌ BŁĄD: Nie można wczytać pliku {VIDEO_PATH}. Upewnij się, że jest w tym samym folderze!")
    exit()

print("--- INSTRUKCJA ---")
print("1. Zaznacz miejsce myszką.")
print("2. Wciśnij SPACJĘ.")
print("3. Zaznaczaj kolejne i wciskaj SPACJĘ.")
print("4. Na koniec wciśnij ESC.")

# Narzędzie do wyklikiwania
rois = cv2.selectROIs("Zaznacz miejsca (ESC by wyjsc)", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Zapis do JSON
miejsca = [{"id": i + 1, "x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])} for i, r in enumerate(rois)]

with open(JSON_PATH, "w") as f:
    json.dump(miejsca, f, indent=4)

print(f"✅ GOTOWE! Zapisano {len(miejsca)} miejsc do pliku {JSON_PATH}.")