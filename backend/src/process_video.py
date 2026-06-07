import cv2
import json
import os
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from ultralytics import YOLO


def process_video(in_path, spots_json_path, model_path, output_path, conf=0.2, imgsz=960, margin=15,
                  iou=0.6, device='0', save_flag=False, progress_callback=None):
    print(f"Ładowanie głównego modelu YOLO (Detektor) z: {model_path}")
    detector = YOLO(model_path)

    # ==========================================
    # TUTAJ PODAJ ŚCIEŻKĘ DO NOWEGO MODELU
    # ==========================================
    KLASYFIKATOR_PATH = r"C:\Users\Mateusz\PycharmProjects\Parking_spot_detector\backend\src\klasyfikator.pt"
    print(f"Ładowanie modelu weryfikującego (Klasyfikator) z: {KLASYFIKATOR_PATH}")
    classifier = YOLO(KLASYFIKATOR_PATH)

    try:
        with open(spots_json_path, "r") as f:
            spots = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Nie znaleziono pliku JSON: {spots_json_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise Exception(f"Nie można otworzyć wideo: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    csv_path = os.path.join(base_dir, f"{base_name}_log.csv")
    plot_path = os.path.join(base_dir, f"{base_name}_plot.png")

    history_time_sec = []
    history_occupied = []
    frame_count = 0

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp (HH:MM:SS)', 'Wolne_miejsca', 'Zajete_miejsca'])

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback((frame_count / total_frames) * 100)

            liczba_zajetych = 0
            liczba_wolnych = 0

            for spot in spots:
                x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]

                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                if w > h:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

                # 1. ETAP: Główna detekcja (Twój stary model)
                results_det = detector.predict(source=crop, conf=conf, imgsz=imgsz, iou=iou, device=device,
                                               verbose=False)

                is_occupied = False
                for r in results_det:
                    for box in r.boxes:
                        if detector.names[int(box.cls[0])] == 'occupied':
                            is_occupied = True
                            break

                # 2. ETAP: Kaskada (Weryfikacja klasyfikatorem TYLKO gdy zajęte)
                if is_occupied:
                    # Klasyfikator sprawdza wycinek
                    results_cls = classifier.predict(source=crop, imgsz=128, verbose=False)

                    # YOLO-cls zwraca wyniki w obiekcie probs
                    top1_id = results_cls[0].probs.top1
                    top1_name = classifier.names[top1_id]

                    # Jeśli uzna, że to 'not_car' (np. cień), nadpisujemy wynik z pierwszego modelu!
                    if top1_name == 'not_car':
                        is_occupied = False

                if is_occupied:
                    liczba_zajetych += 1
                else:
                    liczba_wolnych += 1

                color = (0, 0, 255) if is_occupied else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 20), (x + 50, y), color, -1)
                cv2.putText(frame, f"ID: {spot['id']}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            2)

            # Rysowanie interfejsu
            current_sec = frame_count / fps
            timestamp_str = str(timedelta(seconds=int(current_sec)))
            csv_writer.writerow([timestamp_str, liczba_wolnych, liczba_zajetych])
            history_time_sec.append(current_sec)
            history_occupied.append(liczba_zajetych)

            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"WOLNE MIEJSCA: {liczba_wolnych}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2, cv2.LINE_AA)
            cv2.putText(frame, f"ZAJETE MIEJSCA: {liczba_zajetych}", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

            if liczba_wolnych == 0:
                cv2.rectangle(frame, (20, 130), (450, 180), (0, 0, 255), -1)
                cv2.putText(frame, "UWAGA: PARKING PELNY!", (35, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                            2, cv2.LINE_AA)

            out.write(frame)

    cap.release()
    out.release()

    # Generowanie wykresu
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history_time_sec, history_occupied, color='#e74c3c', linewidth=2)
        plt.fill_between(history_time_sec, history_occupied, color='#e74c3c', alpha=0.3)
        plt.title('Trend zajętości parkingu w czasie', fontsize=16)
        plt.xlabel('Czas wideo (sekundy)', fontsize=12)
        plt.ylabel('Liczba zajętych miejsc', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Błąd podczas generowania wykresu: {e}")

    return output_path