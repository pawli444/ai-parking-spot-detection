import cv2
import json
import os
import csv
import matplotlib

# Ustawienie trybu 'Agg' jest krytyczne dla aplikacji webowych/wątków w tle,
# aby matplotlib nie próbował otwierać okien graficznych.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta
from ultralytics import YOLO


def process_video(in_path, spots_json_path, model_path, output_path, conf=0.2, imgsz=960, margin=15,
                  iou=0.6, device='0', save_flag=False, progress_callback=None):
    print(f"Ładowanie modelu YOLO z: {model_path} (device: {device})")
    model = YOLO(model_path)

    try:
        with open(spots_json_path, "r") as f:
            spots = json.load(f)
    except FileNotFoundError:
        raise Exception(f"Nie znaleziono pliku JSON z miejscami: {spots_json_path}")

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise Exception(f"Nie można otworzyć wideo: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Przygotowanie ścieżek dla logów i wykresu
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    csv_path = os.path.join(base_dir, f"{base_name}_log.csv")
    plot_path = os.path.join(base_dir, f"{base_name}_plot.png")

    # Listy do przechowywania danych na potrzeby wykresu
    history_time_sec = []
    history_occupied = []

    frame_count = 0

    # Otwieramy plik CSV i zapisujemy nagłówki
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp (HH:MM:SS)', 'Wolne_miejsca', 'Zajete_miejsca'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if progress_callback and total_frames > 0:
                procent = (frame_count / total_frames) * 100
                progress_callback(procent)

            liczba_zajetych = 0
            liczba_wolnych = 0

            for spot in spots:
                x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]

                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if w > h:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

                # Predykcja z użyciem nowych parametrów
                results = model.predict(source=crop, conf=conf, imgsz=imgsz, iou=iou, device=device, save=save_flag,
                                        verbose=False)

                najwyzsza_pewnosc = 0.0
                najlepsza_klasa = 'empty'

                for r in results:
                    for box in r.boxes:
                        c = float(box.conf[0])
                        klasa_id = int(box.cls[0])
                        nazwa_klasy = model.names[klasa_id]

                        if c > najwyzsza_pewnosc:
                            najwyzsza_pewnosc = c
                            najlepsza_klasa = nazwa_klasy

                is_occupied = (najlepsza_klasa == 'occupied')

                if is_occupied:
                    liczba_zajetych += 1
                else:
                    liczba_wolnych += 1

                color = (0, 0, 255) if is_occupied else (0, 255, 0)
                x_end = x + w
                y_end = y + h

                cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)
                cv2.rectangle(frame, (x, y - 20), (x + 50, y), color, -1)
                cv2.putText(frame, f"ID: {spot['id']}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- ZAPIS DO LOGÓW (CSV) i ZBIERANIE DANYCH DO WYKRESU ---
            current_sec = frame_count / fps
            timestamp_str = str(timedelta(seconds=int(current_sec)))  # format np. 0:00:15

            csv_writer.writerow([timestamp_str, liczba_wolnych, liczba_zajetych])

            # Zbieramy dane co klatkę (lub można zrobić np. co fps klatek dla mniejszego pliku)
            history_time_sec.append(current_sec)
            history_occupied.append(liczba_zajetych)

            # --- Półprzezroczysty panel z wynikami ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (350, 110), (0, 0, 0), -1)

            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.putText(frame, f"WOLNE MIEJSCA: {liczba_wolnych}", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ZAJETE MIEJSCA: {liczba_zajetych}", (35, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # --- ALERT TEKSTOWY GDY PARKING JEST PEŁNY ---
            if liczba_wolnych == 0:
                # Tło dla lepszej czytelności alertu
                cv2.rectangle(frame, (20, 130), (450, 180), (0, 0, 255), -1)
                cv2.putText(frame, "UWAGA: PARKING PELNY!", (35, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)

    cap.release()
    out.release()

    # --- GENEROWANIE WYKRESU TRENDU ZAJĘTOŚCI ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history_time_sec, history_occupied, color='#e74c3c', linewidth=2)
        plt.fill_between(history_time_sec, history_occupied, color='#e74c3c', alpha=0.3)  # Lekkie wypełnienie pod linią
        plt.title('Trend zajętości parkingu w czasie', fontsize=16)
        plt.xlabel('Czas wideo (sekundy)', fontsize=12)
        plt.ylabel('Liczba zajętych miejsc', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(plot_path)
        plt.close()
        print(f"Wykres zapisano w: {plot_path}")
    except Exception as e:
        print(f"Błąd podczas generowania wykresu: {e}")

    return output_path