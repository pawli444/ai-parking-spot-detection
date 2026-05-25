import os
import re
import pandas as pd
import shutil

# Ścieżki (dostosowane do struktury repo)
CSV_DIR = "CNR-EXT_FULL_IMAGE_1000x750"
IMG_DIR = "CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750"
PATCH_DIR = "runs/sanity_check"

OUT_IMG = "CNR_EXT_YOLO/images"
OUT_LBL = "CNR_EXT_YOLO/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# Oryginalna rozdzielczość bboxów i docelowa
ORIG_W, ORIG_H = 2592, 1944
NEW_W, NEW_H = 1000, 750

scale_x = NEW_W / ORIG_W
scale_y = NEW_H / ORIG_H

# 1) Zbuduj mapę SlotId -> klasa (0/1) z PATCH_DIR (jeżeli dostępne)
slot_class = {}
if os.path.exists(PATCH_DIR):
    for root, dirs, files in os.walk(PATCH_DIR):
        for fn in files:
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            # spróbuj wyciągnąć id z nazwy pliku (ostatnia liczba o długości >=2)
            nums = re.findall(r"(\d{2,})", fn)
            if not nums:
                continue
            slot_id = int(nums[-1])
            cls = 1 if ('busy' in root.lower() or 'busy' in fn.lower() or 'occupied' in root.lower()) else 0
            slot_class[slot_id] = cls

# Jeśli nie znaleziono patchy, domyślnie użyj klasy 0
default_class = 0
if slot_class:
    print(f"Znaleziono mapę klas dla {len(slot_class)} slotów (z {PATCH_DIR})")
else:
    print(f"Nie znaleziono patchy w {PATCH_DIR} — wszystkie sloty będą klasy {default_class}")


def find_images_for_camera(camera_name):
    imgs = []
    for root, dirs, files in os.walk(IMG_DIR):
        # przyjmujemy, że foldery mają nazwy camera1, camera2 itp.
        if camera_name.lower() in os.path.basename(root).lower():
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    imgs.append(os.path.join(root, f))
    return sorted(imgs)


# 2) Przetwarzaj pliki CSV (każdy CSV odpowiada kamerze np. camera1.csv)
for csv_file in os.listdir(CSV_DIR):
    if not csv_file.endswith('.csv'):
        continue

    csv_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(csv_path)

    camera_name = os.path.splitext(csv_file)[0]  # camera1
    images = find_images_for_camera(camera_name)
    if not images:
        print(f"Brak obrazów dla {camera_name} — pomijam {csv_file}")
        continue

    # dla każdej klatki tej kamery zapisz plik label z wszystkimi slotami
    for img_path in images:
        if not os.path.exists(img_path):
            continue

        # skopiuj obraz do outputu (jeśli jeszcze nie skopiowano)
        out_img = os.path.join(OUT_IMG, os.path.basename(img_path))
        if not os.path.exists(out_img):
            shutil.copy(img_path, out_img)

        lbl_path = os.path.join(OUT_LBL, os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')

        lines = []
        for _, row in df.iterrows():
            try:
                slot = int(row['SlotId'])
                x = float(row['X'])
                y = float(row['Y'])
                w_box = float(row['W'])
                h_box = float(row['H'])
            except Exception:
                # oczekujemy kolumn SlotId,X,Y,W,H
                continue

            cls = slot_class.get(slot, default_class)

            xmin = x * scale_x
            ymin = y * scale_y
            xmax = (x + w_box) * scale_x
            ymax = (y + h_box) * scale_y

            x_center = (xmin + xmax) / 2.0 / NEW_W
            y_center = (ymin + ymax) / 2.0 / NEW_H
            w = (xmax - xmin) / NEW_W
            h = (ymax - ymin) / NEW_H

            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # nadpisz plik etykiet (unikniemy duplikatów przy ponownym uruchomieniu)
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))

    print(f"Zapisano etykiety dla {len(images)} obrazów kamery {camera_name}")

print('Połączono CSV + PATCHES → YOLOv8!')

