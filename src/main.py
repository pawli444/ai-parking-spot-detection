from ultralytics import YOLO
import cv2
import os


# def main():
#     model_path = 'runs/detect/train-9/weights/best.pt'
#     model = YOLO(model_path)
#     test_images_path = 'C:/Users/Mateusz/PycharmProjects/Parking_spot_detector/ai-parking-spot-detection/dataset/test/images/'
#     pliki = os.listdir(test_images_path)
#     if not pliki:
#         print("Folder ze zdjęciami jest pusty!")
#         return
#
#     wybrane_zdjecie = pliki[1]
#     pelna_sciezka = os.path.join(test_images_path, wybrane_zdjecie)
#
#     print(f"Testuję model na zdjęciu: {wybrane_zdjecie}")
#
#     #
#     results = model.predict(source=pelna_sciezka, conf=0.5)
#
#     for r in results:
#         obraz_z_ramkami = r.plot()
#         cv2.imshow("parking", obraz_z_ramkami)
#
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

from ultralytics import YOLO


def main():
    # 1. Zmieniamy model bazowy na Medium (pobierze się automatycznie ok. 50MB)
    model = YOLO('yolov8m.pt')

    # 2. Odpalamy ciężki trening na noc
    results = model.train(
        data='C:/Users/Mateusz/PycharmProjects/Parking_spot_detector/ai-parking-spot-detection/dataset/data.yaml',

        epochs=250,  # Dajemy mu ogromny zapas czasu
        patience=50,
        # EARLY STOPPING: Jeśli przez 50 epok z rzędu model nie zanotuje żadnej poprawy, sam się wyłączy, żeby nie marnować prądu

        imgsz=640,  # Zostawiamy 640. Większa rozdzielczość przy modelu Medium mogłaby wywalić błąd "CUDA Out of Memory"

        device=0,
        workers=4,

        # UWAGA: Zmniejszamy batch z 16 na 8! Model 'm' jest ogromny i zjada dużo więcej pamięci VRAM na karcie graficznej.
        batch=8,

        # Nazywamy folder ładnie, żeby rano łatwo go było znaleźć w folderze 'runs'
        name='train-night-medium'
    )

#### aaaa
if __name__ == '__main__':
    main()
