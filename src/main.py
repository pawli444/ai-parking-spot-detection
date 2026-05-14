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

# (import YOLO above) - avoid duplicate import

def main():
    
    model = YOLO('yolov8m.pt')


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'dataset', 'data.yaml')

    results = model.train(
        data=data_path,
        epochs=250,  
        patience=50,
       
        imgsz=640, 

        device=0,
        workers=4,

       
        batch=8,

       
        name='train-night-medium'
    )

#### aaaa
if __name__ == '__main__':
    main()
