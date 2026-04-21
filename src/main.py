from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='C:/Users/Mateusz/PycharmProjects/Parking_spot_detector/ai-parking-spot-detection/dataset/data.yaml',
        epochs=64,
        imgsz=640,
        device=0,        # <--- TO ZMUSZA YOLO DO UŻYCIA KARTY NVIDIA
        workers=4,
        batch=16
    )

if __name__ == '__main__':
    main()