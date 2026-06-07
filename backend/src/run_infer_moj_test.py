from ultralytics import YOLO
import os
import cv2


def run_grid_tests():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(repo_root, 'src', 'runs', 'detect', 'train-9', 'weights', 'klasyfikator.pt')
    image_path = os.path.join(os.path.dirname(__file__), 'moj_test.jpg')

    print('Model path:', model_path)
    print('Image path:', image_path)

    if not os.path.exists(model_path):
        print('ERROR: model file not found at', model_path)
        return
    if not os.path.exists(image_path):
        print('ERROR: image not found at', image_path)
        return

    model = YOLO(model_path)

    confs = [0.25, 0.1, 0.05]
    imgszs = [640, 960, 1280]

    summary = []
    for conf in confs:
        for imgsz in imgszs:
            print(f'\nRunning inference conf={conf} imgsz={imgsz} ...')
            try:
                results = model(image_path, conf=conf, imgsz=imgsz)
            except Exception as e:
                print('  Inference failed:', e)
                summary.append((conf, imgsz, 'error'))
                continue

            r = results[0]
            boxes = getattr(r, 'boxes', None)
            count = 0 if boxes is None else len(boxes)
            print(f'  Detections: {count}')

            # save plotted image with params in filename
            out_name = f'moj_test_pred_conf{str(conf).replace(".","")}_img{imgsz}.jpg'
            out_path = os.path.join(os.path.dirname(__file__), out_name)
            cv2.imwrite(out_path, r.plot())
            print('  Saved:', out_path)
            summary.append((conf, imgsz, count))

    print('\nSummary:')
    for conf, imgsz, count in summary:
        print(f' conf={conf:0.3f} imgsz={imgsz}: {count}')


if __name__ == '__main__':
    run_grid_tests()
