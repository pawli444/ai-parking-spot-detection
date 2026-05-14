import os
import csv
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


def read_yolo_label(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c = float(parts[1]) * img_w
            y_c = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = x_c - w / 2.0
            y1 = y_c - h / 2.0
            x2 = x_c + w / 2.0
            y2 = y_c + h / 2.0
            boxes.append((cls, [x1, y1, x2, y2]))
    return boxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def evaluate(model_path='src/runs/detect/train-9/weights/best.pt',
             test_images_dir='dataset/test/images',
             test_labels_dir='dataset/test/labels',
             iou_th=0.5,
             positive_class_name='space-empty'):

    model = YOLO(model_path)
    names = model.names

    pos_cls = None
    for k, v in names.items():
        if v == positive_class_name:
            pos_cls = k
            break
    if pos_cls is None:
      
        pos_cls = 0

    image_files = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    total_TP = total_FP = total_FN = total_TN = 0
    per_image_rows = []

    for img_name in image_files:
        img_path = os.path.join(test_images_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(test_labels_dir, lbl_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = read_yolo_label(lbl_path, w, h)  

        
        results = model(img_path, conf=0.05, imgsz=1280)
        r = results[0]
        preds = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
            cls_preds = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, 'cpu') else np.array(r.boxes.cls)
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
            for i in range(len(xyxy)):
                preds.append((int(cls_preds[i]), xyxy[i].tolist(), float(confs[i])))

       
        ngt = len(gt_boxes)
        npred = len(preds)
        matched_pred = [-1] * ngt
        pred_assigned = [False] * npred
        pairs = []
        for i, (gcls, gbox) in enumerate(gt_boxes):
            for j, (pcls, pbox, conf) in enumerate(preds):
                pairs.append((iou(gbox, pbox), i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for score, i, j in pairs:
            if score < iou_th:
                break
            if matched_pred[i] == -1 and not pred_assigned[j]:
                matched_pred[i] = j
                pred_assigned[j] = True

        
        TP = FP = FN = TN = 0
        for i, (gcls, gbox) in enumerate(gt_boxes):
            midx = matched_pred[i]
            if gcls == pos_cls:
                if midx != -1 and preds[midx][0] == pos_cls:
                    TP += 1
                else:
                    FN += 1
            else:
                # ground truth negative (occupied)
                if midx != -1 and preds[midx][0] == pos_cls:
                    FP += 1
                else:
                    TN += 1

        # unmatched preds -> if predict positive -> FP
        for j, assigned in enumerate(pred_assigned):
            if not assigned:
                pcls = preds[j][0]
                if pcls == pos_cls:
                    FP += 1

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

        per_image_rows.append((img_name, TP, FP, FN, TN))

    # metrics
    P = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    R = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    total = total_TP + total_FP + total_FN + total_TN
    ACC = (total_TP + total_TN) / total if total > 0 else 0.0

    out_dir = os.path.join('runs', 'eval')
    os.makedirs(out_dir, exist_ok=True)

    # save per-image csv
    csv_path = os.path.join(out_dir, 'per_image_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'TP', 'FP', 'FN', 'TN'])
        for row in per_image_rows:
            writer.writerow(row)

    # save summary
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'TP={total_TP}\nFP={total_FP}\nFN={total_FN}\nTN={total_TN}\n')
        f.write(f'Precision={P:.4f}\nRecall={R:.4f}\nF1={F1:.4f}\nAccuracy={ACC:.4f}\n')

    print('Evaluation finished')
    print(f'TP={total_TP} FP={total_FP} FN={total_FN} TN={total_TN}')
    print(f'Precision={P:.4f} Recall={R:.4f} F1={F1:.4f} Accuracy={ACC:.4f}')

    # confusion matrix plot
    cm = np.array([[total_TP, total_FN], [total_FP, total_TN]])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Pos', 'Pred Neg'])
    ax.set_yticklabels(['GT Pos', 'GT Neg'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im)
    plt.title('Confusion matrix (TP,FN;FP,TN)')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()


if __name__ == '__main__':
    evaluate()
