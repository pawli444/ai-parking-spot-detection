from ultralytics import YOLO
import cv2

model = YOLO('src/runs/detect/train-9/weights/best.pt')
cap = cv2.VideoCapture('NEWVIDEO.mp4')
ret, frame = cap.read()
cap.release()
cv2.imwrite('moj_frame.jpg', frame)
results = model('moj_frame.jpg', conf=0.1)
print('detekcji:', len(results[0].boxes))
results[0].save('moj_wynik.jpg')