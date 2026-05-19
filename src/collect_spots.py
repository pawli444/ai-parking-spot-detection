import argparse
import json
from pathlib import Path
import cv2

def collect(video_path, out_path, frame_idx=0, save_preview=False):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Failed to read frame")
    boxes = []
    current = None
    drawing = False

    disp = frame.copy()

    def draw_all(img, boxes, current_box=None):
        out = img.copy()
        for b in boxes:
            cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
        if current_box is not None:
            x1, y1, x2, y2 = current_box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 200), 1)
        # instructions
        cv2.putText(out, "Drag to draw. Press ESC to finish and save.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
        cv2.putText(out, "Press u to undo last box, c to cancel and exit without save.", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return out

    start_pt = None

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, start_pt, current, disp
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x0, y0 = start_pt
            current = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = start_pt
            box = {"x1": int(min(x0, x)), "y1": int(min(y0, y)), "x2": int(max(x0, x)), "y2": int(max(y0, y))}
            # ignore tiny boxes
            if box["x2"] - box["x1"] > 5 and box["y2"] - box["y1"] > 5:
                boxes.append(box)
            current = None

    win = "Select ROIs (draw, ESC to finish)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        disp = draw_all(frame, boxes, current)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC -> finish and save
            break
        elif key == ord('u'):
            if boxes:
                boxes.pop()
        elif key == ord('c'):
            boxes = []
            break

    cv2.destroyWindow(win)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(boxes, f, indent=2)

    if save_preview and boxes:
        pv = frame.copy()
        for b in boxes:
            cv2.rectangle(pv, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0,255,0), 2)
        cv2.imwrite(str(out_path.with_suffix(".png")), pv)

    print(f"Saved {len(boxes)} boxes -> {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="spots.json")
    p.add_argument("--frame", type=int, default=0, help="Frame index to annotate")
    p.add_argument("--preview", action="store_true", help="Save preview image with boxes")
    args = p.parse_args()
    collect(args.video, args.out, args.frame, args.preview)

if __name__ == "__main__":
    main()