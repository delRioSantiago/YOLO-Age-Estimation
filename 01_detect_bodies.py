import os 
import  csv
import cv2 # pip install opencv-python
import yaml # pip install pyyaml
import numpy as np # pip install numpy
import time
from ultralytics import YOLO # pip install ultralytics


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_DIR = cfg["paths"]["project"]
IN_PATH = os.path.join(PROJECT_DIR, cfg["paths"]["images"])
OUT_BODIES_PATH = os.path.join(PROJECT_DIR, cfg["paths"]["bodies"])
CSV_PATH = os.path.join(PROJECT_DIR, cfg["paths"]["tables"])
os.makedirs(OUT_BODIES_PATH, exist_ok=True)
os.makedirs(CSV_PATH, exist_ok=True)

model = YOLO(cfg["detect"]["yolo_body"])
IMG_SIZE = cfg["detect"]["img_size_body"]
CONF_THRES = cfg["detect"]["body_conf_thres"]  

csv_path = os.path.join(CSV_PATH, "bodies.csv")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW, interH = max(0, xB-xA), max(0, yB-yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

IOU_THRESHOLD = cfg["detect"]["nms_iou_thres"]
counter = 0
start_time = time.time()
with open(csv_path, "a", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    schema_bodies = cfg["schema"]["bodies"]
    csv_writer.writerow(schema_bodies)
    for filename in sorted(os.listdir(IN_PATH)):
        counter += 1
        if counter % 500 == 0:
            print(f"Verarbeite Bild {counter}: {filename}")
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_id = os.path.splitext(filename)[0]
        path = os.path.join(IN_PATH, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        result = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRES,
                               classes=[0], verbose=False)[0]

        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((conf, [x1, y1, x2, y2]))

        boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
        kept = []
        for conf, box in boxes:
            if all(iou(box, k[1]) < IOU_THRESHOLD for k in kept):
                kept.append((conf, box))

        for body_id, (conf, (x1, y1, x2, y2)) in enumerate(kept):
            crop = img[y1:y2, x1:x2]
            out_name = f"{image_id}_body{body_id}.jpg"
            out_path = os.path.join(OUT_BODIES_PATH, out_name)
            cv2.imwrite(out_path, crop)
            csv_writer.writerow([image_id, body_id, f"{conf:.6f}", out_name,
                                 cfg["detect"]["yolo_body"]])

end_time = time.time()
print(f"Verarbeitungszeit: {end_time - start_time:.2f} Sekunden")
