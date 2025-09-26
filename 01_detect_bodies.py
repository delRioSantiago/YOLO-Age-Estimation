import os
import  csv
import cv2
import yaml
from ultralytics import YOLO

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

IN_PATH   = cfg["paths"]["images"]
OUT_BODIES_PATH = cfg["paths"]["out_bodies"]
CSV_PATH  = cfg["paths"]["tables"]
os.makedirs(OUT_BODIES_PATH, exist_ok=True); os.makedirs(CSV_PATH, exist_ok=True)

model = YOLO(cfg["detect"]["yolo_body"])
IMG_SIZE = cfg["detect"]["img_size_body"]
CONF_THRES = min(cfg["detect"]["body_conf_thres"])  

csv_path = os.path.join(CSV_PATH, "bodies.csv")

with open(csv_path, "a", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    schema_bodies = cfg["schema"]["bodies"]
    csv_writer.writerow(schema_bodies)
    for filename in sorted(os.listdir(IN_PATH)):
        if not filename.lower().endswith((".jpg",".jpeg",".png")): 
            continue
        image_id = os.path.splitext(filename)[0]
        path = os.path.join(IN_PATH, filename)
        img = cv2.imread(path); 
        if img is None: 
            continue
        H, W = img.shape[:2]

        result = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)[0]
        body_id = 0
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            w,h = max(0,x2-x1), max(0,y2-y1)
            if w<=0 or h<=0: 
                continue
            conf = float(box.conf[0])

            crop = img[y1:y1+h, x1:x1+w]
            out_name = f"{image_id}_body{body_id}.jpg"
            out_path = os.path.join(OUT_BODIES_PATH, out_name)
            cv2.imwrite(out_path, crop)

            csv_writer.writerow([image_id, body_id, f"{conf:.6f}", out_path, cfg["detect"]["yolo_body"]])
            body_id += 1
