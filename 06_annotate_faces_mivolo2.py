#!/usr/bin/env python3
import os, csv, yaml, time
import cv2
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROJECT = cfg["paths"]["project"]
TABLES  = os.path.join(PROJECT, cfg["paths"]["tables"])
FACES_DIR = os.path.join(PROJECT, cfg["paths"]["faces"])
BODIES_DIR = os.path.join(PROJECT, cfg["paths"]["bodies"])
IN_CSV  = os.path.join(TABLES, "faces.csv")
TMP_CSV = os.path.join(TABLES, "faces.tmp.csv")

schema = cfg["schema"]["faces"]
batch_size = int(cfg["runtime"]["batch_size"])

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

model_id = "iitolstykh/mivolo_v2"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()

skipped = 0
start_time = time.time()
with open(IN_CSV, "r", encoding="utf-8", newline="") as fi, \
     open(TMP_CSV, "w", encoding="utf-8", newline="") as fo:

    r = csv.DictReader(fi)
    w = csv.DictWriter(fo, fieldnames=r.fieldnames if r.fieldnames else schema)
    w.writeheader()

    batch_rows, batch_imgs_face, batch_imgs_body = [], [], []

    def flush_batch():
        if not batch_rows: return
        face_inputs = processor(images=batch_imgs_face)["pixel_values"]
        body_inputs = processor(images=batch_imgs_body)["pixel_values"]
        face_inputs = face_inputs.to(device=model.device, dtype=model.dtype)
        body_inputs = body_inputs.to(device=model.device, dtype=model.dtype)

        with torch.inference_mode():
            out = model(faces_input=face_inputs, body_input=body_inputs)
        ages = out.age_output.squeeze(-1).detach().cpu().tolist()  # [B] statt [B,1]
        for row, age in zip(batch_rows, ages):
            row["age_mivolo"] = round(float(age), 2)
            w.writerow(row)
        batch_rows.clear()
        batch_imgs_face.clear()
        batch_imgs_body.clear()

    counter = 0
    for row in r:
        counter += 1
        if counter % 500 == 0:
            print(f"Processed {counter} faces...")
        if row["usable"] == "False":
            skipped += 1
            w.writerow(row)
            continue

        face_path = os.path.join(FACES_DIR, row["file_name"])
        face_img = cv2.imread(face_path) 
        body_path = os.path.join(BODIES_DIR, row["image_id"] + "_" + row["body_id"] + ".jpg")
        body_img = cv2.imread(body_path) if os.path.exists(body_path) else None
        if face_img is None and body_img is None:
            w.writerow(row); continue

        batch_rows.append(row)
        batch_imgs_face.append(face_img)
        batch_imgs_body.append(body_img)

        if len(batch_rows) >= batch_size:
            flush_batch()

    flush_batch()

os.replace(TMP_CSV, IN_CSV)
dt = time.time() - start_time
print(f"Dauer: {dt:.1f}s")
print(f"Übersprungen: {skipped}")
print(f"MiVOLO2-Annotation fertig. Output aktualisiert: {IN_CSV} | device={device}")
