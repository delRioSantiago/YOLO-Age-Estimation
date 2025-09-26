import os
import csv
import shutil
import yaml
import time
import cv2
from deepface import DeepFace 

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

FACES_CSV   = os.path.join(cfg["paths"]["tables"], "faces.csv")
TMP = os.path.join(cfg["paths"]["tables"], "faces.tmp.csv")
MIN_FACE_CONF   = min(cfg["detect"]["face_conf_thres"])

if not os.path.exists(FACES_CSV):
    raise FileNotFoundError(f"faces.csv nicht gefunden: {FACES_CSV}")
os.makedirs(cfg["paths"]["tables"], exist_ok=True)

schema_faces = cfg["schema"]["faces"]
buffer_rows = []
processed = 0
skipped   = 0
errors    = 0
t0 = time.time()

with open(FACES_CSV, "r", encoding="utf-8") as faces_file, open(TMP, "w", newline="", encoding="utf-8") as tmp_file:
    reader_faces = csv.DictReader(faces_file)
    writer_faces = csv.DictWriter(tmp_file, fieldnames=schema_faces)
    writer_faces.writeheader()
    for row in reader_faces:
        try: 
            img_face = cv2.imread(row["face_crop_path"])
            if img_face is None:
                errors += 1
                continue
            face_predictions = DeepFace.analyze(img_face, actions=["age", "gender", "race"], enforce_detection=False, detector_backend="skip")
            face_predictions = face_predictions[0] if isinstance(face_predictions, list) else face_predictions
            if not face_predictions:
                skipped += 1
                continue
            row["age_deepface"] = face_predictions.get("age")
            row["gender"] = face_predictions.get("dominant_gender")
            row["race"] = face_predictions.get("dominant_race")
        except Exception as e:
            errors += 1
            pass
        writer_faces.writerow(row)
        processed += 1
shutil.move(TMP, FACES_CSV)
dt = time.time() - t0
print(f"Fertig. Neu annotiert: {processed}, übersprungen: {skipped}, Fehler: {errors}, Dauer: {dt:.1f}s")
print(f"Output: {FACES_CSV}")
