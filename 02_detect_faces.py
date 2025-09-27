import os
import  csv
import cv2 
import yaml
import time
from deepface import DeepFace  # pip install deepface

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_DIR = cfg["paths"]["project"]
BODIES_PATH = os.path.join(PROJECT_DIR, cfg["paths"]["bodies"])
BODIES_CSV = os.path.join(PROJECT_DIR, cfg["paths"]["tables"], "bodies.csv")
FACES_CSV  = os.path.join(PROJECT_DIR, cfg["paths"]["tables"], "faces.csv")
OUT_FACES_PATH  = os.path.join(PROJECT_DIR, cfg["paths"]["faces"])
os.makedirs(OUT_FACES_PATH, exist_ok=True)

with open(BODIES_CSV) as file_bodies, open(FACES_CSV, "a", newline="") as file_faces:
    bodies_reader = csv.DictReader(file_bodies)
    faces_writer = csv.writer(file_faces)
    schema_faces = cfg["schema"]["faces"]
    faces_writer.writerow(schema_faces)
    t0 = time.time()

    for row in bodies_reader:
        image_id = row["image_id"]
        body_id = row["body_id"]
        BODY_PATH = os.path.join(BODIES_PATH, row["file_name"])
        img_body = cv2.imread(BODY_PATH)
        if img_body is None: 
            continue
        # Face-Detektion
        faces = DeepFace.extract_faces(
            img_body, enforce_detection=False, detector_backend=cfg["detect"]["deepface_backend"]
        )

        candidates = []
        for fdict in faces: 
            conf = float(fdict.get("confidence", 0.0))
            if conf < min(cfg["detect"]["face_conf_thres"]):
                continue
            candidates.append((conf, fdict))

        candidates.sort(key=lambda z: z[0], reverse=True)
        keep = 1 if candidates else 0

        for face_id, (conf, fdict) in enumerate(candidates):
            face = fdict["face"]
            if face.dtype != "uint8":
                face = (face*255).clip(0,255).astype("uint8")
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face_name = f"{image_id}_body{body_id}_face{face_id}.jpg"
            out_path = os.path.join(OUT_FACES_PATH, face_name)
            cv2.imwrite(out_path, face)

            row_dict = {
                "image_id": image_id,
                "body_id": body_id,
                "face_id": face_id,
                "face_conf": f"{conf:.6f}",
                "file_name": face_name,
                "detector_face": cfg["detect"]["deepface_backend"],
            }

            row_out = [row_dict.get(col, None) for col in schema_faces]
            faces_writer.writerow(row_out)
            del face
        del img_body, faces, candidates
dt = time.time() - t0
print(f"Fertig mit der Gesichtserkennung. Dauer: {dt:.1f}s")
