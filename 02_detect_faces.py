import os
import  csv
import cv2
import yaml
from deepface import DeepFace  

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

BODIES_CSV = os.path.join(cfg["paths"]["tables"], "bodies.csv")
FACES_CSV  = os.path.join(cfg["paths"]["tables"], "faces.csv")
OUT_FACES_PATH  = cfg["paths"]["out_faces"]
os.makedirs(OUT_FACES_PATH, exist_ok=True)

with open(BODIES_CSV) as file_bodies, open(FACES_CSV, "a", newline="") as file_faces:
    bodies_reader = csv.DictReader(file_bodies)
    faces_writer = csv.writer(file_faces)
    header = cfg["schema"]["faces"]
    faces_writer.writerow(header)
    bodies_updates = {}  

    for row in bodies_reader:
        image_id = row["image_id"]
        body_id = row["body_id"]
        body_path = row["body_crop_path"]
        img_body = cv2.imread(body_path)
        if img_body is None: 
            bodies_updates[(image_id,body_id)] = (0)
            continue
        # Face-Detektion
        faces = DeepFace.extract_faces(
            img_body, enforce_detection=False, detector_backend=cfg["detect"]["face_backend"]
        )

        candidates = []
        for fdict in faces: 
            conf = float(fdict.get("confidence", 0.0))
            if conf < min(cfg["detect"]["face_conf_thres"]):
                continue
            candidates.append((conf, fdict))

        candidates.sort(key=lambda z: z[0], reverse=True)
        keep = 1 if candidates else 0
        bodies_updates[(image_id,body_id)] = (keep)

        for face_id, (conf, fdict) in enumerate(candidates):
            face = fdict["face"]
            if face.dtype != "uint8":
                face = (face*255).clip(0,255).astype("uint8")
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face_name = f"{image_id}_body{body_id}_face{face_id}.jpg"
            out_path = os.path.join(OUT_FACES_PATH, face_name)
            # optional: letterbox auf cfg["preprocess"]["face_out_size"] TODO entscheiden
            face = cv2.resize(face, (cfg["preprocess"]["face_out_size"])*2) if False else face
            cv2.imwrite(out_path, face)

            faces_writer.writerow([image_id, body_id, face_id, f"{conf:.6f}", out_path, cfg["detect"]["face_backend"]])
