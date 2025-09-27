import os, csv, shutil, cv2, yaml, time
import numpy as np # pip install numpy

# --- Config laden ---
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROJECT_DIR = cfg["paths"]["project"]
TABLES_DIR = os.path.join(PROJECT_DIR, cfg["paths"]["tables"])
IN_PATH    = os.path.join(TABLES_DIR, "faces.csv")
TMP_PATH   = os.path.join(TABLES_DIR, "faces.tmp.csv")

schema = cfg["schema"]["faces"]

if not os.path.exists(IN_PATH):
    raise FileNotFoundError(f"faces.csv nicht gefunden: {IN_PATH}")
os.makedirs(TABLES_DIR, exist_ok=True)

# --- Qualitätsfunktionen ---
def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

# --- Verarbeitung ---
start_time = time.time()
with open(IN_PATH, "r", encoding="utf-8", newline="") as fi, \
     open(TMP_PATH, "w", encoding="utf-8", newline="") as fo:
    reader = csv.DictReader(fi)
    writer = csv.DictWriter(fo, fieldnames=schema)
    writer.writeheader()

    processed = skipped = errors = usable_count = 0
    for row in reader:
        face_path = os.path.join(PROJECT_DIR, cfg["paths"]["faces"], row["file_name"])
        try:
            img = cv2.imread(face_path)
            if img is None:
                row["sharpness"] = None
                row["brightness"] = None
                row["contrast"]  = None
                row["usable"]    = False
                errors += 1
            else:
                sh = sharpness_score(img)
                br = brightness_score(img)
                ct = contrast_score(img)
                h, w = img.shape[:2]


                row["sharpness"] = f"{sh:.2f}"
                row["brightness"] = f"{br:.2f}"
                row["contrast"]  = f"{ct:.2f}"

                usable = (
                    sh >= cfg["quality"]["min_sharpness"] and
                    ct >= cfg["quality"]["min_contrast"] and
                    cfg["quality"]["min_brightness"] <= br <= cfg["quality"]["max_brightness"] and
                    min(h, w) >= cfg["quality"]["min_face_size"]
                )
                row["usable"] = usable
                if usable:
                    usable_count += 1
                processed += 1
        except Exception:
            row["sharpness"] = None
            row["brightness"] = None
            row["contrast"]  = None
            row["usable"]    = False
            errors += 1

        writer.writerow(row)

shutil.move(TMP_PATH, IN_PATH)
end_time = time.time()
print(f"Quality-Check fertig: {processed} ok, {errors} Fehler, {usable_count} nutzbar, Output: {IN_PATH}")
print(f"Verarbeitungszeit: {end_time - start_time:.2f} Sekunden")
