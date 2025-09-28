import os
from PIL import Image
import imagehash  # pip install ImageHash
import yaml
import csv

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROJECT_DIR = cfg["paths"]["project"]
FACES_PATH  = os.path.join(PROJECT_DIR, cfg["paths"]["faces"])
TABLES_PATH = os.path.join(PROJECT_DIR, cfg["paths"]["tables"])
FACES_CSV   = os.path.join(TABLES_PATH, "faces.csv")

# Hash Duplikate entfernen
seen = {}
dupes = 0
removed_files = set()

for i, fname in enumerate(os.listdir(FACES_PATH), 1):
    path = os.path.join(FACES_PATH, fname)
    try:
        h = str(imagehash.phash(Image.open(path)))
    except Exception:
        continue
    if h in seen:
        os.remove(path)
        removed_files.add(fname)
        dupes += 1
    else:
        seen[h] = path
    if i % 1000 == 0:
        print(f"Processed {i} files...")

print(f"Duplicates removed: {dupes}")

# CSV bereinigen
if os.path.exists(FACES_CSV):
    tmp_csv = FACES_CSV + ".tmp"
    with open(FACES_CSV, "r", encoding="utf-8", newline="") as fi, \
         open(tmp_csv, "w", encoding="utf-8", newline="") as fo:
        reader = csv.DictReader(fi)
        writer = csv.DictWriter(fo, fieldnames=reader.fieldnames)
        writer.writeheader()
        kept, dropped = 0, 0
        for row in reader:
            fname_csv = row["file_name"]
            if fname_csv in removed_files:
                dropped += 1
                continue
            writer.writerow(row)
            kept += 1
    os.replace(tmp_csv, FACES_CSV)
    print(f"CSV updated: {kept} rows kept, {dropped} rows removed")
else:
    print("faces.csv not found, skipped CSV update.")
