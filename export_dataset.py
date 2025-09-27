import os, shutil, csv, yaml, pathlib

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROJECT = cfg["paths"]["project"]
TABLES  = os.path.join(PROJECT, cfg["paths"]["tables"])
FACES_DIR = os.path.join(PROJECT, cfg["paths"]["faces"])   
IN_CSV  = os.path.join(TABLES, "faces.csv")

EXPORT_ROOT = os.path.join(PROJECT, cfg["paths"]["export"])
EXPORT_IMG  = os.path.join(EXPORT_ROOT, "faces")
EXPORT_CSV  = os.path.join(EXPORT_ROOT, "faces.csv")

os.makedirs(EXPORT_IMG, exist_ok=True)

kept, missed = 0, 0
with open(IN_CSV, "r", encoding="utf-8", newline="") as fi, \
     open(EXPORT_CSV, "w", encoding="utf-8", newline="") as fo:
    r = csv.DictReader(fi)
    fieldnames = r.fieldnames
    w = csv.DictWriter(fo, fieldnames=fieldnames)
    w.writeheader()

    for row in r:
        if row.get("usable") == "False":
            continue
        src = os.path.join(FACES_DIR, row["file_name"])
        if not os.path.exists(src):
            missed += 1
            continue
        dst = os.path.join(EXPORT_IMG, row["file_name"])
        shutil.copy2(src, dst)        
        w.writerow(row)
        kept += 1

print(f"Export fertig. Kopiert: {kept} | Fehlend: {missed}")
print(f"Bilder: {EXPORT_IMG}")
print(f"Tabelle: {EXPORT_CSV}")
