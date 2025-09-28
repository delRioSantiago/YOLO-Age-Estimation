#!/usr/bin/env python3
import os, shutil, csv, yaml, pathlib, random, math
from collections import Counter

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROJECT   = cfg["paths"]["project"]
TABLES    = os.path.join(PROJECT, cfg["paths"]["tables"])
FACES_DIR = os.path.join(PROJECT, cfg["paths"]["faces"])
IN_CSV    = os.path.join(TABLES, "faces.csv")

# Export-Root für sauberen Datensatz
EXPORT_ROOT = os.path.join(PROJECT, cfg["paths"]["export"])
EXPORT_IMG  = os.path.join(EXPORT_ROOT, "faces")
EXPORT_CSV  = os.path.join(EXPORT_ROOT, "faces.csv")
os.makedirs(EXPORT_IMG, exist_ok=True)

# YOLO-Zieldataset
OUT_ROOT = os.path.join(PROJECT, cfg["train"]["out_dataset"])
SPLIT    = cfg["train"]["split"]
PRIO     = cfg["train"]["age_priority"]
SEED     = int(cfg["runtime"]["seed"])

CLASSES = ["Säuglinge","Kleinkinder","Kinder","Jugendliche","Erwachsene","Senioren"]

def is_true(v): return str(v).lower() in ("1","true","yes","y","true")

def pick_age(row):
    for k in PRIO:
        v = row.get(k, "")
        if v is None or str(v).strip() == "":
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None

def age_to_class(a):
    #Vorgabe bzgl. der Altersgruppen vom SIT: Säuglinge (0 – 1 Jahre), Kleinkinder (2 – 3 Jahre), Kinder (4 – 12 Jahre), Jugendliche (13 – 17 Jahre), Erwachsene (18 – 64 Jahre) sowie Senioren (65+ Jahre)
    a = max(0.0, a)
    if a < 2:   return "Säuglinge"
    if a < 4:   return "Kleinkinder"
    if a < 13:  return "Kinder"
    if a < 18:  return "Jugendliche"
    if a < 65:  return "Erwachsene"
    return "Senioren"

# ----------------- 1) Export nutzbare Crops -----------------
kept, missed = 0, 0
rows_filtered = []

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
        rows_filtered.append((row, dst))
        kept += 1

print(f"[Export] Fertig. Kopiert: {kept} | Fehlend: {missed}")
print(f"[Export] Bilder: {EXPORT_IMG}")
print(f"[Export] Tabelle: {EXPORT_CSV}")

# ----------------- 2) YOLO Dataset bauen -----------------
random.seed(SEED)
splits = { "train":[], "val":[], "test":[] }

for row, path in rows_filtered:
    age = pick_age(row)
    if age is None: 
        continue
    cls = age_to_class(age)
    splits["train"].append((path, cls))  # zunächst alles in Train

# Splitten pro Klasse
by_cls = {}
for path, cls in splits["train"]:
    by_cls.setdefault(cls, []).append(path)

target = cfg["train"]["target_per_class"]
splits = {"train": [], "val": [], "test": []}

for cls, files in by_cls.items():
    random.shuffle(files)
    n = len(files)

    # Erst Originale in Train/Val/Test aufteilen
    n_train = int(math.floor(0.8 * n))
    n_val   = int(math.floor(0.1 * n))
    n_test  = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    # Oversampling nur im Train-Set
    if len(train_files) < target:
        reps = math.ceil(target / len(train_files))
        train_files = (train_files * reps)[:target]
    else:
        train_files = random.sample(train_files, target)

    splits["train"] += [(p, cls) for p in train_files]
    splits["val"]   += [(p, cls) for p in val_files]
    splits["test"]  += [(p, cls) for p in test_files]



# Ordnerstruktur anlegen
for part in ["train","val","test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_ROOT, part, cls), exist_ok=True)

def copy_into(pairs, part):
    kept = 0
    for src, cls in pairs:
        name = os.path.basename(src)
        dst = os.path.join(OUT_ROOT, part, cls, name)
        try:
            shutil.copy2(src, dst)
            kept += 1
        except Exception:
            pass
    return kept

stats = {}
for part in ["train","val","test"]:
    cnt = Counter([cls for _,cls in splits[part]])
    copy_into(splits[part], part)
    stats[part] = dict(cnt)

print(f"[YOLO] Dataset: {OUT_ROOT}")
for part in ["train","val","test"]:
    print(part, stats[part])
