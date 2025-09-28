import subprocess, sys, os

venv_python = os.path.expanduser("~/environments/age-env/bin/python")  # Linux
# unter Windows evtl: r"C:\Users\santi\OneDrive\Dokumente\Age-Estimation\Code\age-estimation\Scripts\python.exe"

scripts = [
    "01_detect_bodies.py",
    "02_detect_faces.py",
    "03_annotate_faces_Deepface.py",
    "04_quality_faces.py",
    "05_annotate_faces_mivolo2.py",
    "06_annotate_faces_insightface.py",
    "07_export_and_build_yolo_dataset.py",
]

for s in scripts:
    print(f"\n=== Running {s} ===\n")
    ret = subprocess.run([venv_python, s])
    if ret.returncode != 0:
        sys.exit(f"Abgebrochen bei {s}")

