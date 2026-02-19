import os
import csv
from ultralytics import YOLO

IMG_DIR = "baseline_images"
OUT_CSV = "baseline_report.csv"

model = YOLO("yolov8n.pt")

rows = []
for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(IMG_DIR, fname)
    results = model(path, conf=0.25, verbose=False)

    r = results[0]
    names = r.names  # class id -> class name
    boxes = r.boxes

    # summarize detections per image
    count_by_class = {}
    if boxes is not None and len(boxes) > 0:
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            cls_name = names[int(cls_id)]
            count_by_class[cls_name] = count_by_class.get(cls_name, 0) + 1

    rows.append({
        "image": fname,
        "total_detections": sum(count_by_class.values()),
        "classes_detected": ", ".join([f"{k}:{v}" for k, v in sorted(count_by_class.items())])
    })

# write csv
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "total_detections", "classes_detected"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved report -> {OUT_CSV}")
