"""Brick 2 experiment: which confidence threshold works best on our own photos?

Reads data/test/labels.csv (file,item,true_label,scenario), runs YOLO ONCE per photo,
then tries every threshold on the saved detections using the same pick_best() as the app.

true_label may list several acceptable answers separated by "|", e.g. "bottle|none".
Use "none" for photos where the right answer is "no waste item found".

Run:
    python src/threshold_test.py
"""
from datetime import date

import pandas as pd

from detect import all_detections, pick_best

LABELS_PATH = "data/test/labels.csv"
THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def outcome(predicted, true_label):
    """Compare one prediction with the truth -> "correct", "missed" or "wrong"."""
    acceptable = true_label.split("|")
    if predicted in acceptable:
        return "correct"
    if predicted == "none":
        return "missed"     # a real item, but nothing passed the threshold
    return "wrong"          # a label came out, but not an acceptable one (worst case)


labels = pd.read_csv(LABELS_PATH)
rows = []

for photo in labels.itertuples():
    detections = all_detections(f"data/test/{photo.file}")   # YOLO runs once per photo

    for threshold in THRESHOLDS:
        best = pick_best(detections, threshold)
        predicted = "none" if best is None else best[0]
        rows.append({
            "file": photo.file,
            "item": photo.item,
            "scenario": photo.scenario,
            "true_label": photo.true_label,
            "threshold": threshold,
            "predicted": predicted,
            "confidence": None if best is None else round(best[1], 3),
            "outcome": outcome(predicted, photo.true_label),
        })

results = pd.DataFrame(rows)

# One row per threshold, one column per outcome, counting photos.
summary = results.pivot_table(index="threshold", columns="outcome",
                              values="file", aggfunc="count", fill_value=0)
for column in ["correct", "missed", "wrong"]:
    if column not in summary:
        summary[column] = 0     # e.g. no "wrong" at all -> still show the column as 0
summary = summary[["correct", "missed", "wrong"]]

today = date.today().isoformat()
results.to_csv(f"results/threshold_photos_{today}.csv", index=False)
summary.to_csv(f"results/threshold_summary_{today}.csv")

print(f"{len(labels)} photos, model yolo26n.pt\n")
print(summary)
print(f"\nSaved results/threshold_photos_{today}.csv and results/threshold_summary_{today}.csv")
