"""Object detection with YOLO26n.

The app calls detect(image, conf) and gets back the best waste item:
    ("bottle", 0.91, (x1, y1, x2, y2))   or   None

Test one image from the terminal:
    python src/detect.py data/test/bottle_01.jpeg
"""
import sys

import torch
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"  # nano model, pretrained on COCO (80 everyday classes)
DEFAULT_CONF = 0.3         # chosen by the threshold experiment on 34 photos (see notes.md)
MIN_CONF = 0.05            # keep almost everything; thresholds are applied later in pick_best()
OUTPUT_PATH = "results/detect_output.jpg"

# Use the Apple GPU (Metal Performance Shaders) when available, otherwise the CPU.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# COCO classes that can be household waste. Everything else (person, chair, ...) is ignored.
WASTE_CLASSES = {"bottle", "cup", "wine glass", "bowl", "banana", "apple", "orange", "book", "carrot", "spoon"}

# Load the model once when this file is imported, not on every detect() call (~1 s each).
model = YOLO(MODEL_NAME)


def all_detections(image, save_path=None):
    """Run YOLO once and return every detection as a list of (label, conf, box).

    image can be a file path or a PIL image (Streamlit's camera gives us one of those).
    """
    result = model(image, conf=MIN_CONF, device=DEVICE, verbose=False)[0]

    detections = []
    for box in result.boxes:
        label = result.names[int(box.cls)]       # class number -> name, e.g. 39 -> "bottle"
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()    # box corners in pixels
        detections.append((label, conf, (x1, y1, x2, y2)))

    if save_path:
        result.save(filename=save_path)          # image with all boxes drawn on it
    return detections


def pick_best(detections, conf):
    """Keep waste classes at or above the threshold; return the most confident one, or None."""
    waste_items = []
    for label, score, box in detections:
        if label in WASTE_CLASSES and score >= conf:
            waste_items.append((label, score, box))

    if not waste_items:
        return None
    # max() compares items by what key returns: here the confidence (position 1 in the tuple).
    return max(waste_items, key=lambda item: item[1])


def detect(image, conf=DEFAULT_CONF):
    """What the app calls: one photo in, best waste item (or None) out."""
    return pick_best(all_detections(image), conf)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/detect.py <image_path>")
        sys.exit(1)

    detections = all_detections(sys.argv[1], save_path=OUTPUT_PATH)

    print(f"All detections (conf >= {MIN_CONF}):")
    for label, score, box in detections:
        print(f"  {label} {score:.2f}")

    best = pick_best(detections, DEFAULT_CONF)
    if best is None:
        print(f"Best waste item at conf {DEFAULT_CONF}: none")
    else:
        print(f"Best waste item at conf {DEFAULT_CONF}: {best[0]} {best[1]:.2f}")
    print(f"Boxed image saved to {OUTPUT_PATH}")
