"""Brick 1: run YOLO26n on one image and print what it finds.

Usage:
    python src/detect.py data/test/bottle.jpg
"""
import sys

from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"  # nano model, pretrained on COCO (80 everyday classes)
OUTPUT_PATH = "results/detect_output.jpg"

if __name__ == "__main__":
    # The image path comes from the command line: sys.argv[0] is the script name,
    # sys.argv[1] is the first argument the user typed.
    if len(sys.argv) < 2:
        print("Usage: python src/detect.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]

    # Loads the weights. The first run downloads yolo26n.pt (~5 MB).
    model = YOLO(MODEL_NAME)

    # YOLO accepts a list of images, so it always returns a list of results.
    # We gave it one image, so we take the first (and only) result.
    result = model(image_path, verbose=False)[0]

    if len(result.boxes) == 0:
        print("Nothing detected.")

    for box in result.boxes:
        class_id = int(box.cls)            # a number 0-79
        label = result.names[class_id]     # number -> name, e.g. 39 -> "bottle"
        conf = float(box.conf)             # confidence 0-1
        print(f"{label} {conf:.2f}")

    # Draws all boxes + labels on the image and saves it.
    result.save(filename=OUTPUT_PATH)
    print(f"Boxed image saved to {OUTPUT_PATH}")
