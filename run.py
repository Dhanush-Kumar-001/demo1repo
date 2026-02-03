from ultralytics import YOLO
import cv2
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO("yolov8n.pt")

    results = model(
        image_path,
        classes=[0],
        conf=0.25
    )

    
    annotated = results[0].plot()
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    main()