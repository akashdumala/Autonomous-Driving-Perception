
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load pretrained lightweight YOLO model
model = YOLO("yolov8n.pt")

# COCO classes relevant to driving
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect_objects(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    results = model.predict(
        source=image,
        classes=list(TARGET_CLASSES.keys()),
        conf=0.35,
        verbose=False
    )

    annotated = results[0].plot()

    Path("outputs").mkdir(exist_ok=True)
    output_path = Path("outputs/detection_result.jpg")

    cv2.imwrite(str(output_path), annotated)

    print("Detection completed!")
    print("Detected objects:")

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"{TARGET_CLASSES[class_id]} "
            f"- Confidence: {confidence:.2f}"
        )

    print(f"Saved result: {output_path.resolve()}")

if __name__ == "__main__":
    detect_objects("bus.jpg")
