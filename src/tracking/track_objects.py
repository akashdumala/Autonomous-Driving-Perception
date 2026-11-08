
import cv2
from pathlib import Path
from ultralytics import YOLO

# Project paths
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "tests/road_video.mp4"
OUTPUT_PATH = "outputs/tracking_result.mp4"

# Driving-related COCO classes
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    Path("outputs").mkdir(exist_ok=True)

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    print("Starting YOLO + ByteTrack tracking...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES,
            conf=0.35,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot(
            labels=True,
            conf=True
        )

        writer.write(annotated_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    writer.release()

    print("\nTracking completed!")
    print(f"Total frames: {frame_count}")
    print(f"Output: {Path(OUTPUT_PATH).resolve()}")

if __name__ == "__main__":
    main()
