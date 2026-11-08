
import cv2
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = ROOT / "tests" / "road_video.mp4"
OUTPUT_VIDEO = ROOT / "outputs" / "detection_depth_result.mp4"

MODEL_PATH = ROOT / "yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classes relevant to driving
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# --------------------------------------------------
# LOAD YOLO AND MIDAS
# --------------------------------------------------

def load_models():
    print("Loading YOLO model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {MODEL_PATH}"
        )

    yolo = YOLO(str(MODEL_PATH))

    print("Loading MiDaS Small...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small"
    )

    midas.to(DEVICE)
    midas.eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    transform = midas_transforms.small_transform

    print(f"Models loaded successfully on {DEVICE}")

    return yolo, midas, transform


# --------------------------------------------------
# ESTIMATE RELATIVE DEPTH
# --------------------------------------------------

def estimate_depth(frame, midas, transform):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(rgb).to(DEVICE)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map


# --------------------------------------------------
# GET OBJECT DEPTH SCORE
# --------------------------------------------------

def get_relative_depth(depth_map, x1, y1, x2, y2):

    height, width = depth_map.shape

    x1 = max(0, min(int(x1), width - 1))
    x2 = max(0, min(int(x2), width))

    y1 = max(0, min(int(y1), height - 1))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return 0

    object_depth = depth_map[y1:y2, x1:x2]

    if object_depth.size == 0:
        return 0

    # Median is less sensitive to noisy pixels
    return float(np.median(object_depth))


# --------------------------------------------------
# NORMALIZE DEPTH SCORES
# --------------------------------------------------

def normalize_depth_scores(depth_map, values):

    if not values:
        return []

    low, high = np.percentile(depth_map, [5, 95])

    if high <= low:
        return [50] * len(values)

    scores = []

    for value in values:
        score = np.clip(
            (value - low) / (high - low),
            0,
            1
        )

        scores.append(int(score * 100))

    return scores


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

def process_video():

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    yolo, midas, transform = load_models()

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    frame_count = 0

    print("\nStarting YOLO + ByteTrack + MiDaS pipeline...")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # 1. Estimate depth for the frame
            depth_map = estimate_depth(
                frame,
                midas,
                transform
            )

            # 2. Run YOLO with persistent ByteTrack IDs
            results = yolo.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(TARGET_CLASSES.keys()),
                conf=0.30,
                verbose=False
            )

            result = results[0]

            depth_values = []
            object_boxes = []

            # 3. Extract detections and their depth
            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                object_depth = get_relative_depth(
                    depth_map,
                    x1, y1, x2, y2
                )

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                object_boxes.append({
                    "class": TARGET_CLASSES[class_id],
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "track_id": track_id,
                    "depth": object_depth
                })

                depth_values.append(object_depth)

            # 4. Normalize scores for this frame
            scores = normalize_depth_scores(
                depth_map,
                depth_values
            )

            # 5. Draw boxes and labels
            for obj, score in zip(object_boxes, scores):

                x1, y1, x2, y2 = obj["box"]

                label = (
                    f'{obj["class"]} '
                    f'ID:{obj["track_id"] if obj["track_id"] is not None else "N/A"} '
                    f'Depth:{score}/100'
                )

                color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            cv2.putText(
                frame,
                "YOLO + ByteTrack + MiDaS",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            writer.write(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(
                    f"Processed {frame_count}/{total_frames} frames"
                )

    finally:
        cap.release()
        writer.release()

    print("\nCombined perception pipeline completed!")
    print(f"Frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    process_video()
