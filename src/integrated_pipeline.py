
import cv2
import torch
import numpy as np

from pathlib import Path
from ultralytics import YOLO

from lane_detection.lane_detection import detect_lanes


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "tests" / "road_video.mp4"
OUTPUT_PATH = ROOT / "outputs" / "integrated_perception_result.mp4"

YOLO_MODEL_PATH = ROOT / "yolov8n.pt"

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD OBJECT DETECTION MODEL
# --------------------------------------------------

def load_yolo():

    print("Loading YOLOv8...")

    model = YOLO(str(YOLO_MODEL_PATH))

    print("YOLOv8 loaded successfully!")

    return model


# --------------------------------------------------
# LOAD DEPTH ESTIMATION MODEL
# --------------------------------------------------

def load_midas():

    print(f"Loading MiDaS on {DEVICE}...")

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small"
    )

    model.to(DEVICE)
    model.eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    transform = transforms.small_transform

    print("MiDaS loaded successfully!")

    return model, transform


# --------------------------------------------------
# DEPTH ESTIMATION
# --------------------------------------------------

def estimate_depth(frame, model, transform):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(rgb).to(DEVICE)

    with torch.no_grad():

        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()

    normalized = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_INFERNO
    )

    return colored


# --------------------------------------------------
# INTEGRATED PERCEPTION PIPELINE
# --------------------------------------------------

def main():

    print("\nStarting Autonomous Driving Perception System")
    print("-----------------------------------------------")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load models once
    yolo = load_yolo()
    midas, midas_transform = load_midas()

    # Open input video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Final video contains two side-by-side panels
    output_width = width * 2

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    frame_count = 0

    print("\nProcessing video...")
    print("YOLO + ByteTrack + Lane Detection + MiDaS")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # 1. YOLO object detection and tracking
            results = yolo.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=TARGET_CLASSES,
                conf=0.35,
                verbose=False
            )

            detection_frame = results[0].plot(
                labels=True,
                conf=True
            )

            # 2. Lane detection
            lane_frame = detect_lanes(frame)

            # Extract lane markings without blending the
            # original image twice.
            lane_overlay = cv2.absdiff(
                lane_frame,
                frame
            )

            # Combine object tracking and lane markings
            perception_frame = cv2.addWeighted(
                detection_frame,
                1.0,
                lane_overlay,
                1.0,
                0
            )

            # 3. MiDaS depth estimation
            depth_frame = estimate_depth(
                frame,
                midas,
                midas_transform
            )

            # 4. Resize panels to identical dimensions
            perception_frame = cv2.resize(
                perception_frame,
                (width, height)
            )

            depth_frame = cv2.resize(
                depth_frame,
                (width, height)
            )

            # Add panel titles
            cv2.putText(
                perception_frame,
                "OBJECT TRACKING + LANE DETECTION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                depth_frame,
                "MiDaS DEPTH ESTIMATION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # 5. Combine both panels
            combined_frame = np.hstack(
                (perception_frame, depth_frame)
            )

            writer.write(combined_frame)

            frame_count += 1

            if frame_count % 30 == 0:
                print(
                    f"Processed {frame_count}/{total_frames} frames"
                )

    finally:

        cap.release()
        writer.release()

    print("\n-----------------------------------------------")
    print("INTEGRATED PERCEPTION COMPLETED!")
    print(f"Total frames: {frame_count}")
    print(f"Output: {OUTPUT_PATH.resolve()}")
    print("-----------------------------------------------")


if __name__ == "__main__":
    main()
