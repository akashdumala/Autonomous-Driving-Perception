
import cv2
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = ROOT / "tests" / "road_video.mp4"
OUTPUT_VIDEO = ROOT / "outputs" / "depth_estimation_result.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
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

    # Normalize for visualization only
    depth_normalized = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_INFERNO
    )

    return depth_colored


def process_video():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {INPUT_VIDEO}")

    model, transform = load_model()

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

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

    print("Starting MiDaS depth estimation...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            depth_frame = estimate_depth(
                frame,
                model,
                transform
            )

            writer.write(depth_frame)
            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

    finally:
        cap.release()
        writer.release()

    print("\nDepth estimation completed!")
    print(f"Frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    process_video()
