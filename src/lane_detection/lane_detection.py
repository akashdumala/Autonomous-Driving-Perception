
import cv2
import numpy as np
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = ROOT / "tests" / "road_video.mp4"
OUTPUT_VIDEO = ROOT / "outputs" / "lane_detection_result.mp4"

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)


def region_of_interest(image):
    height, width = image.shape[:2]

    # Trapezoid covering the road area
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.42), int(height * 0.58)),
        (int(width * 0.58), int(height * 0.58)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


def average_lane_lines(image, lines):
    height, width = image.shape[:2]
    left_lines = []
    right_lines = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Ignore nearly horizontal lines
        if abs(slope) < 0.5:
            continue

        intercept = y1 - slope * x1

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    averaged = []

    for group in (left_lines, right_lines):
        if not group:
            continue

        slope, intercept = np.mean(group, axis=0)

        y_bottom = height
        y_top = int(height * 0.60)

        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)

        averaged.append((x_bottom, y_bottom, x_top, y_top))

    return averaged


def detect_lanes(frame):
    height, width = frame.shape[:2]

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Keep only the road region
    cropped_edges = region_of_interest(edges)

    # 5. Detect line segments using Hough Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )

    # 6. Average left and right lane lines
    lane_lines = average_lane_lines(frame, lines)

    # 7. Draw detected lanes on an overlay
    overlay = np.zeros_like(frame)

    for x1, y1, x2, y2 in lane_lines:
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

    result = cv2.addWeighted(frame, 0.8, overlay, 1.0, 0)

    return result


def process_video():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

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
        raise RuntimeError("Could not create output video.")

    frame_count = 0

    print("Starting OpenCV lane detection...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = detect_lanes(frame)
            writer.write(result)

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

    finally:
        cap.release()
        writer.release()

    print("\nLane detection completed!")
    print(f"Frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    process_video()
