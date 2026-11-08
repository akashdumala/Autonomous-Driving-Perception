# Autonomous Driving Perception System

An AI-powered computer vision project that analyzes road scenes and driving videos using object detection, multi-object tracking, lane detection, and monocular depth estimation.

The project also combines these modules into an integrated perception pipeline that processes road video and generates an annotated output.

## Features

### 1. Object Detection
- Uses YOLOv8 Nano (`yolov8n.pt`).
- Detects road-relevant objects:
  - Person
  - Bicycle
  - Car
  - Motorcycle
  - Bus
  - Truck
- Displays class labels and confidence scores.
- Saves annotated detection images.

### 2. Multi-Object Tracking
- Uses YOLOv8 tracking with ByteTrack.
- Maintains object identities across video frames.
- Processes road videos frame by frame.
- Generates an annotated tracking video.

### 3. Lane Detection
- Uses OpenCV image processing.
- Converts frames to grayscale.
- Applies Gaussian blur and Canny edge detection.
- Uses a region-of-interest mask and Hough Line Transform.
- Draws estimated left and right lane boundaries.

### 4. Monocular Depth Estimation
- Uses the MiDaS Small pretrained model.
- Estimates relative depth from individual video frames.
- Generates a colorized depth visualization.
- Processes road videos frame by frame.

### 5. Integrated Perception Pipeline
Combines the perception modules into a single video-processing workflow.

The integrated pipeline successfully processed **2,208 frames** and generated:

`outputs/integrated_perception_result.mp4`

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Object detection | YOLOv8 |
| Object tracking | ByteTrack |
| Lane detection | OpenCV |
| Depth estimation | MiDaS |
| Numerical processing | NumPy |
| Deep learning | PyTorch |
| Video processing | OpenCV |

## Project Structure

```text
Autonomous-Driving-Perception/
│
├── src/
│   ├── detection/
│   │   └── detect_objects.py
│   │
│   ├── tracking/
│   │   └── track_objects.py
│   │
│   ├── lane_detection/
│   │   └── lane_detection.py
│   │
│   ├── depth_estimation/
│   │   └── depth_estimation.py
│   │
│   └── integrated_pipeline.py
│
├── simulator/
│   └── carla_test.py
│
├── tests/
│   └── road_video.mp4
│
├── data/
├── docs/
├── models/
├── outputs/
├── runs/
│
├── bus.jpg
├── yolov8n.pt
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Autonomous-Driving-Perception
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run commands from the project root directory.

### Object Detection

```bash
python src/detection/detect_objects.py
```

### Object Tracking

```bash
python src/tracking/track_objects.py
```

### Lane Detection

```bash
python src/lane_detection/lane_detection.py
```

### Depth Estimation

```bash
python src/depth_estimation/depth_estimation.py
```

### Integrated Perception

```bash
python src/integrated_pipeline.py
```

Generated outputs are saved in the `outputs/` directory.

## Execution Results

The individual modules were executed successfully.

| Module | Result |
|---|---|
| Object detection | Completed; detected road objects |
| Object tracking | Completed; processed 2,208 frames |
| Lane detection | Completed; processed 2,208 frames |
| Depth estimation | Completed; processed 2,208 frames |
| Integrated pipeline | Completed; processed 2,208 frames |

### Integrated Pipeline Output

```text
INTEGRATED PERCEPTION COMPLETED!
Total frames: 2208
Output: outputs/integrated_perception_result.mp4
```

## Hardware and Execution Notes

- Designed to run on a Windows development environment.
- Supports CPU execution.
- CUDA acceleration is used when available by the depth-estimation module.
- MiDaS pretrained weights may be downloaded through PyTorch Hub on first use.
- Video-processing time depends on the input video, resolution, and hardware.

## Limitations

- YOLOv8 uses pretrained COCO classes rather than a custom driving-specific dataset.
- Lane detection uses traditional image-processing techniques and may be affected by poor lighting, road markings, and curved roads.
- MiDaS provides relative depth visualization, not calibrated real-world distance measurements.
- This project is a computer vision prototype and is not validated for real-world autonomous vehicle control or safety-critical use.

## Future Improvements

- Integrate camera calibration and real-world distance estimation.
- Improve lane detection under challenging road conditions.
- Train or fine-tune models on driving-specific datasets.
- Add traffic sign and traffic light recognition.
- Integrate a live camera feed.
- Explore simulator-based testing and evaluation.

## Disclaimer

This project is developed for educational and experimental purposes. It is not intended to control real vehicles or replace validated automotive safety systems.

## Author

**Dumala Akash**

Autonomous Driving Perception — Computer Vision Project
