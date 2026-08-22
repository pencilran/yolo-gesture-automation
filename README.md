# YOLO Gesture Automation Controller

A real-time automation control project built with a laptop camera and a custom-trained YOLO model. The system recognizes right-hand gestures, converts visual commands into machine states, and reduces accidental triggers through multi-frame confirmation and a cooldown period.

## Results

| Gesture | Control state |
| --- | --- |
| Open palm (`palm`) | `RUNNING` |
| Closed fist (`fist`) | `STOPPED` |
| Thumbs up (`thumb_up`) | `READY` |

The second model was trained on 774 self-collected camera images using an automated labeling pipeline. Eleven images in which no hand was detected were excluded automatically.

| Metric | Validation result |
| --- | ---: |
| Precision | 98.83% |
| Recall | 97.51% |
| mAP50 | 99.42% |
| mAP50-95 | 91.22% |

![Second training run](docs/training_results.png)

## Approach Comparison: Teachable Machine vs. YOLO

The project began with a browser-based image classification experiment in Teachable Machine. Camera testing revealed its limitations, so the final solution moved to YOLO with MediaPipe handedness detection.

| Comparison | Teachable Machine | YOLO + MediaPipe |
| --- | --- | --- |
| Recognition method | Classifies the entire frame | Locates the hand and classifies its gesture |
| Training workflow | No-code and quick to start | Requires labeling, dataset splitting, and local training |
| Complex backgrounds | May learn irrelevant background features | Focuses on the hand region and is more robust |
| `Other` category | A broad class can dominate the prediction probability | Background is represented by empty labels, not an output class |
| Handedness filtering | Not directly supported | MediaPipe identifies the hand; only the right hand is enabled by default |
| Automation reliability | Useful for rapid concept validation | Multi-frame confirmation and cooldown logic suit real-time control |

Teachable Machine confirmed that the three gestures were visually distinguishable, but the `Other` probability often remained high and whole-frame classification could not locate the hand. The final architecture therefore uses YOLO for gesture detection, MediaPipe for handedness filtering, and a state machine for automation commands.

## Pipeline

`Camera frame → YOLO gesture detection → MediaPipe handedness check → 8-frame stability check → Automation state change`

Key design decisions:

- YOLO detects the gesture class and hand location.
- MediaPipe determines handedness; only right-hand commands are accepted by default.
- Only the highest-confidence detection is used.
- A command must remain stable for eight consecutive frames.
- A 1.2-second cooldown prevents repeated triggers.
- NVIDIA GPU inference is selected automatically, with CPU fallback when CUDA is unavailable.
- Camera frames are never saved while the application is running.

## Quick Start

Recommended environment: Windows 11, Python 3.12, and a camera-equipped computer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Controls:

- `Q` or `Esc`: quit.
- `H`: switch the allowed hand between right and left.

The repository includes the trained gesture model at `models/gesture_best.pt` and the MediaPipe hand landmark model, so retraining is not required for a demonstration.

## Collect Data and Retrain

```powershell
.\.venv\Scripts\python.exe capture.py
.\.venv\Scripts\python.exe auto_label.py
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\yolo.exe detect train data=dataset/data.yaml model=yolo26n.pt epochs=30 imgsz=640 batch=16 device=0 workers=0 cache=True
```

Data collector controls: `1` open palm, `2` closed fist, `3` thumbs up, `4` background, `S` save, and `Q` quit. `auto_label.py` uses MediaPipe to generate YOLO bounding boxes automatically, reducing manual labeling work.

## Project Structure

- `app.py`: real-time recognition and automation state control.
- `capture.py`: camera-based data collection.
- `auto_label.py`: automatic YOLO annotation generation.
- `prepare_dataset.py`: overlay cleanup and stratified train/validation splitting.
- `models/gesture_best.pt`: final custom-trained gesture model.
- `models/hand_landmarker.task`: MediaPipe hand landmark model.

## Privacy

The public repository contains no raw photos, train/validation images, or facial data. The dataset, training cache, and local virtual environment are excluded through `.gitignore`.

## Technology

Python, Ultralytics YOLO, PyTorch, OpenCV, MediaPipe, and NVIDIA CUDA.

This project was created for AI automation learning and as a portfolio project. Dependencies and model assets remain subject to their respective licenses.
