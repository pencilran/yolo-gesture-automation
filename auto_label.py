"""Generate YOLO gesture detection annotations automatically with MediaPipe."""

from pathlib import Path

import cv2
import mediapipe as mp


IMAGE_DIR = Path("dataset/images")
LABEL_DIR = Path("dataset/labels")
FAILED_FILE = Path("dataset/auto_label_failed.txt")
MODEL_FILE = Path("models/hand_landmarker.task")

CLASS_IDS = {
    "palm_": 0,
    "fist_": 1,
    "thumb_up_": 2,
}


def class_id_from_name(filename: str):
    for prefix, class_id in CLASS_IDS.items():
        if filename.startswith(prefix):
            return class_id
    return None


def yolo_box(landmarks, padding=0.04):
    """Convert normalized hand landmarks into a padded YOLO bounding box."""
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    x1 = max(0.0, min(xs) - padding)
    y1 = max(0.0, min(ys) - padding)
    x2 = min(1.0, max(xs) + padding)
    y2 = min(1.0, max(ys) + padding)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_FILE.exists():
        raise SystemExit(f"Missing hand detection model: {MODEL_FILE}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No images were found in {IMAGE_DIR}.")

    failed = []
    labeled = 0
    backgrounds = 0
    unknown = 0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=MODEL_FILE.read_bytes()
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for index, image_path in enumerate(images, start=1):
            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            if image_path.name.startswith("background_"):
                label_path.write_text("", encoding="utf-8")
                backgrounds += 1
                continue

            class_id = class_id_from_name(image_path.name)
            if class_id is None:
                unknown += 1
                failed.append(f"Unknown filename: {image_path.name}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failed.append(f"Unable to read image: {image_path.name}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if not result.hand_landmarks:
                failed.append(f"No hand detected: {image_path.name}")
                continue

            box = yolo_box(result.hand_landmarks[0])
            line = f"{class_id} " + " ".join(f"{value:.6f}" for value in box)
            label_path.write_text(line + "\n", encoding="utf-8")
            labeled += 1

            if index % 50 == 0:
                print(f"Progress: {index}/{len(images)}")

    FAILED_FILE.write_text("\n".join(failed), encoding="utf-8")
    print("\nAutomatic labeling complete")
    print(f"Valid gesture labels: {labeled}")
    print(f"Empty background labels: {backgrounds}")
    print(f"Files requiring review: {len(failed)}")
    print(f"Unknown filenames: {unknown}")
    print(f"Review list: {FAILED_FILE}")


if __name__ == "__main__":
    main()
