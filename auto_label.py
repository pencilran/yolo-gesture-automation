"""使用 MediaPipe 自动生成 YOLO 手势检测标注。"""

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
    """把归一化手部关键点转换成带少量边缘的 YOLO 方框。"""
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
        raise SystemExit(f"缺少手部检测模型：{MODEL_FILE}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"没有在 {IMAGE_DIR} 找到图片。")

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
                failed.append(f"未知文件名: {image_path.name}")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failed.append(f"图片无法读取: {image_path.name}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if not result.hand_landmarks:
                failed.append(f"没有检测到手: {image_path.name}")
                continue

            box = yolo_box(result.hand_landmarks[0])
            line = f"{class_id} " + " ".join(f"{value:.6f}" for value in box)
            label_path.write_text(line + "\n", encoding="utf-8")
            labeled += 1

            if index % 50 == 0:
                print(f"处理进度：{index}/{len(images)}")

    FAILED_FILE.write_text("\n".join(failed), encoding="utf-8")
    print("\n自动标注完成")
    print(f"有效手势标注：{labeled}")
    print(f"背景空标注：{backgrounds}")
    print(f"需要人工检查：{len(failed)}")
    print(f"未知文件名：{unknown}")
    print(f"检查清单：{FAILED_FILE}")


if __name__ == "__main__":
    main()
