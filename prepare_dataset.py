"""清理采集画面并按类别划分 YOLO 训练集和验证集。"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("dataset")
SOURCE_IMAGES = ROOT / "images"
SOURCE_LABELS = ROOT / "labels"
TRAIN_IMAGES = SOURCE_IMAGES / "train"
VAL_IMAGES = SOURCE_IMAGES / "val"
TRAIN_LABELS = SOURCE_LABELS / "train"
VAL_LABELS = SOURCE_LABELS / "val"
SUMMARY_FILE = ROOT / "split_summary.txt"
DATA_CONFIG = ROOT / "data.yaml"

PREFIXES = ("palm_", "fist_", "thumb_up_", "background_")


def group_name(filename: str):
    for prefix in PREFIXES:
        if filename.startswith(prefix):
            return prefix.removesuffix("_")
    return None


def remove_capture_overlay(image):
    """修复左上角采集器文字；不改变图片尺寸，因此标注坐标仍有效。"""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    regions = [
        (10, 5, min(330, width), min(52, height), "green"),
        (10, 52, min(570, width), min(91, height), "white"),
        (10, 88, min(390, width), min(124, height), "white"),
    ]

    for x1, y1, x2, y2, color in regions:
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        if color == "green":
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            local = cv2.inRange(hsv, (30, 45, 25), (95, 255, 255))
        else:
            local = cv2.inRange(roi, (175, 175, 175), (255, 255, 255))
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], local)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def main():
    output_dirs = (TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS)
    if any(folder.exists() and any(folder.iterdir()) for folder in output_dirs):
        raise SystemExit("训练集或验证集文件夹已有内容；为防止覆盖，本次未执行。")
    for folder in output_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    excluded = []
    for image_path in sorted(SOURCE_IMAGES.glob("*.jpg")):
        group = group_name(image_path.name)
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
        if group is None or not label_path.exists():
            excluded.append(image_path.name)
            continue
        groups[group].append((image_path, label_path))

    rng = random.Random(42)
    summary = []
    for group, samples in sorted(groups.items()):
        rng.shuffle(samples)
        val_count = max(1, round(len(samples) * 0.2))
        val_samples = samples[:val_count]
        train_samples = samples[val_count:]

        for split, items, image_dir, label_dir in (
            ("train", train_samples, TRAIN_IMAGES, TRAIN_LABELS),
            ("val", val_samples, VAL_IMAGES, VAL_LABELS),
        ):
            saved = 0
            for image_path, label_path in items:
                image = cv2.imread(str(image_path))
                if image is None:
                    excluded.append(image_path.name)
                    continue
                cleaned = remove_capture_overlay(image)
                cv2.imwrite(str(image_dir / image_path.name), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                shutil.copy2(label_path, label_dir / label_path.name)
                saved += 1
            summary.append(f"{group} {split}: {saved}")

    summary.append(f"excluded: {len(excluded)}")
    if excluded:
        summary.extend(f"  {name}" for name in excluded)
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")
    dataset_path = ROOT.resolve().as_posix()
    DATA_CONFIG.write_text(
        f'path: "{dataset_path}"\n'
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: palm\n"
        "  1: fist\n"
        "  2: thumb_up\n",
        encoding="utf-8",
    )
    print("\n数据集整理完成")
    for line in summary[: len(groups) * 2 + 1]:
        print(line)
    print(f"详细记录：{SUMMARY_FILE}")
    print(f"训练配置：{DATA_CONFIG}")


if __name__ == "__main__":
    main()
