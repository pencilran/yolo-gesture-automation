"""YOLO 摄像头手势控制面板。"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO


CONFIDENCE = 0.25
STABLE_FRAMES = 8
COOLDOWN_SECONDS = 1.2
ALLOWED_HAND = "Right"

COMMANDS = {
    "palm": "RUNNING",
    "fist": "STOPPED",
    "thumb_up": "READY",
}

STATE_COLORS = {
    "READY": (0, 210, 255),
    "RUNNING": (0, 200, 0),
    "STOPPED": (0, 0, 230),
}


def find_best_model():
    release_model = Path("models/gesture_best.pt")
    if release_model.exists():
        return release_model

    models = list(Path("runs").rglob("best.pt"))
    if not models:
        raise FileNotFoundError(
            "没有找到正式模型 models/gesture_best.pt，也没有找到训练结果 best.pt。"
        )
    return max(models, key=lambda path: path.stat().st_mtime)


def create_hand_detector():
    model_file = Path("models/hand_landmarker.task")
    if not model_file.exists():
        raise FileNotFoundError(f"缺少左右手识别模型：{model_file}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=model_file.read_bytes()
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_handedness(detector, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if not result.handedness:
        return None
    detected = result.handedness[0][0].category_name
    # 当前摄像头预览采用镜像显示，将模型标签换回使用者的真实左右手。
    if detected == "Left":
        return "Right"
    if detected == "Right":
        return "Left"
    return detected


def main():
    global ALLOWED_HAND

    model_path = find_best_model()
    print(f"使用模型：{model_path}")
    inference_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"推理设备：{'GPU' if inference_device == 0 else 'CPU'}")
    model = YOLO(str(model_path))

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("摄像头打不开，请关闭其他占用摄像头的程序。")

    state = "READY"
    candidate = None
    stable_count = 0
    last_action_time = 0.0

    with create_hand_detector() as hand_detector:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            # 采集数据时使用了镜像画面，正式识别保持一致。
            frame = cv2.flip(frame, 1)
            handedness = detect_handedness(hand_detector, frame)
            prediction = model.predict(
                frame,
                conf=CONFIDENCE,
                iou=0.45,
                max_det=3,
                device=inference_device,
                verbose=False,
            )[0]

            gesture = None
            confidence = 0.0
            best_box = None
            if prediction.boxes is not None and len(prediction.boxes):
                best_index = int(prediction.boxes.conf.argmax().item())
                box = prediction.boxes[best_index]
                confidence = float(box.conf.item())
                gesture = prediction.names[int(box.cls.item())]
                best_box = [int(value) for value in box.xyxy[0].tolist()]

            valid_command = (
                gesture in COMMANDS
                and handedness == ALLOWED_HAND
                and confidence >= CONFIDENCE
            )

            if valid_command:
                if gesture == candidate:
                    stable_count += 1
                else:
                    candidate = gesture
                    stable_count = 1

                enough_time = time.time() - last_action_time >= COOLDOWN_SECONDS
                if stable_count >= STABLE_FRAMES and enough_time:
                    new_state = COMMANDS[gesture]
                    if new_state != state:
                        state = new_state
                        print(f"控制动作：{gesture} -> {state}")
                    last_action_time = time.time()
                    stable_count = 0
            else:
                candidate = None
                stable_count = 0

            if best_box:
                x1, y1, x2, y2 = best_box
                box_color = (0, 220, 0) if valid_command else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{gesture} {confidence:.2f}"
                cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            panel_color = STATE_COLORS[state]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 105), (25, 25, 25), -1)
            cv2.putText(frame, f"SYSTEM: {state}", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, panel_color, 2)
            hand_text = handedness or "None"
            cv2.putText(frame, f"Hand: {hand_text}  Allowed: {ALLOWED_HAND}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
            progress = min(stable_count / STABLE_FRAMES, 1.0)
            cv2.rectangle(frame, (20, 84), (220, 96), (80, 80, 80), -1)
            cv2.rectangle(frame, (20, 84), (20 + int(200 * progress), 96), panel_color, -1)
            cv2.putText(frame, "Q: quit   H: switch allowed hand", (250, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

            cv2.imshow("YOLO Gesture Automation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                ALLOWED_HAND = "Left" if ALLOWED_HAND == "Right" else "Right"

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
