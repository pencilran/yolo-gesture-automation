from datetime import datetime
from pathlib import Path

import cv2

save_dir = Path("dataset/images")
save_dir.mkdir(parents=True, exist_ok=True)

classes = {
    ord("1"): "palm",
    ord("2"): "fist",
    ord("3"): "thumb_up",
    ord("4"): "background",
}

current_class = "palm"
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("摄像头打不开，请确认没有被其他程序占用。")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    cv2.putText(frame, f"Class: {current_class}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "1:palm  2:fist  3:thumb_up  4:background",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "S: save one photo   Q: quit",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("YOLO Gesture Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in classes:
        current_class = classes[key]
    elif key == ord("s"):
        filename = f"{current_class}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(save_dir / filename), clean_frame)
        print(f"已保存：{filename}")
    elif key in (ord("q"), 27):
        break

camera.release()
cv2.destroyAllWindows()
