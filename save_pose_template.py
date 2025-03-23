import tensorflow as tf
import numpy as np
import cv2
import json
import os

# MoveNet 모델 로딩
interpreter = tf.lite.Interpreter(model_path="movenet/4.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 각도 계산용 키포인트 정의
ANGLE_DEFINITIONS = {
    "left_elbow": [5, 7, 9],
    "right_elbow": [6, 8, 10],
    "left_knee": [11, 13, 15],
    "right_knee": [12, 14, 16],
    "back": [5, 11, 15]
}

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# 키포인트를 이미지 좌표로 변환
def get_point(keypoints, index, shape):
    y, x, c = keypoints[index]
    h, w, _ = shape
    return (int(x * w), int(y * h)), c

# 각도 측정 함수
def get_angles(keypoints, shape):
    angles = {}
    for name, idx in ANGLE_DEFINITIONS.items():
        p1, c1 = get_point(keypoints, idx[0], shape)
        p2, c2 = get_point(keypoints, idx[1], shape)
        p3, c3 = get_point(keypoints, idx[2], shape)
        if min(c1, c2, c3) > 0.3:
            angles[name] = int(calculate_angle(p1, p2, p3))
    return angles

# 웹캠 시작
cap = cv2.VideoCapture(0)

print("✅ 원하는 자세를 잡고 's' 키를 누르면 기준 JSON이 저장됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 관절 및 각도 표시
    for name, idx in ANGLE_DEFINITIONS.items():
        p1, c1 = get_point(keypoints, idx[0], frame.shape)
        p2, c2 = get_point(keypoints, idx[1], frame.shape)
        p3, c3 = get_point(keypoints, idx[2], frame.shape)

        if min(c1, c2, c3) > 0.3:
            # 선 연결
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
            cv2.line(frame, p3, p2, (255, 0, 0), 2)

            # 원 그리기
            cv2.circle(frame, p1, 5, (0, 255, 0), -1)
            cv2.circle(frame, p2, 5, (0, 255, 0), -1)
            cv2.circle(frame, p3, 5, (0, 255, 0), -1)

            # 각도 표시
            angle = int(calculate_angle(p1, p2, p3))
            cv2.putText(frame, f'{angle}°', (p2[0] + 10, p2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("기준 자세 캡처기", frame)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('s'):
        # 각도 측정 및 저장
        angles = get_angles(keypoints, frame.shape)
        template = {
            "name": input("💾 저장할 자세 이름: "),
            "angles": angles,
            "tolerance": 15
        }
        os.makedirs("pose_data", exist_ok=True)
        filename = f"pose_data/{template['name']}.json"
        with open(filename, "w") as f:
            json.dump(template, f, indent=2)
        print(f"✅ {filename} 저장 완료!")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
