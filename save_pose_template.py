import tensorflow as tf
import numpy as np
import cv2
import os
import json

# 사진 경로 (사용자 설정)

IMAGE_PATH = "sample_image/test2.png"
OUTPUT_JSON = "json_file/pose_template2.json"

# 모델 로딩
interpreter = tf.lite.Interpreter(model_path="movenet/4.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ANGLE_DEFINITIONS = {
    "left_elbow": [5, 7, 9],
    "right_elbow": [6, 8, 10],
    "left_knee": [11, 13, 15],
    "right_knee": [12, 14, 16],
    "back": [5, 11, 15]
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, cb = a - b, c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def get_point(keypoints, index, shape):
    y, x, c = keypoints[index]
    h, w, _ = shape
    return (int(x * w), int(y * h)), c

def get_angles(keypoints, shape):
    angles = {}
    for name, idx in ANGLE_DEFINITIONS.items():
        p1, c1 = get_point(keypoints, idx[0], shape)
        p2, c2 = get_point(keypoints, idx[1], shape)
        p3, c3 = get_point(keypoints, idx[2], shape)
        if min(c1, c2, c3) > 0.3:
            angles[name] = int(calculate_angle(p1, p2, p3))
    return angles

# 이미지 로드
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"해당 이미지를 불러올 수 없습니다: {IMAGE_PATH}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192)
input_image = tf.cast(input_image, dtype=tf.uint8)

interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
interpreter.invoke()
keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

angles = get_angles(keypoints, image.shape)

# JSON 저장
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump([{"frame": 1, "angles": angles}], f, indent=2, ensure_ascii=False)

print(f"사진의 자세가 {OUTPUT_JSON} 에 저장되었습니다.")
