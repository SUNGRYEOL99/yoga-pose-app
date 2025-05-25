import cv2
import numpy as np
import tensorflow as tf
import json
import os
import platform
import time
from PIL import ImageFont, ImageDraw, Image
from feedback_utils import speak, stop_speaking, generate_ai_feedback, is_speaking

MODEL_PATH = "movenet/4.tflite"
BASE_TEMPLATE_PATH = "json_file/pose_template.json"
#운영체제 인지해서 경로설정
if platform.system() == "Windows":
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == "Darwin":  # Mac OS
    FONT_PATH = "/System/Library/Fonts/AppleGothic.ttf"

os.makedirs("captures", exist_ok=True)

def load_base_angles(path, frame_num=1):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for frame in data:
        if frame["frame"] == frame_num:
            return frame.get("angles", {})
    return {}

def get_match_ratio(base, current, tolerance=10):
    matched = 0
    total = len(base)
    for joint, base_angle in base.items():
        now = current.get(joint)
        if now is not None and abs(now - base_angle) <= tolerance:
            matched += 1
    return matched / total if total > 0 else 0

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

ANGLE_DEFINITIONS = {
    "left_elbow": [5, 7, 9],
    "right_elbow": [6, 8, 10],
    "left_knee": [11, 13, 15],
    "right_knee": [12, 14, 16],
    "back": [5, 11, 15]
}

POSE_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (0, 1), (1, 3), (0, 2), (2, 4)
]

def get_angles(keypoints, shape):
    angles = {}
    for name, idx in ANGLE_DEFINITIONS.items():
        p1, c1 = get_point(keypoints, idx[0], shape)
        p2, c2 = get_point(keypoints, idx[1], shape)
        p3, c3 = get_point(keypoints, idx[2], shape)
        if min(c1, c2, c3) > 0.3:
            angles[name] = int(calculate_angle(p1, p2, p3))
    return angles

def compare_angles(base, current):
    result = {}
    for joint, base_angle in base.items():
        now = current.get(joint)
        if now is not None:
            result[joint] = abs(base_angle - now)
    return result

def draw_progress_bar(img, progress, position=(20, 20), size=(300, 30)):
    x, y = position
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), -1)
    fill_width = int((progress / 100) * w)
    cv2.rectangle(img, (x, y), (x + fill_width, y + h), (255, 128, 0), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.putText(img, f"{progress}%", (x + w + 10, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img

def draw_korean_text(img, text, position, font_size=22, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)

# 웹캠 해상도 설정 / 화면 비율 수정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_num = 1
total_frames = 2
base_angles = load_base_angles(BASE_TEMPLATE_PATH, frame_num=frame_num)
last_feedback_time = time.time()
feedback_interval = 5

print("▶ 웹캠 실행 중 ('Q'로 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #카메라 좌우 반전
    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

    current_angles = get_angles(keypoints, frame.shape)
    angle_diff = compare_angles(base_angles, current_angles)
    match_ratio = get_match_ratio(base_angles, current_angles)

    h, w, _ = frame.shape
    current_points = []
    for i, (y, x, c) in enumerate(keypoints):
        if c > 0.3:
            pt = (int(x * w), int(y * h))
            current_points.append((i, pt))
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

    for pair in POSE_CONNECTIONS:
        pt1 = next((p[1] for p in current_points if p[0] == pair[0]), None)
        pt2 = next((p[1] for p in current_points if p[0] == pair[1]), None)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # 각도 텍스트 부분
    y_offset = 50
    x_offset = frame.shape[1] - 350  # 화면 오른쪽에서 왼쪽으로 350px 들어온 위치

    for joint, diff in angle_diff.items():
        color = (0, 255, 0) if diff <= 10 else (0, 0, 255)
        frame = draw_korean_text(frame, f"{joint} 차이: {diff}°", (x_offset, y_offset), font_size=35, color=color)
        y_offset += 30

    for joint, angle in current_angles.items():
        idx = ANGLE_DEFINITIONS[joint][1]
        pt, c = get_point(keypoints, idx, frame.shape)
        if c > 0.3:
            frame = draw_korean_text(frame, f"{angle}°", pt, font_size=35, color=(255, 255, 0))

    if frame_num <= total_frames:
        progress = int(((frame_num - 1) / total_frames) * 100)
        frame = draw_progress_bar(frame, progress, position=(20, 100), size=(700, 55))
        frame = draw_korean_text(frame, f"진행도: {progress}%", (20, 20), font_size=35, color=(255, 255, 255))

    current_time = time.time()
    if current_time - last_feedback_time > feedback_interval:
        if not is_speaking:
            feedback_msg = generate_ai_feedback(angle_diff, match_ratio)
            print("▶ AI 피드백:", feedback_msg)
            speak(feedback_msg)
            last_feedback_time = current_time

    if match_ratio >= 0.8:
        if is_speaking:
            stop_speaking()
            speak("좋아요. 다음 자세로 넘어갑니다.")
            cv2.waitKey(1500)

        cv2.imwrite(f"captures/frame_{frame_num}.jpg", frame)
        frame_num += 1
        base_angles = load_base_angles(BASE_TEMPLATE_PATH, frame_num=frame_num)
        if not base_angles:
            print(f"모든 동작 완료! 총 {frame_num - 1}개 프레임 확인됨.")
            break
        continue

    cv2.imshow("자세 비교 피드백", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
