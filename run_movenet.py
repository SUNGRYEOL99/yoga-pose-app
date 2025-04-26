import tensorflow as tf
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

# 한글 폰트 경로 설정 (Mac 기준)
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

# MoveNet 모델 로드
interpreter = tf.lite.Interpreter(model_path="movenet/4.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 키포인트 → 관절 좌표 변환
def get_point(keypoints, index, image_shape):
    y, x, c = keypoints[index]
    h, w, _ = image_shape
    return (int(x * w), int(y * h)), c

# 관절 각도 계산
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# 한글 텍스트 출력 함수 (Pillow 사용)
def draw_korean_text(img, text, position, font_size=28, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 키포인트 연결 선 정의 (MoveNet 기준)
SKELETON = [
    (5, 7), (7, 9),       # 왼팔
    (6, 8), (8, 10),      # 오른팔
    (11, 13), (13, 15),   # 왼다리
    (12, 14), (14, 16),   # 오른다리
    (5, 6),               # 어깨
    (11, 12),             # 엉덩이
    (5, 11), (6, 12)      # 몸통
]

# 키포인트 연결 선 그리기
def draw_skeleton(img, keypoints, threshold=0.3):
    h, w, _ = img.shape
    for p1, p2 in SKELETON:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > threshold and c2 > threshold:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

# MoveNet 실행 함수
def movenet(image):
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints[0][0]

# 키포인트 인덱스 (MoveNet 기준)
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = movenet(image)

    # 관절 점 찍기
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:
            h, w, _ = frame.shape
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 선 연결
    draw_skeleton(frame, keypoints)

    # 왼팔 각도 계산 및 피드백
    p1, c1 = get_point(keypoints, LEFT_SHOULDER, frame.shape)
    p2, c2 = get_point(keypoints, LEFT_ELBOW, frame.shape)
    p3, c3 = get_point(keypoints, LEFT_WRIST, frame.shape)

    if c1 > 0.3 and c2 > 0.3 and c3 > 0.3:
        angle = calculate_angle(p1, p2, p3)

        # 각도 텍스트
        frame = draw_korean_text(frame, f"왼팔 각도: {int(angle)}도", (20, 30), font_size=24)

        # 피드백 메시지
        if angle < 140:
            frame = draw_korean_text(frame, "왼팔을 더 펴세요!", (20, 60), font_size=28, color=(255, 0, 0))
        else:
            frame = draw_korean_text(frame, "좋은 자세입니다!", (20, 60), font_size=28, color=(0, 255, 0))

    # 결과 출력
    cv2.imshow("요가 자세 피드백", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
