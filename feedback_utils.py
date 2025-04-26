import os
import queue
import threading
import tempfile
from gtts import gTTS
from playsound import playsound
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 전역 상태
tts_queue = queue.Queue()
is_speaking = False

# TTS 스레드 루프 (gTTS → mp3 저장 → playsound)
def tts_loop():
    global is_speaking
    while True:
        text = tts_queue.get()
        if text is None:
            break
        is_speaking = True
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                tts.save(fp.name)
                playsound(fp.name)
        except Exception as e:
            print(f"TTS 오류: {e}")
        is_speaking = False
        tts_queue.task_done()

# TTS 전용 백그라운드 스레드 시작
tts_thread = threading.Thread(target=tts_loop, daemon=True)
tts_thread.start()

# speak(): 말할 텍스트를 큐에 넣기
def speak(text):
    if not text:
        return
    tts_queue.put(text)

# 중단 요청 시 큐 비우기
def stop_speaking():
    global is_speaking
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
            tts_queue.task_done()
        except queue.Empty:
            break
    is_speaking = False

# GPT 기반 피드백 생성 함수
def generate_ai_feedback(angle_diff, match_ratio):
    if not angle_diff:
        return "자세 인식이 명확하지 않아요. 한 번만 더 자세를 맞춰보세요."

    joint_summary = ", ".join([f"{joint}: {diff}도" for joint, diff in angle_diff.items()])
    prompt = f'''
    요가 자세에 대한 실시간 피드백을 생성해주세요.

    조건:
    - 각 관절 오차: {joint_summary}
    - 자세 일치율: {int(match_ratio * 100)}%

    규칙:
    - 자연스럽고 부드러운 톤으로 한국어로 코칭해주세요.
    - 중요한 관절 한두 개를 언급하세요.
    - 피드백은 간결하고 이해하기 쉽게 그리고 너무 길지 않게 한 두 문장으로 부탁해요.
    '''

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
