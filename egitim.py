# Model Eğitimi
"""from ultralytics import YOLO

# Modeli yükleyin (önceden eğitilmiş bir model veya sıfırdan eğitim)
model = YOLO("yolov8n.pt")

# Eğitim verisini belirtin
model.train(data="data.yaml", epochs=50)
"""

#sesli uyari ile
from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
from gtts import gTTS
import pygame
import time
import threading
from io import BytesIO

app = Flask(__name__)
 
# Model yükle
model = YOLO("D:/3_sinif_bahar/Uygulama_Tasarimi/trafik.v20i.yolov7pytorch/runs/detect/train/weights/best.pt")  # Kendi model yolunu yaz

# Kamera
cap = cv2.VideoCapture(0)

# Ses ayarları
pygame.mixer.init()
speech_lock = threading.Lock()
last_detected = {}
CONFIDENCE_THRESHOLD = 0.85
MIN_SPEECH_INTERVAL = 4.0  # Aynı nesne için minimum 2 saniye bekle


def speak(text):
    def _speak():
        try:
            # Türkçe ses oluştur (online)
            tts = gTTS(text=text, lang='tr', slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            # Ses oynat
            with speech_lock:
                pygame.mixer.music.load(audio_bytes, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print("🔇 Ses hatası:", e)

    threading.Thread(target=_speak, daemon=True).start()


def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        current_time = time.time()
        results = model(frame)
        current_detections = set()

        # Tespit edilen nesneleri kaydet
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= CONFIDENCE_THRESHOLD:
                    class_name = model.names[int(box.cls[0])]
                    current_detections.add(class_name)

                    # Yeni tespit edildiyse sesli uyarı ver (2 saniye bekle)
                    if class_name not in last_detected or (current_time - last_detected[class_name]['time']) > MIN_SPEECH_INTERVAL:
                        speak(f"{class_name} algılandı")
                        last_detected[class_name] = {'time': current_time, 'alerted': True}

                    # Görsel işaretleme
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {box.conf[0]:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Video akışı
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
        pygame.mixer.quit()
