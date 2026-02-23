from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from src.utils.speech_to_text import GestureViewer
from src.utils.Processing import HandGestureDetector
import speech_recognition as sr
import io
from pydub import AudioSegment
from pydub.utils import which
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

detector = HandGestureDetector()
gv = GestureViewer(socketio)

import tensorflow as tf 
import time

MODEL_PATH = '../models/dynamic.tflite'
CLASSES = ["hello", "thanks", "bye", "good", "congrats"]

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info("Dynamic TFLite model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    raise

SEQUENCE_LENGTH = 20   
FEATURES = 1662       
CONFIDENCE_THRESHOLD = 0.7

@app.route('/predict_dynamic', methods=['POST'])
def predict_dynamic():
    try:
        start_time = time.time() 
        
        data = request.json.get('sequence')
        if not data or len(data) != SEQUENCE_LENGTH:
            return jsonify({'error': 'Invalid sequence length'}), 400

        input_data = np.array(data, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, FEATURES)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx])

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Inference Latency: {latency_ms:.2f}ms")

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'gesture': 'Unknown', 'confidence': confidence, 'latency': latency_ms})

        predicted_class = CLASSES[predicted_idx]
        return jsonify({
            'gesture': predicted_class, 
            'confidence': confidence,
            'latency': f"{latency_ms:.2f}ms"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = detector.process_frame(frame)
    return jsonify(result)

r = sr.Recognizer()

@socketio.on("audio_utterence")
def speech_to_gesture(data):
    try:
        audio_bytes = bytes(data["buffer"])
        webm_io = io.BytesIO(audio_bytes)

        audio_seg = AudioSegment.from_file(webm_io, format="webm")
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)

        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            gv.display_gesture(text)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            text = f"Speech service error: {e}"

        socketio.emit("recognized_text", {'text': text.lower()})
    except Exception as e:
        print("Error processing audio:", e)
        socketio.emit("recognized_text", {"text": ""})

# ----------------- Main ---------------------
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
