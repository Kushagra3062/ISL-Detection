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
from tensorflow.keras.models import load_model #type: ignore
import logging

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- FFMPEG ------------------
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# ----------------- Flask -------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------- Static Gesture -----------
detector = HandGestureDetector()
gv = GestureViewer(socketio)

import tensorflow as tf # Use tf.lite for inference
import time

# ... (rest of your imports remain the same)

# ----------------- Dynamic Model Setup (TFLite) ------
MODEL_PATH = '../models/dynamic.tflite'
CLASSES = ["hello", "thanks", "bye", "good", "congrats"]

try:
    # 1. Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details for inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info("Dynamic TFLite model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {e}")
    raise

SEQUENCE_LENGTH = 20   
FEATURES = 1662       
CONFIDENCE_THRESHOLD = 0.7

# ----------- DYNAMIC GESTURE ROUTE (Optimized) -------------
@app.route('/predict_dynamic', methods=['POST'])
def predict_dynamic():
    try:
        start_time = time.time() # Start Latency Profiler
        
        data = request.json.get('sequence')
        if not data or len(data) != SEQUENCE_LENGTH:
            return jsonify({'error': 'Invalid sequence length'}), 400

        # 2. Prepare Input Data (Optimize with float32)
        # Reshape to (1, 20, 1662)
        input_data = np.array(data, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, FEATURES)

        # 3. TFLite Inference Step
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 4. Get Output
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx])

        # Latency Calculation
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
# ----------- STATIC GESTURE ROUTE -------------
@app.route('/frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = detector.process_frame(frame)
    return jsonify(result)

# ----------- SPEECH TO GESTURE ROUTE -------------
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
