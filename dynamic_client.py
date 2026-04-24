import os
# Suppress TensorFlow/MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import requests
import json
from collections import deque

API_URL = "http://127.0.0.1:5000/predict_dynamic"
SEQ_LEN = 20
THRESHOLD = 0.7  
FONT = cv2.FONT_HERSHEY_COMPLEX

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def mediapipe_detection(image, mp_model):
    """
    Process an image using a MediaPipe model (Holistic).
    Converts the internal color space from BGR to RGB for processing,
    then back to BGR for output visualization.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = mp_model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_out, results

def draw_styled_landmarks(image, results):
    """
    Draw default styled landmarks over the webcam feed.
    Tracks face meshes, left and right hands, and full body pose.
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

def extract_keypoints(results):
    """
    Extract coordinate values (x, y, z, visibility) for all detected landmarks.
    Concatenates them into a flat unified numpy array for sequence prediction.
    Zero-pads when a specific landmark grouping isn't visible.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def main():
    print("Initializing VideoCapture(0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    else:
        print("Webcam opened successfully.")

    sequence = deque(maxlen=SEQ_LEN)
    
    current_prediction = "Waiting..."
    current_confidence = 0.0
    latency = "0ms"

    print(f"Starting client... Connecting to {API_URL}")

    print("Initializing MediaPipe Holistic...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("MediaPipe Holistic initialized. Starting loop...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints.tolist()) 
            
            if len(sequence) == SEQ_LEN:
                try:
                    payload = {'sequence': list(sequence)}
                    headers = {'Content-Type': 'application/json'}
                    
                    resp = requests.post(API_URL, json=payload, headers=headers, timeout=1.0)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        current_prediction = data.get('gesture', 'Unknown')
                        current_confidence = data.get('confidence', 0.0)
                        latency = data.get('latency', '0ms')
                    else:
                        print(f"API Error: {resp.status_code} - {resp.text}")
                        
                except Exception as e:
                    print(f"Connection Error: {e}")

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            text = f"{current_prediction} ({current_confidence:.2f}) Latency: {latency}"
            cv2.putText(image, text, (3,30), FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Dynamic Gesture Client', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
