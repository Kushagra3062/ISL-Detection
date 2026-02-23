import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from src.utils.hands_utils import get_both_hands
from src.utils.finger_utils import finger_open, get_center, distance

class HandGestureDetector:
    def __init__(self, model_path="../models/landmark_model.h5"):
        
        self.mp_hands = mp.solutions.hands
        self.model = load_model(model_path)

        self.classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z']

        self.last_pred = None
        self.word = ""
        self.buffer = ""
        self.gesture_count = 0
        self.prev_x = None
        self.movement_threshold = 40
        self.last_time = 0
        self.cooldown = 0.25
        self.space_count = 0
        self.SPACE_FRAME = 5
        self.SPACE_DISPLAY_DURATION = 1.0
        self.space_added_time = 0
        self.prediction = None
       
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def delete_word(self):
        if len(self.word) > 0:
            self.buffer = self.word[-1] + self.buffer
            self.word = self.word[:-1]

    def undo_word(self):
        if len(self.buffer) > 0:
            self.word += self.buffer[0]
            self.buffer = self.buffer[1:]

    def add_space(self):
        if not self.word.endswith(" "):
            self.word += " "
            print("Space added:", repr(self.word))

    def process_frame(self, frame):
        """
        Process a single BGR frame (OpenCV image)
        Returns: dict with prediction, word, and confidence
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                dist = distance(thumb_tip, index_tip)

                if dist < 0.04:
                    if finger_open(hand_landmarks) == 3:
                        self.space_count += 1
                        if self.space_count == self.SPACE_FRAME:
                            self.add_space()
                            self.prediction = "Space"
                            self.space_added_time = time.time()
                            self.space_count = 0
                            return {
                            "prediction": self.prediction,
                            "confidence": 1.0,
                            "word": self.word
                            }
                else:
                    self.space_count = 0

                hand = handedness.classification[0].label
                
                cen_x, cen_y = get_center(hand_landmarks, w, h)

                if hand == 'Left':
                    if finger_open(hand_landmarks) >= 4:
                        if self.prev_x is not None:
                            if (cen_x - self.prev_x) > self.movement_threshold and (time.time() - self.last_time > self.cooldown):
                                self.delete_word()
                                self.last_time = time.time()
                                return {
                                "prediction": "word deleted",
                                "confidence": 1.0,
                                "word": self.word
                                }
                if hand == 'Right':
                    if finger_open(hand_landmarks) >= 4:
                        if self.prev_x is not None:
                            if (self.prev_x - cen_x) > self.movement_threshold and (time.time() - self.last_time > self.cooldown):
                                self.undo_word()
                                self.last_time = time.time()
                                return {
                                "prediction": "delete undo",
                                "confidence": 1.0,
                                "word": self.word
                                }

                self.prev_x = cen_x

            # Predict gesture
            row = get_both_hands(results, frame)
            row_np = np.array(row).reshape(1, -1)
            pred_prob = self.model.predict(row_np)
            pred_idx = np.argmax(pred_prob)
            self.prediction = self.classes[pred_idx]
            confidence = float(pred_prob[0][pred_idx])

            if self.prediction == self.last_pred:
                self.gesture_count += 1
            else:
                self.last_pred = self.prediction
                self.gesture_count = 0

            if self.gesture_count == 20:
                self.word += self.last_pred
                self.gesture_count = 0

            return {
                "prediction": self.prediction,
                "confidence": confidence,
                "word": self.word
            }

        else:
            self.gesture_count = 0
            self.last_pred = None
            return {
                "prediction": None,
                "confidence": None,
                "word": self.word
            }
