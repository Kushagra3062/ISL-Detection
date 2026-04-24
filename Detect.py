import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# -------------------- CONFIG --------------------
MODEL_PATH = r'models\Dynamic.h5'
ACTIONS = np.array(['hello', 'thanks', 'bye', 'good', 'congrats'])
SEQ_LEN = 20            # <<< set this to the length you TRAINED with (e.g., 30). If you trained with 20, set 20.
THRESHOLD = 0.55        # a bit forgiving for real-time
SMOOTH_WINDOW = 10      # number of recent predictions to consider
MAJORITY_MIN = 7        # how many in the window must agree to append to sentence
BAR_MAX_W = 300         # pixel width of confidence bars
FONT = cv2.FONT_HERSHEY_COMPLEX

# Optional color per action (ensure same length as ACTIONS)
COLORS = [
    (245,117,16),
    (117,245,16),
    (16,117,245),
    (117,16,245),
    (16,245,117),
][:len(ACTIONS)]

# -------------------- LOAD MODEL --------------------
model = load_model(MODEL_PATH)

# -------------------- MEDIAPIPE --------------------
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# -------------------- HELPERS --------------------
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
    return np.concatenate([pose, face, lh, rh])  # 1662 total if all included

def prob_viz(probs, actions, frame, colors):
    """
    Draw horizontal bars representing real-time gesture confidence probabilities.
    """
    if probs is None:
        return frame
    out = frame.copy()
    start_x, start_y = 10, 60
    h = 28
    for i, p in enumerate(probs):
        p = float(np.clip(p, 0.0, 1.0))
        y1 = start_y + i* (h + 12)
        y2 = y1 + h
        x2 = start_x + int(p * BAR_MAX_W)
        cv2.rectangle(out, (start_x, y1), (start_x + BAR_MAX_W, y2), (60,60,60), 1)
        cv2.rectangle(out, (start_x, y1), (x2, y2), colors[i % len(colors)], -1)
        cv2.putText(out, f'{actions[i]}: {p:.2f}', (start_x + BAR_MAX_W + 10, y2-6), FONT, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return out

def main():
    # -------------------- RUNTIME BUFFERS --------------------
    sequence = deque(maxlen=SEQ_LEN)  # keeps order oldest->newest
    pred_history = deque(maxlen=SMOOTH_WINDOW)
    sentence = deque(maxlen=5)
    last_res = np.zeros(len(ACTIONS), dtype=np.float32)  # until first prediction

    # -------------------- CAPTURE LOOP --------------------
    cap = cv2.VideoCapture(0)
    # Optional: set camera resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, result = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, result)

            # Collect keypoints
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)

            # Predict when we have exactly SEQ_LEN frames
            if len(sequence) == SEQ_LEN:
                # shape -> (1, SEQ_LEN, feature_dim)
                input_seq = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                res = model.predict(input_seq, verbose=0)[0]  # probs vector
                last_res = res

                top_idx = int(np.argmax(res))
                top_prob = float(res[top_idx])

                if top_prob > THRESHOLD:
                    pred_history.append(top_idx)

                    # Majority vote: require at least MAJORITY_MIN of last window agree with the last idx
                    if pred_history.count(pred_history[-1]) >= MAJORITY_MIN:
                        word = ACTIONS[pred_history[-1]]
                        if len(sentence) == 0 or sentence[-1] != word:
                            sentence.append(word)

            # UI: probs and sentence
            image = prob_viz(last_res, ACTIONS, image, COLORS)

            # Header bar
            cv2.rectangle(image, (0, 0), (image.shape[1], 44), (245,117,16), -1)
            cv2.putText(image, ' '.join(list(sentence)), (10, 30), FONT, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('ISL Dynamic Gestures (Webcam)', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
