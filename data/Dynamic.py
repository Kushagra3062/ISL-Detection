import cv2
import numpy as np
import mediapipe as mp
import os

# -------------------- MEDIAPIPE --------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh 
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# -------------------- HELPERS --------------------
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

def extract_keypoints(results):
    if results.pose_landmarks:
        pose_orig = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        center_pose = (pose_orig[11][:3] + pose_orig[12][:3]) / 2
        pose_coords = pose_orig[:, :3] - center_pose
        pose = np.hstack([pose_coords, pose_orig[:, 3:]]).flatten()
    else:
        pose = np.zeros(33*4)

    if results.face_landmarks:
        face_orig = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        center_face = face_orig[1]
        face = (face_orig - center_face).flatten()
    else:
        face = np.zeros(468*3)

    if results.left_hand_landmarks:
        lh_orig = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        center_lh = lh_orig[0]

        dist_lh = np.linalg.norm(lh_orig[0] - lh_orig[9]) + 1e-6 
        lh = ((lh_orig - center_lh) / dist_lh).flatten()
    else:
        lh = np.zeros(21*3)

    if results.right_hand_landmarks:
        rh_orig = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        center_rh = rh_orig[0]
        dist_rh = np.linalg.norm(rh_orig[0] - rh_orig[9]) + 1e-6
        rh = ((rh_orig - center_rh) / dist_rh).flatten()
    else:
        rh = np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

# -------------------- SETTINGS --------------------
DATA_PATH = os.path.join('data\MP_Data')
actions = np.array(['hello', 'thanks', 'bye', 'good', 'congrats'])
no_sequences = 50        
sequence_length = 20     # frames per sequence

# Folder structure
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# -------------------- CAPTURE LOOP --------------------
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    for action in actions:
        print(f"\n👉 Ready for action: {action}")
        cv2.waitKey(1000)

        for sequence in range(no_sequences):
            print(f"   Waiting to record sequence {sequence} for {action}... (press 'r')")
            
            # Wait until user presses 'r'
            while True:
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                cv2.putText(image, f"Action: {action} | Sequence: {sequence}", (20,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(image, "Press 'r' to start recording", (20,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('r'):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):   # quit option
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Start recording frames
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: continue
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                cv2.putText(image, f"Recording {action} | Seq {sequence} | Frame {frame_num}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            print(f"   ✅ Recorded sequence {sequence} for {action}")

cap.release()
cv2.destroyAllWindows()
