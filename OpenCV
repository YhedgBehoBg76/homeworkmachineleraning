import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIMILARITY_THRESHOLD = 0.86
OWNER_IMAGE_PATH = os.path.join(SCRIPT_DIR, "owner.jpg")
FACE_MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# Модели
face_base = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(base_options=face_base, num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

hand_base = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Владелец
owner_img = cv2.imread(OWNER_IMAGE_PATH)
if owner_img is None:
    print("owner.jpg не найден!")
    exit()

owner_rgb = cv2.cvtColor(owner_img, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=owner_rgb)
result = face_landmarker.detect(mp_image)
owner_embedding = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks[0]]).flatten()
owner_embedding = owner_embedding / np.linalg.norm(owner_embedding)

def count_fingers(hand_landmarks):
    fingers = 0
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers += 1
    for tip in [8, 12, 16, 20]:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers += 1
    return fingers

# Камера
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_result = face_landmarker.detect(mp_image)

    owner_present = False

    if face_result.face_landmarks:
        face_lm = face_result.face_landmarks[0]
        emb = np.array([[lm.x, lm.y, lm.z] for lm in face_lm]).flatten()
        emb = emb / np.linalg.norm(emb)
        similarity = np.dot(owner_embedding, emb)

        if similarity > SIMILARITY_THRESHOLD:
            owner_present = True
            color = (0, 255, 0)
            text = "OWNER"
        else:
            color = (0, 100, 255)
            text = "UNKNOWN"

        x_coords = [int(lm.x * w) for lm in face_lm]
        y_coords = [int(lm.y * h) for lm in face_lm]
        cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), color, 3)
        cv2.putText(frame, text, (min(x_coords), min(y_coords)-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    if owner_present:
        hand_result = hand_detector.detect(mp_image)
        total_fingers = 0
        if hand_result.hand_landmarks:
            for hand_lm in hand_result.hand_landmarks:
                fingers = count_fingers(hand_lm)
                total_fingers += fingers
                
        cv2.putText(frame, f"Fingers: {total_fingers}", (40, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        cv2.putText(frame, "Owner not detected", (40, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(frame, "Press Q to exit", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face & Finger Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
