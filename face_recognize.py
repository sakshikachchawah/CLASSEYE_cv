
"""
# face_recognize.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from datetime import datetime
import onnxruntime as ort
import sys
import csv

# ------------------- Get Class Code from Argument -------------------
if len(sys.argv) < 2:
    print("Error: Please provide class code as argument")
    sys.exit(1)

class_code = sys.argv[1]

# ------------------- ONNX Model Setup -------------------
model_path = "arcface.onnx"
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = (img.astype(np.float32) - 127.5) / 128.0
    return img[np.newaxis, ...]

def get_embedding(face_img):
    emb = sess.run([output_name], {input_name: preprocess(face_img)})[0][0]
    return emb / np.linalg.norm(emb)

# ------------------- Load Student Embeddings -------------------
embedding_folder = "embeddings"
students = []
for file in os.listdir(embedding_folder):
    if file.endswith(".pkl"):
        with open(os.path.join(embedding_folder, file), "rb") as f:
            students.append(pickle.load(f))

attendance = {}

# ------------------- MediaPipe Face Detection -------------------
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    last_marked = {}  # To avoid multiple marking in short time
    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                emb = get_embedding(face_crop)
                matched_name = "Unknown"
                min_dist = 0.5  # distance threshold

                for student in students:
                    for e in student['embeddings']:
                        dist = np.linalg.norm(emb - e)
                        if dist < min_dist:
                            min_dist = dist
                            matched_name = student['name']

                # Attendance marking (once per minute)
                now = datetime.now()
                if matched_name != "Unknown":
                    last_time = last_marked.get(matched_name, datetime.min)
                    if (now - last_time).seconds >= 60:
                        attendance[matched_name] = now.strftime("%Y-%m-%d %H:%M:%S")
                        last_marked[matched_name] = now
                        print(f"Attendance marked for {matched_name} at {attendance[matched_name]}")

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, matched_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Camera running message
        cv2.putText(frame, f"Class Code: {class_code}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        # Stop camera on ESC or after 10 minutes
        if key == 27 or (datetime.now() - start_time).seconds >= 600:
            break

cap.release()
cv2.destroyAllWindows()

# ------------------- Save Attendance CSV -------------------
csv_file = f"attendance_{class_code}.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])
    for name, time in attendance.items():
        writer.writerow([name, time])

print(f"Attendance saved to {csv_file}")
"""
# face_recognize.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from datetime import datetime
import onnxruntime as ort
import sys
import csv

# ------------------- Get Class Code from Argument -------------------
if len(sys.argv) < 2:
    print("Error: Please provide class code as argument")
    sys.exit(1)

class_code = sys.argv[1]

# ------------------- ONNX Model Setup -------------------
model_path = "arcface.onnx"
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = (img.astype(np.float32) - 127.5) / 128.0
    return img[np.newaxis, ...]

def get_embedding(face_img):
    emb = sess.run([output_name], {input_name: preprocess(face_img)})[0][0]
    return emb / np.linalg.norm(emb)

# ------------------- Load Student Embeddings -------------------
embedding_folder = "embeddings"
students = []
for file in os.listdir(embedding_folder):
    if file.endswith(".pkl"):
        with open(os.path.join(embedding_folder, file), "rb") as f:
            students.append(pickle.load(f))

attendance = {}

# ------------------- MediaPipe Face Detection -------------------
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    last_marked = {}  # To avoid multiple marking in short time
    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        message_text = ""  # Reset message each frame

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue

                emb = get_embedding(face_crop)
                matched_name = "Unknown"
                min_dist = 0.5  # distance threshold

                for student in students:
                    for e in student['embeddings']:
                        dist = np.linalg.norm(emb - e)
                        if dist < min_dist:
                            min_dist = dist
                            matched_name = student['name']

                # Attendance marking (once per minute)
                now = datetime.now()
                if matched_name != "Unknown":
                    last_time = last_marked.get(matched_name, datetime.min)
                    if (now - last_time).seconds >= 60:
                        attendance[matched_name] = now.strftime("%Y-%m-%d %H:%M:%S")
                        last_marked[matched_name] = now
                        print(f"Attendance marked for {matched_name} at {attendance[matched_name]}")
                        message_text = f"{matched_name} present. Now you can move."

                # Draw rectangle and name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, matched_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Show attendance message just below the bounding box
                if message_text:
                    cv2.putText(frame, message_text, (x, y+h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display class code on top-left
        cv2.putText(frame, f"Class Code: {class_code}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        # Stop camera on ESC or after 10 minutes
        if key == 27 or (datetime.now() - start_time).seconds >= 600:
            break

cap.release()
cv2.destroyAllWindows()

# ------------------- Save Attendance CSV -------------------
csv_file = f"attendance_{class_code}.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])
    for name, time in attendance.items():
        writer.writerow([name, time])

print(f"Attendance saved to {csv_file}")

