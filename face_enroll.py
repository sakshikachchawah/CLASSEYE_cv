"""
# face_enroll.py
import cv2
import mediapipe as mp
import numpy as np
import os
import onnxruntime as ort
import pickle

# ------------------- ONNX Model Setup -------------------
model_path = "arcface.onnx"  # put your model path
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
    emb = emb / np.linalg.norm(emb)
    return emb

# ------------------- MediaPipe Face Detection -------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ------------------- Dataset Path -------------------
dataset_path = "embeddings"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# ------------------- OpenCV Camera -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# ------------------- Input Student Info -------------------
# Instead of tkinter, use OpenCV window for input message
name = input("Enter student name: ")
enroll = input("Enter enrollment number: ")
batch = input("Enter batch year: ")
contact = input("Enter contact: ")
branch = input("Enter branch: ")
section = input("Enter section: ")
group = input("Enter group: ")

student_file = os.path.join(dataset_path, f"{name}_{enroll}.pkl")

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    count = 0
    embeddings_list = []
    message_displayed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size != 0:
                    emb = get_embedding(face_crop)
                    embeddings_list.append(emb)
                    count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display message inside OpenCV window
        if len(embeddings_list) > 0:
            cv2.putText(frame, "Face detected. Press 's' to save embeddings.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            message_displayed = True
        else:
            cv2.putText(frame, "Please align your face in camera.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enroll Face", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to stop
            break
        elif key == ord('s') and len(embeddings_list) > 0:  # S to save
            data = {"name": name, "enroll": enroll, "batch": batch, "contact": contact,
                    "branch": branch, "section": section, "group": group,
                    "embeddings": embeddings_list}
            with open(student_file, "wb") as f:
                pickle.dump(data, f)

            # Save student details to CSV
            import csv
            csv_file = "student_details.csv"
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                if not file_exists:
                    writer.writerow(["Name", "Enroll", "Batch", "Contact", "Branch", "Section", "Group"])
                writer.writerow([name, enroll, batch, contact, branch, section, group])

            cv2.putText(frame, "Face saved successfully! You can save details now.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Enroll Face", frame)
            cv2.waitKey(2000)  # show message for 2 seconds
            embeddings_list = []  # clear after save

cap.release()
cv2.destroyAllWindows()
"""
# face_enroll.py
import cv2
import mediapipe as mp
import numpy as np
import os
import onnxruntime as ort
import pickle
import sys
import csv

# ------------------- Read Student Info from Arguments -------------------
# Usage: python face_enroll.py <name> <enroll> <batch> <contact> <branch> <section> <group>
if len(sys.argv) < 8:
    print("Error: Please provide all student info as command-line arguments")
    sys.exit(1)

name = sys.argv[1]
enroll = sys.argv[2]
batch = sys.argv[3]
contact = sys.argv[4]
branch = sys.argv[5]
section = sys.argv[6]
group = sys.argv[7]

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
    emb = emb / np.linalg.norm(emb)
    return emb

# ------------------- MediaPipe -------------------
mp_face_detection = mp.solutions.face_detection

dataset_path = "embeddings"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

student_file = os.path.join(dataset_path, f"{name}_{enroll}.pkl")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    embeddings_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(int(bboxC.xmin * iw), 0)
                y = max(int(bboxC.ymin * ih), 0)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size != 0:
                    emb = get_embedding(face_crop)
                    embeddings_list.append(emb)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if len(embeddings_list) > 0:
            cv2.putText(frame, "Face detected. Press 's' to save.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Align your face in camera.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enroll Face", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord('s') and len(embeddings_list) > 0:
            # Save embeddings
            data = {"name": name, "enroll": enroll, "batch": batch, "contact": contact,
                    "branch": branch, "section": section, "group": group,
                    "embeddings": embeddings_list}
            with open(student_file, "wb") as f:
                pickle.dump(data, f)

            # Save student info to CSV
            csv_file = "student_details.csv"
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, "a", newline="") as csvf:
                writer = csv.writer(csvf)
                if not file_exists:
                    writer.writerow(["Name", "Enroll", "Batch", "Contact", "Branch", "Section", "Group"])
                writer.writerow([name, enroll, batch, contact, branch, section, group])

            cv2.putText(frame, "Face & details saved! You can quit now.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Enroll Face", frame)
            cv2.waitKey(2000)
            embeddings_list = []

cap.release()
cv2.destroyAllWindows()