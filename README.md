👁️ ClassEYE

Smart Student Attendance System using Face Recognition.
ClassEYE is a modern, automated solution for tracking student attendance. By leveraging Deep Learning (ArcFace) and Computer Vision, it eliminates manual roll calls and provides a seamless, secure login and tracking experience via a Streamlit web interface.

✨ Key Features

Dual-Role Dashboard: Separate workflows for Admin (Student Enrollment) and Class-End (Attendance Marking).
ArcFace Integration: Uses High-precision arcface.onnx models for generating facial embeddings.
Real-time Recognition: Powered by MediaPipe for fast face detection and OpenCV for live camera feedback.
Automated Logging: Saves student details and attendance timestamps directly into CSV files.
Class-Specific Sessions: Start attendance by entering unique class codes.

🛠️ Tech Stack

Frontend: Streamlit
Face Detection: MediaPipe
Feature Extraction: ArcFace (via ONNX Runtime)
Database: CSV & Pickle (for embeddings)
Core: Python, OpenCV, NumPy


🚀 Getting Started

Clone the repo:

git clone https://github.com/sakshikachchawah/CLASSEYE_cv.git

cd CLASSEYE_cv

Install Requirements:

pip install streamlit opencv-python mediapipe onnxruntime numpy

Add the Model: Place your arcface.onnx model file in the root directory.

Run the App:

streamlit run app.py

📝 Usage

Login as Admin (Default: admin / admin123) to enroll new students by capturing their faces.
Login as Class-End and enter a Class Code to start the 10-minute automated attendance window.

Check attendance_[CODE].csv for the generated attendance reports.
