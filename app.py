import cv2
import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# --- Directories ---
DATA_DIR = "faces"
MODEL_FILE = "face_model.yml"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# --- Streamlit UI ---
st.title("Face Recognition Attendance System")

menu = ["Register User", "Mark Attendance", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Helper Functions ---
def capture_faces(user_id, user_name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    st.info("Look at the camera...")

    while count < 20:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{DATA_DIR}/{user_id}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    st.success(f"Captured {count} face images for {user_name}")

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        user_id = int(file.split("_")[0])
        faces.append(img)
        ids.append(user_id)

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(MODEL_FILE)
        st.success("Model trained successfully!")

def mark_attendance():
    if not os.path.exists(MODEL_FILE):
        st.warning("No trained model found. Register users first!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    st.info("Press 'q' to exit camera after recognition")

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            user_id, conf = recognizer.predict(face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {user_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Mark attendance
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            df = pd.DataFrame()
            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_csv(ATTENDANCE_FILE)
            df = df.append({"ID": user_id, "Date": date_str, "Time": time_str}, ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            st.success(f"Attendance marked for ID: {user_id}")

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# --- Menu Actions ---
if choice == "Register User":
    st.subheader("Register New User")
    user_name = st.text_input("Name")
    user_id = st.text_input("ID")

    if st.button("Capture Face"):
        if user_name and user_id:
            capture_faces(user_id, user_name)
            train_model()
        else:
            st.warning("Please enter Name and ID")

elif choice == "Mark Attendance":
    st.subheader("Mark Attendance")
    if st.button("Start Camera"):
        mark_attendance()

elif choice == "View Attendance":
    st.subheader("Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.info("No attendance records found yet")
