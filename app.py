import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# Paths
DATA_DIR = "registered_users"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["ID", "Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Sidebar
st.sidebar.title("Attendance System")
choice = st.sidebar.radio("Menu", ["Register User", "Mark Attendance", "Dashboard"])

# ---------------- Register User ----------------
if choice == "Register User":
    st.header("Register New User")
    name = st.text_input("Enter Name")
    user_id = st.text_input("Enter ID")
    photo = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if name and user_id and photo:
            img = Image.open(photo)
            img_path = os.path.join(DATA_DIR, f"{user_id}.png")
            img.save(img_path)
            st.success(f"User {name} registered successfully!")
        else:
            st.error("Please provide all details")

# ---------------- Mark Attendance ----------------
elif choice == "Mark Attendance":
    st.header("Mark Attendance")
    run = st.button("Open Camera")
    if run:
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        registered = {}
        for file in os.listdir(DATA_DIR):
            img = cv2.imread(os.path.join(DATA_DIR, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            registered[file.split(".")[0]] = gray

        stframe = st.empty()
        recognized = False

        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to access camera")
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray_frame[y:y+h, x:x+w]
                for uid, reg_face in registered.items():
                    resized_face = cv2.resize(reg_face, (w, h))
                    diff = cv2.absdiff(resized_face, face_img)
                    if np.mean(diff) < 50:  # simple threshold
                        st.success(f"Recognized: {uid}")
                        # Mark attendance
                        df = pd.read_csv(ATTENDANCE_FILE)
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if uid not in df["ID"].values:
                            df.loc[len(df)] = [uid, uid, now]
                            df.to_csv(ATTENDANCE_FILE, index=False)
                        recognized = True
                        break

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if recognized:
                break

        cam.release()
        cv2.destroyAllWindows()

# ---------------- Dashboard ----------------
elif choice == "Dashboard":
    st.header("Attendance Dashboard")
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df)
