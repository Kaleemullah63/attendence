import streamlit as st
import face_recognition
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="AI Smart Attendance", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    student_id TEXT UNIQUE,
    image_path TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    student_id TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

# ---------------- FOLDER ----------------
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")

# ---------------- LOAD FACES ----------------
def load_known_faces():
    encodings = []
    data = []

    for file in os.listdir("known_faces"):
        path = os.path.join("known_faces", file)
        image = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(image)

        if enc:
            encodings.append(enc[0])
            student_id = file.split("_")[0]
            cursor.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
            result = cursor.fetchone()
            if result:
                data.append((result[0], student_id))
    return encodings, data

# ---------------- REGISTER ----------------
def register_student(name, student_id, image):
    file_path = f"known_faces/{student_id}_{name}.jpg"
    cv2.imwrite(file_path, image)
    cursor.execute(
        "INSERT INTO students (name, student_id, image_path) VALUES (?, ?, ?)",
        (name, student_id, file_path)
    )
    conn.commit()

# ---------------- MARK ATTENDANCE ----------------
def mark_attendance(name, student_id):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute(
        "SELECT * FROM attendance WHERE student_id=? AND date=?",
        (student_id, date)
    )

    if cursor.fetchone():
        return "Already marked today"

    cursor.execute(
        "INSERT INTO attendance (name, student_id, date, time) VALUES (?, ?, ?, ?)",
        (name, student_id, date, time)
    )
    conn.commit()
    return "Attendance Marked Successfully"

# ---------------- UI ----------------
st.title("🎓 AI Smart Attendance System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Student", "Mark Attendance", "Dashboard"]
)

# ================= REGISTER =================
if menu == "Register Student":
    st.subheader("Register New Student")
    name = st.text_input("Student Name")
    student_id = st.text_input("Student ID")
    camera_image = st.camera_input("Take Student Photo")

    if camera_image is not None and st.button("Register"):
        image = np.array(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        register_student(name, student_id, image)
        st.success("Student Registered Successfully")

# ================= MARK ATTENDANCE =================
elif menu == "Mark Attendance":
    st.subheader("Camera Attendance")
    camera_image = st.camera_input("Scan Your Face")

    if camera_image is not None:
        image = np.array(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        known_encodings, known_data = load_known_faces()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            distances = face_recognition.face_distance(known_encodings, encoding)

            if len(distances) > 0:
                best_match = np.argmin(distances)
                if matches[best_match]:
                    name, student_id = known_data[best_match]
                    st.success(f"Recognized: {name}")
                    st.info(f"Student ID: {student_id}")
                    status = mark_attendance(name, student_id)
                    st.success(status)
                    break
        else:
            st.error("Face Not Recognized")

# ================= DASHBOARD =================
elif menu == "Dashboard":
    st.subheader("Attendance Dashboard")
    df = pd.read_sql_query("SELECT * FROM attendance", conn)

    if df.empty:
        st.warning("No attendance yet")
    else:
        st.dataframe(df)
