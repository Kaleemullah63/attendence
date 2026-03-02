import streamlit as st
import face_recognition
import cv2
import numpy as np
import sqlite3
import os
import easyocr
from datetime import datetime
from pyzbar.pyzbar import decode
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Smart Attendance", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
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
    names = []
    for file in os.listdir("known_faces"):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join("known_faces", file)
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

# ---------------- LIVENESS ----------------
def check_liveness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 80

# ---------------- REGISTER STUDENT ----------------
def register_student(name, image):
    file_path = f"known_faces/{name}.jpg"
    cv2.imwrite(file_path, image)
    try:
        cursor.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
    except:
        pass

# ---------------- MARK ATTENDANCE ----------------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if cursor.fetchone():
        return "Already marked today"

    cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    return f"Attendance marked for {name}"

# ---------------- PROCESS IMAGE ----------------
reader = easyocr.Reader(['en'])

def process_image(uploaded_file):
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if not check_liveness(image_np):
        return "Fake / Blurry image detected", ""

    known_encodings, known_names = load_known_faces()

    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    detected_name = "Unknown"

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        distances = face_recognition.face_distance(known_encodings, encoding)

        if len(distances) > 0:
            best_match = np.argmin(distances)
            if matches[best_match]:
                detected_name = known_names[best_match]

    # OCR
    ocr_results = reader.readtext(image_np)
    extracted_text = " ".join([res[1] for res in ocr_results])

    # QR
    qr_text = ""
    decoded = decode(image_np)
    for obj in decoded:
        qr_text = obj.data.decode("utf-8")

    if detected_name != "Unknown" and (
        detected_name.lower() in extracted_text.lower()
        or detected_name.lower() in qr_text.lower()
    ):
        status = mark_attendance(detected_name)
    else:
        status = "Face and ID/QR do not match"

    return status, extracted_text

# ---------------- UI ----------------
st.title("🎓 AI Smart Attendance System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Mark Attendance", "Register Student", "Dashboard"]
)

# -------- MARK ATTENDANCE --------
if menu == "Mark Attendance":

    st.subheader("Upload ID Card with Face")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, width=300)

        if st.button("Mark Attendance"):
            status, text = process_image(uploaded_file)
            st.success(status)
            st.info(f"Extracted Text: {text}")

# -------- REGISTER STUDENT --------
elif menu == "Register Student":

    st.subheader("Register New Student")

    name = st.text_input("Student Name")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, width=200)

        if st.button("Register"):
            register_student(name, image_np)
            st.success(f"{name} registered successfully!")

# -------- DASHBOARD --------
elif menu == "Dashboard":

    st.subheader("Attendance Dashboard")

    df = pd.read_sql_query("SELECT * FROM attendance", conn)

    if df.empty:
        st.warning("No attendance data yet.")
    else:
        st.dataframe(df)

        counts = df["name"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Attendance Count Per Student")
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "attendance_export.csv",
            "text/csv"
        )
