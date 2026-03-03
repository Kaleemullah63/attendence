import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.title("Face Recognition Attendance System")

# --- Load Haar Cascade for face detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Attendance storage ---
if "attendance" not in st.session_state:
    st.session_state.attendance = pd.DataFrame(columns=["Name", "Time"])

# Upload image
uploaded_file = st.file_uploader("Upload image with face(s)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    st.write(f"Detected {len(faces)} face(s)")

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(img_array, caption="Processed Image", use_column_width=True)

    # --- Mark attendance ---
    if len(faces) > 0:
        name = st.text_input("Enter your name to mark attendance")
        if st.button("Mark Attendance") and name:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.attendance = st.session_state.attendance.append(
                {"Name": name, "Time": now}, ignore_index=True
            )
            st.success(f"Attendance marked for {name} at {now}")

# Show attendance
st.subheader("Attendance Sheet")
st.dataframe(st.session_state.attendance)
