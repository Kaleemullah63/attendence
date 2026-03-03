import streamlit as st
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from PIL import Image
import os
import pickle

st.set_page_config(page_title="Face Attendance App", layout="wide")

# ------------------------
# Helper functions
# ------------------------
detector = MTCNN()
EMBEDDING_MODEL_PATH = "facenet_keras.h5"  # pretrained FaceNet Keras model
ENCODINGS_PATH = "encodings.pkl"           # stored user embeddings

if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings = pickle.load(f)
else:
    known_encodings = {}

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# ------------------------
# Load FaceNet model
# ------------------------
@st.cache_resource
def load_facenet_model():
    return load_model(EMBEDDING_MODEL_PATH)

facenet_model = load_facenet_model()

def get_embedding(face_pixels):
    face_pixels = preprocess_face(face_pixels)
    embedding = facenet_model.predict(face_pixels)
    return embedding[0]

def recognize_face(face_pixels):
    embedding = get_embedding(face_pixels)
    min_dist = 100
    identity = "Unknown"
    for name, known_emb in known_encodings.items():
        dist = np.linalg.norm(embedding - known_emb)
        if dist < 0.7 and dist < min_dist:
            min_dist = dist
            identity = name
    return identity

# ------------------------
# UI
# ------------------------
st.title("📸 Face Recognition Attendance")

menu = ["Register User", "Mark Attendance", "View Dashboard"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register User":
    st.subheader("Register New User")
    user_name = st.text_input("Enter Name")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    
    if st.button("Register") and user_name and uploaded_file:
        img = np.array(Image.open(uploaded_file))
        faces = detector.detect_faces(img)
        if faces:
            x, y, w, h = faces[0]['box']
            face_img = img[y:y+h, x:x+w]
            emb = get_embedding(face_img)
            known_encodings[user_name] = emb
            with open(ENCODINGS_PATH, "wb") as f:
                pickle.dump(known_encodings, f)
            st.success(f"User {user_name} registered successfully!")
        else:
            st.error("No face detected in the image!")

elif choice == "Mark Attendance":
    st.subheader("Mark Attendance via Webcam")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    if st.button("Start Camera"):
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access camera")
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            for face in faces:
                x, y, w, h = face['box']
                face_img = img_rgb[y:y+h, x:x+w]
                name = recognize_face(face_img)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
elif choice == "View Dashboard":
    st.subheader("Registered Users & Attendance")
    if known_encodings:
        df = pd.DataFrame(list(known_encodings.keys()), columns=["Name"])
        st.dataframe(df)
    else:
        st.info("No users registered yet.")
