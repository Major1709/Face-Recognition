import os
import pickle
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import face_recognition
import Settings as Settings

st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("👤 Face Recognition (Image Upload)")
st.write("Upload an image, the app will detect faces and try to recognize them.")

ENC_PATH = Settings.Dir_Model+"/model/encodings.pkl"

@st.cache_resource
def load_gallery(enc_path: str):
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encodings file: {enc_path}")
    with open(enc_path, "rb") as f:
        known_encodings, known_names = pickle.load(f)
    known_encodings = np.array(known_encodings)
    return known_encodings, known_names

def draw_boxes_bgr(bgr_img, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(bgr_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(bgr_img, (left, top - 28), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(bgr_img, name, (left + 6, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return bgr_img

# Sidebar controls
st.sidebar.header("Settings")
detector_model = st.sidebar.selectbox("Face detector", ["hog", "cnn"], index=0)
tolerance = st.sidebar.slider("Tolerance (lower = stricter)", 0.35, 0.70, 0.50, 0.01)

st.sidebar.caption("Tip: If you have GPU-enabled dlib, choose 'cnn' for better accuracy.")

# Load known faces
try:
    known_encodings, known_names = load_gallery(ENC_PATH)
    st.sidebar.success(f"Loaded gallery: {len(known_names)} encodings")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil_img)

    with st.spinner("Detecting faces..."):
        face_locations = face_recognition.face_locations(rgb, model=detector_model)

    if len(face_locations) == 0:
        st.warning("No faces detected.")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
        st.stop()

    with st.spinner("Computing embeddings & matching..."):
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []
        for enc in face_encodings:
            # Best-match by minimum distance (better than taking first True)
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < tolerance:
                face_names.append(known_names[best_idx])
            else:
                face_names.append("Unknown")

    # Draw results
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_annot = draw_boxes_bgr(bgr.copy(), face_locations, face_names)
    rgb_annot = cv2.cvtColor(bgr_annot, cv2.COLOR_BGR2RGB)

    st.subheader("Result")
    st.image(rgb_annot, use_container_width=True)

    st.subheader("Detections")
    for i, (loc, name) in enumerate(zip(face_locations, face_names), start=1):
        top, right, bottom, left = loc
        st.write(f"**Face {i}:** {name}  |  box=({left},{top})-({right},{bottom})")
