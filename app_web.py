import os
import pickle
import numpy as np
import cv2
import streamlit as st
import face_recognition
import Settings

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Face Recognition Webcam", layout="wide")
st.title("🎥 Real-time Face Recognition (Webcam)")

ENC_PATH = Settings.Dir_Model+"/model/encodings.pkl"

@st.cache_resource
def load_gallery(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        known_encodings, known_names = pickle.load(f)
    return np.array(known_encodings), list(known_names)

# Sidebar controls
st.sidebar.header("Settings")
detector_model = st.sidebar.selectbox("Face detector", ["hog", "cnn"], index=0)
tolerance = st.sidebar.slider("Tolerance (lower = stricter)", 0.35, 0.70, 0.50, 0.01)
downscale = st.sidebar.slider("Downscale factor (speed)", 1, 4, 2, 1)

st.sidebar.caption(
    "Tip: 'cnn' is more accurate and uses GPU only if dlib was built with CUDA. "
    "'hog' is CPU-only but faster."
)

try:
    KNOWN_ENCODINGS, KNOWN_NAMES = load_gallery(ENC_PATH)
    st.sidebar.success(f"Loaded gallery: {len(KNOWN_NAMES)} encodings")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

def best_match_name(face_enc: np.ndarray, tolerance: float) -> str:
    # Best-match via minimum distance (better than first True)
    distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_enc)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist < tolerance:
        return KNOWN_NAMES[best_idx]
    return "Unknown"

def draw_boxes(frame_bgr, locations, names):
    for (top, right, bottom, left), name in zip(locations, names):
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame_bgr, (left, top - 28), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_bgr, name, (left + 6, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame_bgr

class FaceRecTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Downscale for speed
        if downscale > 1:
            small = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
        else:
            small = img

        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        locations = face_recognition.face_locations(rgb_small, model=detector_model)

        # Compute encodings + match
        names = []
        if locations:
            encs = face_recognition.face_encodings(rgb_small, locations)
            for enc in encs:
                names.append(best_match_name(enc, tolerance))

        # Scale boxes back to original size
        if downscale > 1 and locations:
            scaled_locations = []
            for (top, right, bottom, left) in locations:
                scaled_locations.append((
                    top * downscale, right * downscale,
                    bottom * downscale, left * downscale
                ))
            locations_to_draw = scaled_locations
        else:
            locations_to_draw = locations

        # Draw
        out = draw_boxes(img, locations_to_draw, names)

        return out

st.markdown(
    """
**How it works:**  
Webcam frames → face detection → face embeddings → best-match by distance → bounding boxes + labels.
"""
)

webrtc_streamer(
    key="face-recognition",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceRecTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

st.info(
    "Performance tips: Increase downscale (2→3 or 4) for speed. "
    "Use 'hog' for fastest CPU mode. Use 'cnn' for better accuracy (GPU only if dlib CUDA=True)."
)
