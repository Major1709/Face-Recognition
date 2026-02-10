import pickle

import numpy as np
import face_recognition
from app import Settings


def predict_face(image_path, tolerance: float = 0.5) -> str:
    known_dir = Settings.Dir_Model

    with open(known_dir + "/model/encodings.pkl", "rb") as f:
        known_encodings, known_names = pickle.load(f)

    test_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(test_image, model="hog")
    if not face_locations:
        return "Unknown"

    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    if not face_encodings:
        return "Unknown"

    for enc in face_encodings:
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist < tolerance:
            return known_names[best_idx]

    return "Unknown"


if __name__ == "__main__":
    pass
