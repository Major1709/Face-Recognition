import pickle
import face_recognition
import Settings

KNOWN_DIR = Settings.Dir_Model

with open(KNOWN_DIR+"/model/encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

test_image = face_recognition.load_image_file("Here picture")
face_locations = face_recognition.face_locations(test_image, model="hog")
face_encodings = face_recognition.face_encodings(test_image, face_locations)

for enc in face_encodings:
    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
    name = "Unknown"

    if True in matches:
        idx = matches.index(True)
        name = known_names[idx]

    print("Prediction:", name)
