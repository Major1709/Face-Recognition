import os
import face_recognition
import pickle
import Settings

KNOWN_DIR = Settings.Dir
MODEL_DIR = Settings.Dir_Model
known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_DIR):
    person_path = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image, model="hog")

        if len(locations) == 0:
            continue

        encoding = face_recognition.face_encodings(image, locations)[0]
        known_encodings.append(encoding)
        known_names.append(person_name)

# Save encodings
os.makedirs(MODEL_DIR+"model", exist_ok=True)
with open(MODEL_DIR+"model/encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Saved encodings:", len(known_encodings))
