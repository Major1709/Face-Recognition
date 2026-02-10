import os
import pickle
import face_recognition
from app import Settings

ENC_PATH = Settings.Dir_Model + "/model/encodings.pkl"
IMG_ROOT = Settings.Dir #root img_path
PERSON_NAME = "Fanny" # name's person to add

person_dir = os.path.join(IMG_ROOT, PERSON_NAME)
if not os.path.isdir(person_dir):
    raise FileNotFoundError(f"Folder not found: {person_dir}")


if os.path.exists(ENC_PATH):
    with open(ENC_PATH, "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings, known_names = [], []

added = 0
for file in os.listdir(person_dir):
    img_path = os.path.join(person_dir, file)
    image = face_recognition.load_image_file(img_path)
    locations = face_recognition.face_locations(image, model="hog")
    if len(locations) == 0:
        continue

    enc = face_recognition.face_encodings(image, locations)[0]
    known_encodings.append(enc)
    known_names.append(PERSON_NAME)
    added += 1

os.makedirs(os.path.dirname(ENC_PATH), exist_ok=True)
with open(ENC_PATH, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(f"Added {added} face encodings for {PERSON_NAME}. Total = {len(known_names)}")
