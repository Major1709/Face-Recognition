import cv2
import pickle
import face_recognition
import app.Settings as Settings

KNOWN_DIR = Settings.Dir_Model

with open(KNOWN_DIR+"/model/encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), enc in zip(locations, encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
