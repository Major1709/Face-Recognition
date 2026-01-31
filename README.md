# Face Recognition System 🧠🎥

A complete **Face Recognition system in Python**, supporting **training**, **image-based inference**, and **real-time webcam recognition**, including **web applications built with Streamlit**.

This project is designed for **educational, experimental, and prototyping purposes**, and demonstrates a full face recognition pipeline using modern computer vision techniques.

---

## 📌 Overview

The system uses **deep face embeddings** to identify individuals by comparing facial features extracted from images or video streams. Once trained, the model can recognize known faces in real time with configurable accuracy and performance settings.

---

## ✨ Key Features

* Face recognition using deep embeddings
* Training pipeline with labeled images
* Persistent model storage (pickle)
* Face recognition from:

  * Static images
  * Live webcam feed (OpenCV)
  * Real-time web interface (Streamlit + WebRTC)
* Configurable detection models (`hog`, `cnn`)
* Adjustable recognition tolerance
* Modular and extensible architecture

---

## 🏗️ Architecture

```
Images → Face Detection → Face Embeddings → Distance Matching → Identity
```

* **Face Detection**: HOG or CNN-based detector
* **Feature Extraction**: 128-D face embeddings
* **Matching**: Euclidean distance with configurable threshold

---

## 📂 Project Structure

```
Face-Recognition/
│
├── img/                         # Training images (one folder per identity)
│   ├── Person_A/
│   ├── Person_B/
│
├── model/
│   └── encodings.pkl            # Serialized face embeddings
│
├── Train_face_recognition.py    # Model training script
├── Use_model.py                 # Image-based recognition
├── Camera_with_Model.py         # Real-time webcam recognition
├── app.py                       # Streamlit app (image upload)
├── app_web.py                   # Streamlit app (real-time webcam)
├── Settings.py                  # Global configuration
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:Major1709/Face-Recognition.git
cd Face-Recognition
```

### 2. Install dependencies

```bash
pip install face_recognition opencv-python streamlit streamlit-webrtc numpy pillow
```

> ⚠️ **Important**
> `face_recognition` depends on **dlib**.
>
> * Linux/macOS: usually installs automatically
> * Windows: use a precompiled dlib wheel

---

## 🔧 Configuration

Edit **Settings.py** to configure project paths:

```python
Dir = "/absolute/path/to/img/"
Dir_Model = "/absolute/path/to/project/"
```

---

## 🧠 Training the Model

### Dataset format

Each person must have their own folder:

```
img/
├── Alice/
│   ├── img1.jpg
│   ├── img2.jpg
├── Bob/
│   ├── img1.jpg
```

### Run training

```bash
python Train_face_recognition.py
```

This will generate:

```
model/encodings.pkl
```

---

## 🖼️ Face Recognition (Image)

```bash
python Use_model.py
```

The script:

* Loads trained encodings
* Detects faces in the input image
* Outputs predicted identities

---

## 📷 Face Recognition (Webcam – OpenCV)

```bash
python Camera_with_Model.py
```

* Real-time face recognition
* Press **Q** to exit

---

## 🌐 Web Applications (Streamlit)

### Image Upload App

```bash
streamlit run app.py
```

Features:

* Upload an image
* Face detection and recognition
* Adjustable tolerance and detector model

---

### Real-Time Webcam App

```bash
streamlit run app_web.py
```

Features:

* Real-time face recognition in browser
* WebRTC webcam streaming
* Performance controls (downscaling, tolerance, model)

---

## ⚖️ Detection Models

| Model | Speed  | Accuracy | Hardware   |
| ----- | ------ | -------- | ---------- |
| hog   | Fast   | Medium   | CPU        |
| cnn   | Slower | High     | GPU (CUDA) |

---

## 🎯 Best Practices

* Use high-quality, frontal face images
* Avoid multiple faces per training image
* Lower tolerance → stricter recognition
* Increase downscale for better real-time performance

---

## 🚧 Limitations

* No built-in liveness detection
* Sensitive to lighting conditions
* Accuracy depends on training data quality

---

## 🛠️ Future Improvements

* Face registration via webcam
* Database-backed identity management
* REST API deployment
* Liveness detection
* Model evaluation metrics

---

## 📄 License

This project is intended for **educational and experimental use**.
Not recommended for production or security-critical applications.

---

## 👤 Author

**Kevin**
GitHub: [https://github.com/Major1709](https://github.com/Major1709)
