# 🚗 Automatic Number Plate Recognition (ANPR) using YOLOv8 & EasyOCR

A real-time **Automatic Number Plate Recognition (ANPR)** system developed using **YOLOv8**, **OpenCV**, and **EasyOCR** for intelligent traffic surveillance. The system detects vehicles, localizes license plates, and extracts license plate text from live webcam feeds and recorded traffic videos.

---

## 📌 Project Overview

Automatic Number Plate Recognition (ANPR) is a Computer Vision application used in smart traffic management, toll collection, parking automation, and security systems.

This project implements an end-to-end ANPR pipeline capable of:

- Detecting vehicles in real-time
- Classifying vehicle types
- Detecting license plates using a custom-trained YOLOv8 model
- Extracting license plate regions
- Performing OCR-based text recognition
- Supporting both webcam and recorded video inputs

---

## 🚀 Features

- Real-time Vehicle Detection
- Vehicle Classification
- Custom License Plate Detection
- Webcam & Video File Support
- Image Preprocessing using OpenCV
- OCR Integration using EasyOCR
- Automatic Plate Region Cropping
- Output Video Generation
- Real-time Bounding Box Visualization

---

## 🏗️ Project Pipeline

```
Input Video / Webcam
        │
        ▼
Video Capture (OpenCV)
        │
        ▼
Vehicle Detection (YOLOv8)
        │
        ▼
Vehicle Classification
        │
        ▼
License Plate Detection (Custom YOLOv8)
        │
        ▼
Crop License Plate
        │
        ▼
Image Preprocessing
        │
        ▼
EasyOCR
        │
        ▼
Recognized License Plate Number
        │
        ▼
Display & Save Output
```

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- EasyOCR
- NumPy
- PyTorch

---

## 📂 Project Structure

```
Automatic-Number-Plate-Recognition/
│
├── models/
│   ├── best.pt
│   └── yolov8n.pt
│
├── input/
│   ├── traffic.mp4
│   ├── vehicles.mp4
│   └── sample.jpg
│
├── output/
│   ├── output_video.avi
│   ├── webcam_output.avi
│   └── detected_frames/
│
├── src/
│   ├── anpr.py
│   ├── vehicle_detection.py
│   ├── plate_detection.py
│   ├── preprocess.py
│   ├── ocr.py
│   └── utils.py
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Automatic-Number-Plate-Recognition.git
```

Move to project directory

```bash
cd Automatic-Number-Plate-Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Using Webcam

```python
cap = cv2.VideoCapture(0)
```

Run

```bash
python anpr.py
```

---

### Using Video

Replace

```python
cap = cv2.VideoCapture(0)
```

with

```python
cap = cv2.VideoCapture("traffic.mp4")
```

or

```python
cap = cv2.VideoCapture("vehicles.mp4")
```

Run

```bash
python anpr.py
```

---

## 📊 Detection Pipeline

### Step 1
Vehicle Detection using YOLOv8

↓

### Step 2
Vehicle Classification

↓

### Step 3
License Plate Detection using Custom YOLO Model

↓

### Step 4
License Plate Cropping

↓

### Step 5
Image Preprocessing

↓

### Step 6
OCR using EasyOCR

↓

### Step 7
Recognized License Plate Display

---

## 📸 Results

### Vehicle Detection

- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles
- Trains

### License Plate Detection

Custom-trained YOLOv8 model detects license plates from vehicles in real-time.

### OCR

EasyOCR extracts alphanumeric license plate text from detected plate regions after image preprocessing.

---

## 📈 Current Status

✔ Vehicle Detection

✔ Vehicle Classification

✔ License Plate Detection

✔ Video Input Support

✔ Webcam Support

✔ OCR Integration

🔄 OCR accuracy is currently being optimized through improved preprocessing and dataset enhancement.

---

## 🔬 Future Improvements

- PaddleOCR Integration
- DeepSORT Vehicle Tracking
- Database Integration (MySQL/SQLite)
- FastAPI Deployment
- Streamlit Dashboard
- Multi-camera Support
- Speed Estimation
- Vehicle Counting
- Cloud Deployment (AWS/GCP)

---

## 💻 Requirements

```
Python >=3.10

opencv-python

ultralytics

easyocr

numpy

torch

torchvision
```

or simply

```bash
pip install -r requirements.txt
```

---

## 📚 Applications

- Smart Parking Systems
- Toll Plaza Automation
- Traffic Surveillance
- Vehicle Access Control
- Law Enforcement
- Smart City Solutions
- Security Monitoring

---

## 👩‍💻 Author

**Mansi Gaul**

B.Tech | AI/ML Enthusiast | Computer Vision | Deep Learning
