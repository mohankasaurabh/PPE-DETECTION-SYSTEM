# 🦺 PPE Detection System

An Enterprise-Level AI-Powered PPE (Personal Protective Equipment) Detection & Compliance Monitoring System built using **Flask, YOLO26s, ByteTrack, OpenCV, SQLite, and Real-Time Video Analytics**.

This project detects whether workers are wearing required safety equipment such as:

- 🪖 Helmet
- 🦺 Safety Vest


The system supports:

- 📷 Image Detection
- 🎥 Video Detection
- 📹 Live Webcam Monitoring
- 📡 RTSP / CCTV Stream Monitoring
- 📊 Dashboard Analytics
- 🚨 Real-Time Violation Alerts
- 🧠 Person-PPE Association Layer
- 🛰️ Multi-Person Tracking using ByteTrack
- 🗂️ Event History & Screenshot Logging

---

# 🚀 Features

## ✅ Real-Time PPE Detection
This project uses a custom-trained YOLOv26s model (best.pt) for real-time PPE (Personal Protective Equipment) detection.

The model was trained on a custom PPE dataset for approximately 1–1.5 hours using the Ultralytics YOLO26s framework. The training process included multiple PPE classes such as:

👷 Person
🪖 Helmet
🦺 Safety Vest
😷 Mask
⛑️ Other PPE Equipment



### Base Model

The project was initially trained using:

```bash
yolo26s.pt
``` 
After training, the best-performing weights were saved as:

```bash
best.pt
```

---

## ✅ Association Layer (Human + PPE Logic)
The system intelligently associates PPE items with detected persons.

Instead of only detecting objects independently, the project:

1. Detects persons
2. Detects PPE objects
3. Matches PPE objects to nearby persons
4. Determines compliance status

This creates a real-world industrial PPE compliance engine.

### Example:

| Person | Helmet | Vest | Result |
|---|---|---|---|
| Person #12 | ❌ | ✅ | Violation |
| Person #18 | ✅ | ✅ | Safe |

---

## ✅ ByteTrack Object Tracking
Each person gets a unique tracking ID.

### Benefits:

- Prevents duplicate alerts
- Tracks people across frames
- Maintains person identity
- Enables screenshot-based history
- Helps with future ReID integration

---

## ✅ Dashboard Analytics
Interactive dashboard showing:

- Total detections
- PPE compliance count
- Violations count
- Active streams
- Recent alerts
- Event history
- Compliance trends

---

## ✅ Event Screenshot Storage
Whenever a PPE violation occurs:

- Screenshot is captured automatically
- Event is stored in SQLite database
- Timestamp is recorded
- Compliance information is saved

---

## ✅ RTSP / CCTV Support
Supports:

- CCTV Cameras
- IP Cameras
- Mobile IP Camera Apps
- RTSP Streams

Example:

```bash
rtsp://username:password@ip:port/path
```

---

## ✅ Multiple Input Modes
### Supported Inputs:

| Mode | Supported |
|---|---|
| Image Upload | ✅ |
| Video Upload | ✅ |
| Webcam | ✅ |
| RTSP Stream | ✅ |
| CCTV Feed | ✅ |

---

# 🏗️ System Architecture

```text
Video/Input Stream
        ↓
YOLO Detection Engine
        ↓
ByteTrack Tracking
        ↓
Association Layer
(Person ↔ PPE Mapping)
        ↓
Compliance Engine
        ↓
Violation Detection
        ↓
Alert + Screenshot + Database Logging
        ↓
Dashboard Analytics
```

---

# 📂 Project Structure

```bash
PPE-DETECTION-SYSTEM/
│
├── app.py
├── best.pt
├── requirements.txt
├── database.db
│
├── static/
│   ├── uploads/
│   ├── violations/
│   ├── css/
│   ├── js/
│   └── screenshots/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── history.html
│   ├── image.html
│   ├── video.html
│   └── webcam.html
│
├── utils/
│   ├── detector.py
│   ├── tracker.py
│   ├── association.py
│   ├── compliance.py
│   ├── alert_manager.py
│   ├── stream_manager.py
│   └── database.py
│
└── models/
    └── best.pt
```

---

# ⚙️ Technologies Used

## Backend
- Python
- Flask
- OpenCV
- SQLite
- SocketIO

## AI / Computer Vision
- YOLO
- ByteTrack
- Object Detection
- Multi-Object Tracking
- Association Logic

## Frontend
- HTML
- CSS
- JavaScript
- Bootstrap
- Chart.js

---

# 🧠 Core Logic Explanation

# ======================================
# MAIN DETECTION FLOW
# ======================================

## 1️⃣ Frame Capture
Frames are captured from:

- Webcam
- RTSP stream
- Uploaded video
- Uploaded image

---

## 2️⃣ YOLO Detection
YOLO processes each frame and returns:

```python
[
    {
        "class": "person",
        "confidence": 0.95,
        "bbox": [x1, y1, x2, y2]
    },
    {
        "class": "helmet",
        "confidence": 0.91,
        "bbox": [x1, y1, x2, y2]
    }
]
```

---

## 3️⃣ ByteTrack Tracking
People are tracked frame-by-frame.

Each detected person receives:

```python
person_id = 17
```

Tracking helps maintain identity consistency.

---

## 4️⃣ Association Layer
This is one of the most important components.

The system associates PPE items with persons using:

- Bounding box overlap
- Distance calculation
- Center point matching
- Region constraints

### Example:

```python
if helmet_center inside person_box:
    assign helmet to person
```

---

## 5️⃣ Compliance Engine
Rules are evaluated.

### Example:

```python
if not helmet:
    violation = True
```

Rules can be customized.

---

## 6️⃣ Alert Manager
When a violation occurs:

- Screenshot is captured
- Event is logged
- Dashboard updates
- Browser notification can trigger

---

## 7️⃣ Database Logging
Violations are stored in SQLite.

### Stored Data:

- Event ID
- Timestamp
- PPE status
- Screenshot path
- Camera source
- Person ID

---

# 📊 Dashboard Features

## Dashboard Includes:

- Live counters
- PPE summaries
- Violation trends
- Event timeline
- Camera feed preview
- Detection statistics

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/PPE-DETECTION-SYSTEM.git
cd PPE-DETECTION-SYSTEM
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Application

```bash
python app.py
```

---

## 5️⃣ Open Browser

```bash
http://127.0.0.1:5000
```

---

# 📡 RTSP Example

Example using Mobile IP Camera:

```bash
rtsp://192.168.1.5:8080/h264_ulaw.sdp
```

Or:

```bash
http://192.168.1.5:8080/video
```

---

# 📷 Screenshots

## Home Page
- Upload image/video
- Start webcam
- Connect CCTV

## Dashboard
- Analytics
- Live detection counters
- Recent violations

## History Page
- Violation screenshots
- Event filtering
- Delete selected events

---

# 🛡️ Safety Compliance Logic

The project follows industrial PPE compliance logic.

### Example Rules:

| Rule | Result |
|---|---|
| No Helmet | Violation |
| No Vest | Violation |
| Helmet + Vest Present | Safe |

Rules can be modified based on industry requirements.

---

# 📈 Performance Optimizations

- Frame skipping
- Confidence thresholding
- Lightweight YOLO inference
- Efficient object tracking
- Stream buffering

---

# 👨‍💻 Author

Developed by:

**Saurabh Kumar Mohanka**




