# 🦺 PPE Detection System

An Enterprise-Level AI-Powered PPE (Personal Protective Equipment) Detection & Compliance Monitoring System built using **Flask, YOLO26s, ByteTrack, OSNet ReID, OpenCV, SQLite, and Real-Time Video Analytics**.

This project detects whether workers are wearing required safety equipment such as:

- 🪖 Helmet
- 🦺 Safety Vest
- 😷 Mask
- 👢 Safety Shoes *(future support)*
- 🧤 Gloves *(future support)*
- 🥽 Goggles *(future support)*

The system supports:

- 📷 Image Detection
- 🎥 Video Detection
- 📹 Live Webcam Monitoring
- 📡 RTSP / CCTV Stream Monitoring
- 📊 Dashboard Analytics
- 🚨 Real-Time Violation Alerts
- 🧠 Person-PPE Association Layer
- 🛰️ Multi-Person Tracking using ByteTrack
- 🧬 Person Re-Identification (ReID) using OSNet
- 🔁 Persistent Worker Identity Recovery
- 🗂️ Event History & Screenshot Logging

---

# 🚀 Features

## ✅ Real-Time PPE Detection
This project uses a custom-trained YOLOv26s model (`best.pt`) for real-time PPE detection.

The model detects:

- 👷 Person
- 🪖 Helmet
- 🦺 Safety Vest
- 😷 Mask
- ⛑️ Additional PPE Classes
- ❌ Missing PPE Violations

### Base Model

```bash
yolo26s.pt
```

After training:

```bash
best.pt
```

---

# 🧬 Enterprise ReID System (NEW)

## ✅ OSNet Person Re-Identification
The project now includes:

- OSNet-based feature extraction
- Persistent worker identity tracking
- Temporal embedding memory
- Cross-track identity recovery
- Stable worker global IDs

This upgrades the system from:

```text
Basic object tracking
```

into:

```text
Enterprise intelligent surveillance
```

---

## 🔥 Why ReID Is Important

Normal trackers like ByteTrack only maintain:

```text
Temporary Track IDs
```

Example:

```text
Track 1 → Track 17 → Track 43
```

when:

- worker turns
- occlusion happens
- person exits frame
- lighting changes
- motion blur occurs

This causes:

❌ Duplicate alerts  
❌ Broken identity tracking  
❌ Event fragmentation  
❌ Unstable monitoring

---

## ✅ Our Solution

The system now combines:

```text
ByteTrack + OSNet ReID
```

to create:

```text
Persistent Global Worker Identity
```

Example:

```text
Track 1
Track 17
Track 1000001
```

all become:

```text
gid_d074b67dab3d
```

This means:

✅ Same worker identity recovered  
✅ Stable event history  
✅ Enterprise-grade monitoring  
✅ Intelligent incident lifecycle

---

# 🧠 How ReID Works

## ReID Pipeline

```text
Camera Frame
      ↓
YOLO Detection
      ↓
ByteTrack Tracking
      ↓
Person Crop Extraction
      ↓
OSNet Feature Extractor
      ↓
Embedding Vector
      ↓
Cosine Similarity Matching
      ↓
Persistent Global Identity
```

---

## 🔹 OSNet Feature Extractor

OSNet converts a person image into:

```text
Mathematical Appearance Embedding
```

It learns:

- clothing texture
- body shape
- appearance patterns
- visual identity
- color distribution

Example embedding:

```python
[0.172, -0.552, 0.913, ...]
```

---

## 🔹 Cosine Similarity Matching

Embeddings are compared using:

```text
Cosine Similarity
```

High similarity:

```text
Same worker
```

Low similarity:

```text
Different worker
```

---

## 🔹 Temporal Embedding Memory

The project includes:

```python
embedding_history = deque(maxlen=50)
```

This stabilizes identities across frames.

Benefits:

✅ Reduces identity switching  
✅ Handles webcam noise  
✅ Handles motion blur  
✅ Handles temporary occlusion

---

## 🔹 Unknown Counter Protection

The system prevents false identity creation.

Instead of:

```text
1 bad frame → new identity
```

it uses:

```text
multi-frame validation
```

This dramatically improves:

- identity consistency
- event reliability
- enterprise stability

---

# ✅ Association Layer (Human + PPE Logic)

The system intelligently associates PPE items with detected persons.

Instead of only detecting objects independently, the project:

1. Detects persons
2. Detects PPE objects
3. Matches PPE objects to nearby persons
4. Determines compliance status

### Example:

| Person | Helmet | Vest | Result |
|---|---|---|---|
| Worker A | ❌ | ✅ | Violation |
| Worker B | ✅ | ✅ | Safe |

---

# ✅ ByteTrack Multi-Object Tracking

Each person receives a tracking ID.

### Benefits:

- Prevents duplicate alerts
- Tracks workers across frames
- Maintains temporal continuity
- Supports ReID recovery
- Improves event management

---

# ✅ Event Lifecycle Engine

The project includes enterprise event management.

### Event States

```text
NEW
 ↓
ACTIVE
 ↓
RESOLVED
 ↓
EXPIRED
```

This prevents:

❌ Alert flooding  
❌ Duplicate screenshots  
❌ Continuous repeated alerts

---

# ✅ Dashboard Analytics

Interactive dashboard showing:

- Total detections
- PPE compliance count
- Violations count
- Active streams
- Recent alerts
- Event history
- Live alert sidebar
- Stable worker identities
- Compliance trends

---

# ✅ Event Screenshot Storage

Whenever a PPE violation occurs:

- Screenshot captured automatically
- Event stored in SQLite database
- Timestamp recorded
- Compliance information saved
- Worker identity linked
- Evidence image generated

---

# ✅ RTSP / CCTV Support

Supports:

- CCTV Cameras
- IP Cameras
- Mobile IP Camera Apps
- RTSP Streams
- Webcam Devices

Example:

```bash
rtsp://username:password@ip:554/stream
```

---

# 🏗️ Enterprise System Architecture

```text
Video/Input Stream
        ↓
YOLO Detection Engine
        ↓
ByteTrack Tracking
        ↓
OSNet ReID Engine
        ↓
Global Identity Recovery
        ↓
Association Layer
(Person ↔ PPE Mapping)
        ↓
Compliance Rule Engine
        ↓
Violation Detection
        ↓
Event Lifecycle Manager
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
├── ppe.db
│
├── static/
│   ├── uploads/
│   ├── violations/
│   ├── css/
│   ├── js/
│   └── outputs/
│
├── templates/
│   ├── dashboard.html
│   ├── history.html
│   ├── image.html
│   ├── video.html
│   ├── webcam.html
│   ├── cctv.html
│   └── camera_manager.html
│
├── utils/
│   ├── detector.py
│   ├── tracker_manager.py
│   ├── association_engine.py
│   ├── compliance_engine.py
│   ├── event_manager.py
│   ├── feature_extractor.py
│   ├── reid_matcher.py
│   ├── stream_manager.py
│   ├── analytics_engine.py
│   ├── socket_events.py
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
- Flask-SocketIO
- SQLite
- OpenCV

## AI / Computer Vision

- YOLO26s
- ByteTrack
- OSNet
- TorchReID
- Multi-Object Tracking
- Person Re-Identification
- Association Logic
- Temporal Embedding Memory

## Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Chart.js

---

# 🧠 Core Detection Flow

## 1️⃣ Frame Capture

Frames captured from:

- Webcam
- RTSP stream
- Uploaded video
- Uploaded image

---

## 2️⃣ YOLO Detection

YOLO detects:

```python
{
    "class": "person",
    "confidence": 0.95,
    "bbox": [x1, y1, x2, y2]
}
```

---

## 3️⃣ ByteTrack Tracking

Workers are tracked frame-by-frame.

Example:

```python
track_id = 17
```

---

## 4️⃣ OSNet ReID

Person crops are extracted.

OSNet generates embeddings.

Embeddings are matched using cosine similarity.

Global identity example:

```text
gid_a8d1e0ee5324
```

---

## 5️⃣ Association Engine

PPE items are mapped to workers.

Example:

```python
if helmet_inside_person_box:
    assign_helmet()
```

---

## 6️⃣ Compliance Engine

Rules evaluated.

Example:

```python
if not helmet:
    violation = True
```

---

## 7️⃣ Event Lifecycle Engine

Creates intelligent incidents.

Stores:

- ACTIVE events
- RESOLVED events
- evidence images
- timestamps
- worker identities

---

# 📊 Dashboard Features

## Dashboard Includes

- Live counters
- PPE summaries
- Violation trends
- Event timeline
- Camera feed preview
- Detection statistics
- Live alert sidebar
- Screenshot evidence
- Worker ReID identities

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

# 📈 Performance Optimizations

- Frame skipping
- Confidence thresholding
- Efficient ByteTrack tracking
- Temporal embedding smoothing
- Identity memory caching
- Stream buffering
- Lightweight OSNet model

---

# 👨‍💻 Author

Developed by:

**Saurabh Kumar Mohanka**