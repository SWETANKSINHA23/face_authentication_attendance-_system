# Face Authentication Attendance System

🚀 **Live Demo:** [https://face-auth-attendance-8627.onrender.com/](https://face-auth-attendance-8627.onrender.com/)

A comprehensive face recognition-based attendance management system using OpenCV SFace, Google MediaPipe, and Streamlit. This lightweight stack is optimized for instant deployment and fast execution even on free-tier cloud platforms.

---

## Overview

The Face Authentication Attendance System is an automated attendance management solution that uses modern, lightweight facial recognition technology to identify and authenticate users. The system includes:

- **Face Registration**: Multi-sample capture with quality validation
- **Attendance Marking**: Automated punch-in/punch-out with face authentication
- **Anti-Spoofing**: Liveness detection to prevent photo/video attacks
- **Web Interface**: Streamlit-based UI for all operations
- **Optimized for Deployment**: Utilizes MediaPipe and OpenCV SFace to bypass heavy C++ compilation limits.

---

## Model and Approach

### Face Detection

**Method**: Google MediaPipe Face Detection

- **Architecture**: BlazeFace (optimized for mobile/edge use cases)
- **Library**: `mediapipe`
- **Performance**: High accuracy, incredibly fast, and lightweight on system memory.

**Detection Pipeline**:
```python
import mediapipe as mp

# Face detection using MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

results = face_detector.process(rgb_image)
```

### Face Recognition

**Model**: OpenCV SFace (FaceRecognizerSF)

- **Architecture**: Lightweight state-of-the-art Convolutional Neural Network
- **Embedding Dimension**: 128-dimensional float32 face encodings
- **Distance Metric**: Cosine Similarity / Distance
- **Tolerance Threshold**: 0.6 (default, configurable)

**Model Details**:
- Native support within OpenCV Zoo
- Extremely fast inference: <10ms per face
- Completely eliminates the need for large memory builds (e.g., dlib, cmake)

**Recognition Pipeline**:
```python
import cv2

# Initialize SFace Model
sface_recognizer = cv2.FaceRecognizerSF.create("models/face_recognition_sface.onnx", "")

# Extract 128D face encoding
feature = sface_recognizer.feature(aligned_face)

# Compare with stored encodings using Cosine Distance
similarity = sface_recognizer.match(known_features, query_features, cv2.FaceRecognizerSF_FR_COSINE)
distance = 1.0 - similarity
matches = distance <= 0.6
```

---

## Anti-Spoofing

**Techniques Implemented** (5 complementary methods):

1. **Blink Detection**
   - Method: MediaPipe Face Mesh landmarks integration
   - Requirement: Blinks within a set detection timeframe

2. **Face Movement Detection**
   - Tracks face position across frames
   - Measures Euclidean distance between centers
   - Threshold: 20+ pixels total movement
   - Detects static photos/screens

3. **Texture Variance Analysis**
   - Laplacian operator for edge detection
   - Variance calculation on face region
   - Threshold: Variance > 50
   - Differentiates real skin from printed photos

4. **Moiré Pattern Detection**
   - Fast Fourier Transform (FFT) analysis
   - Detects periodic patterns from screens
   - Threshold: FFT magnitude > 1000
   - Identifies photos of monitors

5. **Color Distribution Analysis**
   - HSV histogram diversity calculation
   - Standard deviation across channels
   - Threshold: Diversity > 15
   - Detects limited color gamut of prints

---

## Database

**Technology**: SQLite3 (local file-based database)

**Schema**:
```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    employee_id TEXT UNIQUE,
    name TEXT,
    face_encoding BLOB,  -- Pickled 128D numpy array (float32, SFace format)
    department TEXT,
    email TEXT,
    registered_date DATE,
    is_active INTEGER DEFAULT 1
);

-- Attendance table
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    punch_in_time TIMESTAMP,
    punch_out_time TIMESTAMP,
    date DATE,
    duration REAL,  -- Hours
    status TEXT,    -- 'present', 'half-day', 'early-exit'
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Why SQLite**:
- Zero configuration
- Serverless
- Self-contained
- Easy backup (single file)

---

## Training Process

### Pre-trained Model

**No training required** - the system uses pre-trained, lightweight models:

**OpenCV SFace Face Recognition Model**:
- Deeply optimized ONNX model
- Size: ~2MB
- Blazing fast performance on both local and cloud instances

**Face Detection Models**:
- MediaPipe BlazeFace: Packaged directly within the MediaPipe Python dependency.

### User Registration Process

**Enrollment Pipeline**:

1. **Multi-Sample Capture** (5-10 images per user)
   ```python
   REGISTRATION_SAMPLES = 5  # Configurable
   ```

2. **Quality Validation** (Real-time per image)
   - Face size: 50-500 pixels
   - Blur score: Laplacian variance > 50
   - Brightness: 30-230 (0-255 scale)
   - Face angle: ±30 degrees maximum

3. **Face Detection** (Per image)
   - Detect all faces using MediaPipe
   - Select largest face
   - Extract face region

4. **Encoding Extraction** (Per valid image)
   - Generate Float32 128D embedding via SFace
   - Store in temporary list

5. **Encoding Merging** (Multiple images → Single encoding)
   ```python
   # Average multiple encodings
   final_encoding = np.mean(encodings_list, axis=0)
   ```

6. **Database Storage**
   - Serialize encoding using pickle
   - Store as BLOB in SQLite

---

## Accuracy Expectations

### Face Detection

**MediaPipe Model**:
- Extremely reliable in varied conditions
- Good lighting: 98-99% detection rate
- Processing speed: >30 FPS on CPU

### Face Recognition

**Overall Accuracy**: ~98% in controlled conditions

**Distance-Based Accuracy (Cosine Distance)**:
- Distance < 0.2: Very High Confidence
- Distance 0.2-0.4: High Confidence
- Distance 0.4-0.6: Medium Confidence
- Distance > 0.6: Low Confidence (reject)

**Optimal Conditions**:
- Distance from camera: 50-150 cm
- Face angle: Front-facing (±15 degrees)
- Lighting: 300-1000 lux (normal indoor)
- Resolution: Minimum 80×80 pixels for face region
- Image quality: No blur, adequate contrast

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd face_authentication_attendance-_system-main
```

### Step 2: Install Dependencies and Download Models

Our system is designed to deploy seamlessly on lightweight environments like Render by avoiding memory-intensive compilation.

```bash
# Install dependencies
pip install -r requirements.txt

# Download required OpenCV SFace .onnx models
python download_models.py
```

**Dependencies** (see `requirements.txt`):
- opencv-python-headless 4.8.1+
- mediapipe 0.10+
- streamlit 1.32+

### Step 3: Run Application

```bash
# Launch web interface
streamlit run app.py
```

Application opens at: `http://localhost:8501`

*(End of technical documentation for the Lightweight Version)*
