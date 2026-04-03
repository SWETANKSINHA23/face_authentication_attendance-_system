"""
Configuration file for Face Authentication Attendance System
"""

import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "database.db"
FACE_IMAGES_DIR = DATA_DIR / "face_images"
MODELS_DIR = BASE_DIR / "models"
DETECTION_MODEL_PATH = MODELS_DIR / "face_detection_yunet.onnx"
RECOGNITION_MODEL_PATH = MODELS_DIR / "face_recognition_sface.onnx"

# Deployment Settings
# 'local': Use local webcam (cv2) - Faster, Real-time
# 'cloud': Use browser camera (st.camera_input) - Required for Streamlit Cloud/Render
DEPLOYMENT_ENVIRONMENT = "cloud"

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.5  # MediaPipe/YuNet confidence threshold
MIN_FACE_SIZE = 50  # Minimum face size in pixels
MAX_FACE_SIZE = 500  # Maximum face size in pixels

# Face Recognition Settings
FACE_MATCH_THRESHOLD = 0.6  # Lower is more strict (0.0-1.0)
RECOGNITION_TOLERANCE = 0.6  # Distance threshold for face matching
MIN_FACE_SIZE = 50  # Minimum face size in pixels
MAX_FACE_SIZE = 500  # Maximum face size in pixels

# Anti-Spoofing Settings
ENABLE_ANTI_SPOOFING = True

# Blink Detection
BLINK_DETECTION_ENABLED = True
EYE_AR_THRESHOLD = 0.25  # Eye aspect ratio threshold for blink detection
MIN_BLINKS_REQUIRED = 2  # Minimum blinks needed for liveness
LIVENESS_DETECTION_TIME = 4  # Seconds to collect data for liveness check

# Face Movement Detection
FACE_MOVEMENT_ENABLED = True
MIN_FACE_MOVEMENT = 20  # Minimum pixels of movement required

# Texture Analysis
TEXTURE_ANALYSIS_ENABLED = True
MIN_TEXTURE_VARIANCE = 100  # Minimum Laplacian variance for real face
MOIRE_THRESHOLD = 1000  # FFT magnitude threshold for moiré pattern detection
MIN_COLOR_DIVERSITY = 15  # Minimum color histogram diversity

# Overall Liveness Thresholds
LIVENESS_THRESHOLD = 0.6  # Overall confidence threshold (0-1)


# Registration Settings
REGISTRATION_SAMPLES = 5  # Number of face samples to capture during registration
CAPTURE_DELAY = 1  # Seconds between captures
MIN_REGISTRATION_QUALITY = 0.5  # Minimum image quality score (0-1)

# Attendance Settings
WORK_START_TIME = "09:00"
WORK_END_TIME = "18:00"
MIN_WORK_HOURS = 8
HALF_DAY_HOURS = 4
PUNCH_IN_COOLDOWN = 300  # 5 minutes cooldown between punch-ins
PUNCH_OUT_COOLDOWN = 60  # 1 minute cooldown between punch-outs

# Image Processing
IMAGE_PREPROCESSING = True
HISTOGRAM_EQUALIZATION = True
FACE_ALIGNMENT = True
BRIGHTNESS_ADJUSTMENT = True
TARGET_BRIGHTNESS = 128  # 0-255

# UI Settings
STREAMLIT_THEME = "light"
STREAMLIT_TITLE = "Face Authentication Attendance System"
STREAMLIT_ICON = "📸"
SHOW_CONFIDENCE_SCORE = True
SHOW_PROCESSING_TIME = True

# Database Settings
DB_ECHO = False  # Set to True for SQL debugging

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "app.log"

# Security
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Performance
ENABLE_GPU = False  # Set to True if CUDA-enabled GPU available
BATCH_PROCESSING = False
CACHE_FACE_ENCODINGS = True

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
