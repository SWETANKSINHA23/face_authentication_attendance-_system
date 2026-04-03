"""
Face recognition module using MediaPipe for detection and OpenCV SFace for recognition.
This is a lightweight alternative to dlib that works perfectly on Render free tier.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
import config

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1, # 0 for short-range, 1 for long-range (2-5 meters)
    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
)

# Initialize OpenCV SFace (FaceRecognizerSF)
# We use this to generate 128D encodings compatible with your existing logic.
try:
    # Ensure models exist
    if not os.path.exists(str(config.RECOGNITION_MODEL_PATH)):
        print(f"ERROR: Recognition model not found at {config.RECOGNITION_MODEL_PATH}")
        sface_recognizer = None
    else:
        sface_recognizer = cv2.FaceRecognizerSF.create(
            str(config.RECOGNITION_MODEL_PATH), 
            ""
        )
except Exception as e:
    print(f"Error initializing SFace: {e}")
    sface_recognizer = None

def get_face_locations_mediapipe(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using MediaPipe and return in (top, right, bottom, left) format"""
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_image)
    
    locations = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            left = int(bbox.xmin * w)
            top = int(bbox.ymin * h)
            right = int((bbox.xmin + bbox.width) * w)
            bottom = int((bbox.ymin + bbox.height) * h)
            
            # Constraint to image boundaries
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)
            
            # Reformat to (top, right, bottom, left) to match your app's expectations
            locations.append((top, right, bottom, left))
            
    return locations

def encode_face(image: np.ndarray, face_location: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
    """
    Generate face encoding using SFace
    """
    if sface_recognizer is None:
        return None
        
    if face_location is None:
        locations = get_face_locations_mediapipe(image)
        if not locations:
            return None
        face_location = locations[0]
    
    # SFace expects the image and the face bounding box in [x, y, w, h] format
    top, right, bottom, left = face_location
    bbox = np.array([left, top, right - left, bottom - top], dtype=np.float32)
    
    # SFace requires alignment first (though we can pass just the bbox)
    # For simplicity and speed on Render, we use the direct feature extraction
    # Note: SFace outputs 128D float32 vector
    try:
        # We need to simulate the 'aligned_face' or pass the detection result
        # Since we are using MediaPipe for detection, we'll crop and resize briefly 
        # to mimic what SFace expects if not using its internal detector.
        
        # Correct way for SFace with external detector:
        # SFace needs a 112x112 aligned face or we can use FaceRecognizerSF.feature()
        # To keep it simple, we'll crop
        face_img = image[top:bottom, left:right]
        if face_img.size == 0:
            return None
            
        aligned_face = cv2.resize(face_img, (112, 112))
        feature = sface_recognizer.feature(aligned_face)
        return feature[0]
    except Exception as e:
        print(f"Encoding error: {e}")
        return None

def compare_faces(known_encodings: List[np.ndarray], face_encoding: np.ndarray,
                  tolerance: float = None) -> List[bool]:
    """Compare encodings using Cosine Similarity (SFace standard)"""
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    matches = []
    for known in known_encodings:
        # Cast to float32 to prevent cv2 arithm_op error
        known_features = known.reshape(1, -1).astype(np.float32)
        query_features = face_encoding.reshape(1, -1).astype(np.float32)
        similarity = sface_recognizer.match(known_features, query_features, cv2.FaceRecognizerSF_FR_COSINE)
        distance = 1.0 - similarity
        matches.append(distance <= tolerance)
    return matches

def get_face_distances(known_encodings: List[np.ndarray], face_encoding: np.ndarray) -> np.ndarray:
    """Calculate distances (1.0 - cosine_similarity)"""
    distances = []
    for known in known_encodings:
        known_features = known.reshape(1, -1).astype(np.float32)
        query_features = face_encoding.reshape(1, -1).astype(np.float32)
        similarity = sface_recognizer.match(known_features, query_features, cv2.FaceRecognizerSF_FR_COSINE)
        distances.append(1.0 - similarity)
    return np.array(distances)

def find_best_match(face_encoding: np.ndarray, user_encodings: Dict[int, np.ndarray],
                    tolerance: float = None) -> Tuple[Optional[int], float]:
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    if not user_encodings: return None, float('inf')
    
    user_ids = list(user_encodings.keys())
    encodings = list(user_encodings.values())
    distances = get_face_distances(encodings, face_encoding)
    
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    
    if min_dist <= tolerance:
        return user_ids[min_idx], min_dist
    return None, min_dist

def calculate_similarity_percentage(distance: float) -> float:
    # distance is 0.0 to 2.0 (for cosine)
    # We normalize to 0-100%
    return max(0, (1.0 - distance) * 100)

def get_confidence_level(distance: float) -> str:
    if distance < 0.2: return "Very High"
    elif distance < 0.4: return "High"
    elif distance < 0.6: return "Medium"
    else: return "Low"

def recognize_face_in_frame(frame: np.ndarray, user_encodings: Dict[int, np.ndarray],
                            tolerance: float = None) -> List[Dict[str, Any]]:
    face_locations = get_face_locations_mediapipe(frame)
    results = []
    
    for loc in face_locations:
        encoding = encode_face(frame, loc)
        if encoding is not None:
            user_id, distance = find_best_match(encoding, user_encodings, tolerance)
            results.append({
                'user_id': user_id,
                'distance': float(distance),
                'confidence': get_confidence_level(distance),
                'location': loc,
                'similarity': calculate_similarity_percentage(distance)
            })
    return results

def validate_encoding(encoding: np.ndarray) -> bool:
    return isinstance(encoding, np.ndarray) and encoding.shape == (128,)

def merge_encodings(encodings: List[np.ndarray], method: str = 'average') -> np.ndarray:
    if not encodings: return None
    return np.mean(encodings, axis=0) if method == 'average' else np.median(encodings, axis=0)
