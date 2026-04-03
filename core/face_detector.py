"""
Face detection module using MediaPipe
Handles face detection, quality validation, and landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import config


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1, # 0 for short-range, 1 for long-range
    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
)

# Initialize MediaPipe Face Mesh for landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)


def detect_faces(image: np.ndarray, model: str = None) -> List[Tuple[int, int, int, int]]:
    """Detect faces in an image using MediaPipe"""
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
            
            locations.append((top, right, bottom, left))
            
    return locations


def get_largest_face(image: np.ndarray, model: str = None) -> Optional[Tuple[int, int, int, int]]:
    """Detect and return the largest face in the image"""
    face_locations = detect_faces(image, model)
    
    if not face_locations:
        return None
    
    # Find largest face by area
    largest = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
    
    return largest


def extract_face_region(image: np.ndarray, face_location: Tuple[int, int, int, int],
                        padding: int = 0) -> np.ndarray:
    """Extract face region from image"""
    top, right, bottom, left = face_location
    
    height, width = image.shape[:2]
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    
    face_image = image[top:bottom, left:right]
    
    return face_image


def validate_face_size(face_location: Tuple[int, int, int, int]) -> bool:
    """Validate if face size is within acceptable range"""
    top, right, bottom, left = face_location
    
    width = right - left
    height = bottom - top
    
    if width < config.MIN_FACE_SIZE or height < config.MIN_FACE_SIZE:
        return False
    
    if width > config.MAX_FACE_SIZE or height > config.MAX_FACE_SIZE:
        return False
    
    return True


def validate_face_quality(image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Dict[str, any]:
    """Validate face image quality"""
    issues = []
    
    face_img = extract_face_region(image, face_location)
    if face_img.size == 0:
        return {'valid': False, 'issues': ["Invalid face region"]}
    
    size_valid = validate_face_size(face_location)
    if not size_valid:
        issues.append("Face too small or too large")
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = laplacian_var
    
    if blur_score < 50:  # MediaPipe handles blur better, lowered threshold
        issues.append("Image too blurry")
    
    brightness = np.mean(gray)
    
    if brightness < 30:
        issues.append("Too dark")
    elif brightness > 230:
        issues.append("Too bright")
    
    valid = len(issues) == 0 and size_valid
    
    return {
        'valid': valid,
        'blur_score': blur_score,
        'brightness': brightness,
        'size_valid': size_valid,
        'issues': issues
    }


def draw_face_boxes(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]],
                    labels: List[str] = None, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes around detected faces"""
    output = image.copy()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        
        if labels and i < len(labels):
            cv2.putText(output, labels[i], (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return output


def get_face_landmarks(image: np.ndarray, face_locations=None) -> List[Dict]:
    """
    Get 468 facial landmarks from MediaPipe Face Mesh
    We extract nose and eyes to match the old format expected by some modules
    """
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    landmarks_list = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Create a dictionary matching the old face_recognition format
            lms = face_landmarks.landmark
            
            # Simple average points for eyes and nose
            left_eye_indices = [33, 133, 160, 158, 153, 144]
            right_eye_indices = [362, 263, 387, 385, 380, 373]
            nose_bridge_indices = [168, 6, 197, 195]
            
            left_eye = [(int(lms[idx].x * w), int(lms[idx].y * h)) for idx in left_eye_indices]
            right_eye = [(int(lms[idx].x * w), int(lms[idx].y * h)) for idx in right_eye_indices]
            nose_bridge = [(int(lms[idx].x * w), int(lms[idx].y * h)) for idx in nose_bridge_indices]
            
            landmarks_list.append({
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose_bridge': nose_bridge
            })
            
    return landmarks_list


def calculate_face_angle(landmarks: Dict) -> Tuple[float, float, float]:
    """Calculate basic roll angle from MediaPipe landmarks"""
    if not landmarks:
        return (0.0, 0.0, 0.0)
    
    left_eye = landmarks.get('left_eye', [])
    right_eye = landmarks.get('right_eye', [])
    
    if not (left_eye and right_eye):
        return (0.0, 0.0, 0.0)
    
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    roll = np.degrees(np.arctan2(dy, dx))
    
    return (0.0, 0.0, roll)


def is_frontal_face(landmarks: Dict, max_roll: float = 30.0) -> bool:
    """Check if face is approximately frontal based on roll angle"""
    _, _, roll = calculate_face_angle(landmarks)
    return abs(roll) <= max_roll
