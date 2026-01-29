"""
Face detection module using face_recognition library
Handles face detection in images and video frames
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional, Dict
import config


def detect_faces(image: np.ndarray, model: str = None) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image
    
    Args:
        image: Input image (BGR format from OpenCV)
        model: Detection model ('hog' for CPU, 'cnn' for GPU)
    
    Returns:
        List of face locations as (top, right, bottom, left) tuples
    """
    model = model or config.FACE_DETECTION_MODEL
    
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_image,
        number_of_times_to_upsample=config.FACE_DETECTION_UPSAMPLE,
        model=model
    )
    
    return face_locations


def get_largest_face(image: np.ndarray, model: str = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect and return the largest face in the image
    Useful for single-person authentication scenarios
    
    Args:
        image: Input image (BGR format)
        model: Detection model ('hog' or 'cnn')
    
    Returns:
        Largest face location as (top, right, bottom, left) or None
    """
    face_locations = detect_faces(image, model)
    
    if not face_locations:
        return None
    
    # Find largest face by area
    largest = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
    
    return largest


def extract_face_region(image: np.ndarray, face_location: Tuple[int, int, int, int],
                        padding: int = 0) -> np.ndarray:
    """
    Extract face region from image
    
    Args:
        image: Input image
        face_location: Face location as (top, right, bottom, left)
        padding: Extra padding around face in pixels
    
    Returns:
        Cropped face image
    """
    top, right, bottom, left = face_location
    
    # Add padding
    height, width = image.shape[:2]
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    
    face_image = image[top:bottom, left:right]
    
    return face_image


def validate_face_size(face_location: Tuple[int, int, int, int]) -> bool:
    """
    Validate if face size is within acceptable range
    
    Args:
        face_location: Face location as (top, right, bottom, left)
    
    Returns:
        True if face size is valid
    """
    top, right, bottom, left = face_location
    
    width = right - left
    height = bottom - top
    
    # Check minimum and maximum size
    if width < config.MIN_FACE_SIZE or height < config.MIN_FACE_SIZE:
        return False
    
    if width > config.MAX_FACE_SIZE or height > config.MAX_FACE_SIZE:
        return False
    
    return True


def validate_face_quality(image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Dict[str, any]:
    """
    Validate face image quality
    Checks for blur, brightness, and size
    
    Args:
        image: Input image
        face_location: Face location as (top, right, bottom, left)
    
    Returns:
        Dict with quality metrics: {
            'valid': bool,
            'blur_score': float,
            'brightness': float,
            'size_valid': bool,
            'issues': List[str]
        }
    """
    issues = []
    
    # Extract face region
    face_img = extract_face_region(image, face_location)
    
    # Check size
    size_valid = validate_face_size(face_location)
    if not size_valid:
        issues.append("Face too small or too large")
    
    # Check blur using Laplacian variance
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = laplacian_var
    
    if blur_score < 100:  # Threshold for blur detection
        issues.append("Image too blurry")
    
    # Check brightness
    brightness = np.mean(gray)
    
    if brightness < 50:
        issues.append("Too dark")
    elif brightness > 200:
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
    """
    Draw bounding boxes around detected faces
    
    Args:
        image: Input image
        face_locations: List of face locations
        labels: Optional labels for each face
        color: Box color in BGR format
    
    Returns:
        Image with drawn boxes
    """
    output = image.copy()
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Draw rectangle
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        
        # Draw label if provided
        if labels and i < len(labels):
            cv2.putText(output, labels[i], (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return output


def get_face_landmarks(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
    """
    Get facial landmarks for detected faces
    Used for face alignment and anti-spoofing
    
    Args:
        image: Input image (BGR format)
        face_locations: Optional pre-detected face locations
    
    Returns:
        List of landmark dictionaries for each face
    """
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces if not provided
    if face_locations is None:
        face_locations = detect_faces(image)
    
    # Get landmarks
    landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
    
    return landmarks


def calculate_face_angle(landmarks: Dict) -> Tuple[float, float, float]:
    """
    Calculate face rotation angles from landmarks
    
    Args:
        landmarks: Facial landmarks dictionary
    
    Returns:
        Tuple of (pitch, yaw, roll) angles in degrees
    """
    # Simplified angle calculation using key points
    # For production, consider using more sophisticated methods
    
    if not landmarks:
        return (0.0, 0.0, 0.0)
    
    # Get key points
    nose_bridge = landmarks.get('nose_bridge', [])
    left_eye = landmarks.get('left_eye', [])
    right_eye = landmarks.get('right_eye', [])
    
    if not (nose_bridge and left_eye and right_eye):
        return (0.0, 0.0, 0.0)
    
    # Calculate roll (head tilt) from eye positions
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    roll = np.degrees(np.arctan2(dy, dx))
    
    # Simplified pitch and yaw (would need 3D model for accuracy)
    pitch = 0.0
    yaw = 0.0
    
    return (pitch, yaw, roll)


def is_frontal_face(landmarks: Dict, max_roll: float = 30.0) -> bool:
    """
    Check if face is approximately frontal
    
    Args:
        landmarks: Facial landmarks
        max_roll: Maximum allowed roll angle in degrees
    
    Returns:
        True if face is frontal enough
    """
    pitch, yaw, roll = calculate_face_angle(landmarks)
    
    return abs(roll) <= max_roll


if __name__ == "__main__":
    # Test face detection
    print("Face detection module loaded successfully")
    print(f"Using model: {config.FACE_DETECTION_MODEL}")
    print(f"Face size range: {config.MIN_FACE_SIZE}-{config.MAX_FACE_SIZE} pixels")
