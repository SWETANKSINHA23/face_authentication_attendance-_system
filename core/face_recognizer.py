"""
Face recognition module using face_recognition library (dlib ResNet)
Handles face encoding generation and matching
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import config


def encode_face(image: np.ndarray, face_location: Tuple[int, int, int, int] = None,
                num_jitters: int = None) -> Optional[np.ndarray]:
    """
    Generate 128-dimensional face encoding
    
    Args:
        image: Input image (BGR format from OpenCV)
        face_location: Optional pre-detected face location
        num_jitters: Number of times to resample for encoding (higher = more accurate)
    
    Returns:
        128D face encoding array or None if no face found
    """
    num_jitters = num_jitters or config.NUM_JITTERS
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # If face location not provided, detect it
    if face_location is None:
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=config.FACE_DETECTION_MODEL
        )
        
        if not face_locations:
            return None
        
        # Use the first/largest face
        face_location = face_locations[0]
    
    # Generate encoding
    encodings = face_recognition.face_encodings(
        rgb_image,
        known_face_locations=[face_location],
        num_jitters=num_jitters
    )
    
    if not encodings:
        return None
    
    return encodings[0]


def encode_faces_batch(image: np.ndarray, face_locations: List[Tuple[int, int, int, int]] = None,
                       num_jitters: int = None) -> List[np.ndarray]:
    """
    Generate face encodings for multiple faces in one image
    
    Args:
        image: Input image (BGR format)
        face_locations: Optional list of pre-detected face locations
        num_jitters: Number of times to resample
    
    Returns:
        List of face encodings
    """
    num_jitters = num_jitters or config.NUM_JITTERS
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces if not provided
    if face_locations is None:
        face_locations = face_recognition.face_locations(
            rgb_image,
            model=config.FACE_DETECTION_MODEL
        )
    
    # Generate encodings
    encodings = face_recognition.face_encodings(
        rgb_image,
        known_face_locations=face_locations,
        num_jitters=num_jitters
    )
    
    return encodings


def compare_faces(known_encodings: List[np.ndarray], face_encoding: np.ndarray,
                  tolerance: float = None) -> List[bool]:
    """
    Compare a face encoding against a list of known encodings
    
    Args:
        known_encodings: List of known face encodings
        face_encoding: Face encoding to compare
        tolerance: Distance threshold (lower = stricter, default 0.6)
    
    Returns:
        List of boolean matches
    """
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    matches = face_recognition.compare_faces(
        known_encodings,
        face_encoding,
        tolerance=tolerance
    )
    
    return matches


def get_face_distances(known_encodings: List[np.ndarray], 
                       face_encoding: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances between face encoding and known encodings
    
    Args:
        known_encodings: List of known face encodings
        face_encoding: Face encoding to compare
    
    Returns:
        Array of distances (lower = more similar)
    """
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    
    return distances


def find_best_match(face_encoding: np.ndarray, user_encodings: Dict[int, np.ndarray],
                    tolerance: float = None) -> Tuple[Optional[int], float]:
    """
    Find the best matching user for a face encoding
    
    Args:
        face_encoding: Face encoding to match
        user_encodings: Dict mapping user_id to face_encoding
        tolerance: Distance threshold
    
    Returns:
        Tuple of (user_id, distance) or (None, infinity) if no match
    """
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    if not user_encodings:
        return None, float('inf')
    
    # Get all user IDs and encodings
    user_ids = list(user_encodings.keys())
    encodings = list(user_encodings.values())
    
    # Calculate distances
    distances = get_face_distances(encodings, face_encoding)
    
    # Find minimum distance
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]
    
    # Check if within tolerance
    if min_distance <= tolerance:
        best_user_id = user_ids[min_distance_idx]
        return best_user_id, min_distance
    
    return None, min_distance


def find_all_matches(face_encoding: np.ndarray, user_encodings: Dict[int, np.ndarray],
                     tolerance: float = None, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find top K matching users for a face encoding
    
    Args:
        face_encoding: Face encoding to match
        user_encodings: Dict mapping user_id to face_encoding
        tolerance: Distance threshold
        top_k: Number of top matches to return
    
    Returns:
        List of (user_id, distance) tuples sorted by distance
    """
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    if not user_encodings:
        return []
    
    # Get all user IDs and encodings
    user_ids = list(user_encodings.keys())
    encodings = list(user_encodings.values())
    
    # Calculate distances
    distances = get_face_distances(encodings, face_encoding)
    
    # Create list of (user_id, distance) tuples
    matches = [(user_ids[i], distances[i]) for i in range(len(user_ids))]
    
    # Filter by tolerance
    matches = [(uid, dist) for uid, dist in matches if dist <= tolerance]
    
    # Sort by distance and take top K
    matches.sort(key=lambda x: x[1])
    matches = matches[:top_k]
    
    return matches


def calculate_similarity_percentage(distance: float, max_distance: float = 1.0) -> float:
    """
    Convert face distance to similarity percentage
    
    Args:
        distance: Face distance (0.0 = identical, 1.0 = very different)
        max_distance: Maximum distance to consider
    
    Returns:
        Similarity percentage (0-100)
    """
    # Invert and normalize distance to percentage
    similarity = max(0, (max_distance - distance) / max_distance * 100)
    
    return min(100, similarity)


def get_confidence_level(distance: float) -> str:
    """
    Get human-readable confidence level from face distance
    
    Args:
        distance: Face distance
    
    Returns:
        Confidence level string
    """
    if distance < 0.4:
        return "Very High"
    elif distance < 0.5:
        return "High"
    elif distance < 0.6:
        return "Medium"
    elif distance < 0.7:
        return "Low"
    else:
        return "Very Low"


def merge_encodings(encodings: List[np.ndarray], method: str = 'average') -> np.ndarray:
    """
    Merge multiple face encodings into a single representative encoding
    Useful when storing multiple samples per user
    
    Args:
        encodings: List of face encodings
        method: Merge method ('average' or 'median')
    
    Returns:
        Merged face encoding
    """
    if not encodings:
        raise ValueError("No encodings provided")
    
    if len(encodings) == 1:
        return encodings[0]
    
    encodings_array = np.array(encodings)
    
    if method == 'average':
        merged = np.mean(encodings_array, axis=0)
    elif method == 'median':
        merged = np.median(encodings_array, axis=0)
    else:
        raise ValueError(f"Unknown merge method: {method}")
    
    return merged


def validate_encoding(encoding: np.ndarray) -> bool:
    """
    Validate face encoding format
    
    Args:
        encoding: Face encoding to validate
    
    Returns:
        True if valid encoding
    """
    # Check if it's a numpy array
    if not isinstance(encoding, np.ndarray):
        return False
    
    # Check shape (should be 128-dimensional)
    if encoding.shape != (128,):
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
        return False
    
    return True


def recognize_face_in_frame(frame: np.ndarray, user_encodings: Dict[int, np.ndarray],
                            tolerance: float = None) -> List[Dict[str, Any]]:
    """
    Recognize all faces in a video frame
    
    Args:
        frame: Video frame (BGR format)
        user_encodings: Dict mapping user_id to face_encoding
        tolerance: Distance threshold
    
    Returns:
        List of recognition results: [{
            'user_id': int or None,
            'distance': float,
            'confidence': str,
            'location': tuple,
            'similarity': float
        }]
    """
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame,
        model=config.FACE_DETECTION_MODEL
    )
    
    # Encode detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Find best match
        user_id, distance = find_best_match(face_encoding, user_encodings, tolerance)
        
        result = {
            'user_id': user_id,
            'distance': distance,
            'confidence': get_confidence_level(distance),
            'location': face_location,
            'similarity': calculate_similarity_percentage(distance)
        }
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test face recognition module
    print("Face recognition module loaded successfully")
    print(f"Recognition tolerance: {config.RECOGNITION_TOLERANCE}")
    print(f"Encoding dimensions: 128D")
    print(f"Distance metric: Euclidean")
