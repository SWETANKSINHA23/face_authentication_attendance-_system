"""
Anti-Spoofing / Liveness Detection Module
Detects photo/video spoofing attempts using multiple techniques
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import distance as dist

import config


class LivenessResult:
    """Result of liveness detection"""
    
    def __init__(self, is_live: bool, confidence: float, 
                 blink_detected: bool = False, movement_detected: bool = False,
                 texture_score: float = 0.0, reasons: List[str] = None):
        self.is_live = is_live
        self.confidence = confidence
        self.blink_detected = blink_detected
        self.movement_detected = movement_detected
        self.texture_score = texture_score
        self.reasons = reasons or []


def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection
    
    Eye landmarks should be 6 points in order:
    [0] outer corner, [1] top-left, [2] top-right, [3] inner corner,
    [4] bottom-right, [5] bottom-left
    
    Args:
        eye_landmarks: Array of 6 (x, y) points
    
    Returns:
        Eye aspect ratio value
    """
    # Vertical eye landmarks
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal eye landmark
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear


def detect_blink(eye_landmarks: np.ndarray, threshold: float = None) -> bool:
    """
    Detect blink based on eye aspect ratio
    
    Args:
        eye_landmarks: Eye landmark points (6 points)
        threshold: EAR threshold for blink (default from config)
    
    Returns:
        True if blink detected
    """
    threshold = threshold or config.EYE_AR_THRESHOLD
    
    ear = calculate_eye_aspect_ratio(eye_landmarks)
    
    return ear < threshold


def extract_eye_landmarks(face_landmarks: List[Tuple[int, int]], 
                         eye: str = "left") -> np.ndarray:
    """
    Extract eye landmarks from face landmarks
    
    Args:
        face_landmarks: 68-point face landmarks
        eye: "left" or "right"
    
    Returns:
        6 eye landmark points as numpy array
    """
    # 68-point landmark indices for eyes
    # Left eye: 36-41, Right eye: 42-47
    if eye == "left":
        return np.array(face_landmarks[36:42])
    else:
        return np.array(face_landmarks[42:48])


def detect_blinks_in_sequence(face_landmarks_sequence: List[List[Tuple[int, int]]],
                               min_blinks: int = None) -> Tuple[bool, int]:
    """
    Detect blinks in a sequence of frames
    
    Args:
        face_landmarks_sequence: List of face landmarks for each frame
        min_blinks: Minimum number of blinks required (default from config)
    
    Returns:
        Tuple of (blinks_detected, blink_count)
    """
    min_blinks = min_blinks or config.MIN_BLINKS_REQUIRED
    
    blink_count = 0
    was_blinking = False
    
    for face_landmarks in face_landmarks_sequence:
        # Get both eyes
        left_eye = extract_eye_landmarks(face_landmarks, "left")
        right_eye = extract_eye_landmarks(face_landmarks, "right")
        
        # Calculate average EAR
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if currently blinking
        is_blinking = avg_ear < config.EYE_AR_THRESHOLD
        
        # Count blink when eyes open after closing
        if was_blinking and not is_blinking:
            blink_count += 1
        
        was_blinking = is_blinking
    
    return blink_count >= min_blinks, blink_count


def calculate_face_movement(face_locations: List[Tuple[int, int, int, int]]) -> float:
    """
    Calculate total face movement across frames
    
    Args:
        face_locations: List of face bounding boxes (top, right, bottom, left)
    
    Returns:
        Total movement distance in pixels
    """
    if len(face_locations) < 2:
        return 0.0
    
    total_movement = 0.0
    
    for i in range(1, len(face_locations)):
        # Calculate center of face
        prev_top, prev_right, prev_bottom, prev_left = face_locations[i-1]
        curr_top, curr_right, curr_bottom, curr_left = face_locations[i]
        
        prev_center = ((prev_left + prev_right) / 2, (prev_top + prev_bottom) / 2)
        curr_center = ((curr_left + curr_right) / 2, (curr_top + curr_bottom) / 2)
        
        # Euclidean distance
        movement = dist.euclidean(prev_center, curr_center)
        total_movement += movement
    
    return total_movement


def check_face_movement(face_locations: List[Tuple[int, int, int, int]],
                       min_movement: float = None) -> Tuple[bool, float]:
    """
    Check if face has sufficient natural movement
    
    Args:
        face_locations: List of face bounding boxes
        min_movement: Minimum movement threshold (default from config)
    
    Returns:
        Tuple of (movement_detected, total_movement)
    """
    min_movement = min_movement or config.MIN_FACE_MOVEMENT
    
    total_movement = calculate_face_movement(face_locations)
    
    return total_movement >= min_movement, total_movement


def calculate_texture_variance(image: np.ndarray, face_region: Tuple[int, int, int, int]) -> float:
    """
    Calculate texture variance in face region
    Photos/screens have different texture patterns than real faces
    
    Args:
        image: Input image (BGR)
        face_region: Face bounding box (top, right, bottom, left)
    
    Returns:
        Variance score (higher = more texture variation = more likely real)
    """
    top, right, bottom, left = face_region
    
    # Extract face region
    face_img = image[top:bottom, left:right]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian (edge detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate variance
    variance = laplacian.var()
    
    return variance


def detect_screen_moire(image: np.ndarray, face_region: Tuple[int, int, int, int]) -> bool:
    """
    Detect moiré patterns typical in photos of screens
    
    Args:
        image: Input image (BGR)
        face_region: Face bounding box
    
    Returns:
        True if moiré pattern detected (likely spoof)
    """
    top, right, bottom, left = face_region
    
    # Extract face region
    face_img = image[top:bottom, left:right]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT to detect periodic patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Remove DC component (center)
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10] = 0
    
    # Check for strong periodic patterns
    max_magnitude = np.max(magnitude_spectrum)
    
    # Threshold (calibrated experimentally)
    return max_magnitude > config.MOIRE_THRESHOLD


def analyze_color_distribution(image: np.ndarray, face_region: Tuple[int, int, int, int]) -> float:
    """
    Analyze color distribution to detect printed photos
    Real faces have wider color range than printed photos
    
    Args:
        image: Input image (BGR)
        face_region: Face bounding box
    
    Returns:
        Color diversity score (lower = potential spoof)
    """
    top, right, bottom, left = face_region
    
    # Extract face region
    face_img = image[top:bottom, left:right]
    
    # Convert to HSV
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for each channel
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Calculate standard deviation (diversity)
    h_std = np.std(h_hist)
    s_std = np.std(s_hist)
    v_std = np.std(v_hist)
    
    # Combined diversity score
    diversity = (h_std + s_std + v_std) / 3.0
    
    return diversity


def is_live_face(frames: List[np.ndarray], 
                 face_locations: List[Tuple[int, int, int, int]],
                 face_landmarks_sequence: List[List[Tuple[int, int, int, int]]] = None) -> LivenessResult:
    """
    Comprehensive liveness detection using multiple techniques
    
    Args:
        frames: List of video frames
        face_locations: List of face bounding boxes for each frame
        face_landmarks_sequence: Optional list of face landmarks for each frame
    
    Returns:
        LivenessResult object
    """
    reasons = []
    scores = []
    
    # Technique 1: Blink Detection
    blink_detected = False
    blink_count = 0
    
    if face_landmarks_sequence and len(face_landmarks_sequence) > 0:
        blink_detected, blink_count = detect_blinks_in_sequence(face_landmarks_sequence)
        
        if blink_detected:
            scores.append(1.0)
            reasons.append(f"Blink detected ({blink_count} blinks)")
        else:
            scores.append(0.0)
            reasons.append(f"No blinks detected ({blink_count} blinks)")
    
    # Technique 2: Face Movement Detection
    movement_detected = False
    total_movement = 0.0
    
    if len(face_locations) > 1:
        movement_detected, total_movement = check_face_movement(face_locations)
        
        if movement_detected:
            scores.append(1.0)
            reasons.append(f"Natural movement detected ({total_movement:.1f}px)")
        else:
            scores.append(0.0)
            reasons.append(f"Insufficient movement ({total_movement:.1f}px)")
    
    # Technique 3: Texture Analysis
    texture_score = 0.0
    
    if len(frames) > 0 and len(face_locations) > 0:
        # Use middle frame
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]
        face_region = face_locations[mid_idx]
        
        variance = calculate_texture_variance(frame, face_region)
        texture_score = variance
        
        # Higher variance = more likely real
        if variance > config.MIN_TEXTURE_VARIANCE:
            scores.append(1.0)
            reasons.append(f"Good texture variance ({variance:.1f})")
        else:
            scores.append(0.5)
            reasons.append(f"Low texture variance ({variance:.1f})")
        
        # Check for moiré patterns
        moire_detected = detect_screen_moire(frame, face_region)
        if moire_detected:
            scores.append(0.0)
            reasons.append("Moiré pattern detected (screen/photo)")
        else:
            scores.append(0.8)
            reasons.append("No moiré pattern")
        
        # Color distribution
        color_diversity = analyze_color_distribution(frame, face_region)
        if color_diversity > config.MIN_COLOR_DIVERSITY:
            scores.append(1.0)
            reasons.append(f"Good color diversity ({color_diversity:.1f})")
        else:
            scores.append(0.5)
            reasons.append(f"Low color diversity ({color_diversity:.1f})")
    
    # Calculate overall confidence
    if scores:
        confidence = sum(scores) / len(scores)
    else:
        confidence = 0.0
    
    # Determine if live based on threshold
    is_live = confidence >= config.LIVENESS_THRESHOLD
    
    return LivenessResult(
        is_live=is_live,
        confidence=confidence,
        blink_detected=blink_detected,
        movement_detected=movement_detected,
        texture_score=texture_score,
        reasons=reasons
    )


def quick_liveness_check(frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Tuple[bool, float]:
    """
    Quick single-frame liveness check (texture only)
    Use for fast screening, not as reliable as multi-frame
    
    Args:
        frame: Single frame
        face_location: Face bounding box
    
    Returns:
        Tuple of (is_live, confidence)
    """
    # Texture variance
    variance = calculate_texture_variance(frame, face_location)
    
    # Moiré detection
    moire = detect_screen_moire(frame, face_location)
    
    # Color diversity
    color = analyze_color_distribution(frame, face_location)
    
    # Simple heuristic
    scores = []
    
    if variance > config.MIN_TEXTURE_VARIANCE:
        scores.append(1.0)
    else:
        scores.append(0.0)
    
    if not moire:
        scores.append(1.0)
    else:
        scores.append(0.0)
    
    if color > config.MIN_COLOR_DIVERSITY:
        scores.append(1.0)
    else:
        scores.append(0.0)
    
    confidence = sum(scores) / len(scores)
    is_live = confidence >= config.LIVENESS_THRESHOLD
    
    return is_live, confidence


if __name__ == "__main__":
    print("Anti-Spoofing Module Loaded")
    print("\nLiveness Detection Techniques:")
    print("  1. Blink Detection (Eye Aspect Ratio)")
    print("  2. Face Movement Analysis")
    print("  3. Texture Variance Analysis")
    print("  4. Moiré Pattern Detection")
    print("  5. Color Distribution Analysis")
