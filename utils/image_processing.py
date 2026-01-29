"""
Image preprocessing utilities for face recognition
Handles lighting normalization, quality assessment, and enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import config


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing pipeline to improve face recognition accuracy
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Preprocessed image
    """
    if not config.IMAGE_PREPROCESSING:
        return image
    
    processed = image.copy()
    
    # Apply histogram equalization if enabled
    if config.HISTOGRAM_EQUALIZATION:
        processed = equalize_histogram(processed)
    
    # Adjust brightness if enabled
    if config.BRIGHTNESS_ADJUSTMENT:
        processed = adjust_brightness(processed, config.TARGET_BRIGHTNESS)
    
    return processed


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast
    Useful for varying lighting conditions
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        Equalized image
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split channels
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_equalized = clahe.apply(y)
    
    # Merge channels
    ycrcb_equalized = cv2.merge([y_equalized, cr, cb])
    
    # Convert back to BGR
    result = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)
    
    return result


def adjust_brightness(image: np.ndarray, target_brightness: int = 128) -> np.ndarray:
    """
    Adjust image brightness to target level
    
    Args:
        image: Input image (BGR format)
        target_brightness: Target mean brightness (0-255)
    
    Returns:
        Brightness-adjusted image
    """
    # Convert to grayscale to calculate brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    
    # Calculate adjustment factor
    adjustment = target_brightness - current_brightness
    
    # Apply adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=adjustment)
    
    return adjusted


def reduce_noise(image: np.ndarray, strength: int = 5) -> np.ndarray:
    """
    Reduce image noise using bilateral filter
    
    Args:
        image: Input image
        strength: Noise reduction strength (higher = more smoothing)
    
    Returns:
        Denoised image
    """
    denoised = cv2.bilateralFilter(image, strength, 75, 75)
    
    return denoised


def sharpen_image(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen image to enhance edges
    
    Args:
        image: Input image
        amount: Sharpening amount (0-2, default 1.0)
    
    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) * amount / 9
    
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate image blur score using Laplacian variance
    Higher score = sharper image
    
    Args:
        image: Input image
    
    Returns:
        Blur score (higher is better, <100 is blurry)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance


def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculate average image brightness
    
    Args:
        image: Input image
    
    Returns:
        Brightness value (0-255)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    return brightness


def assess_image_quality(image: np.ndarray) -> Dict[str, any]:
    """
    Comprehensive image quality assessment
    
    Args:
        image: Input image
    
    Returns:
        Dict with quality metrics
    """
    blur_score = calculate_blur_score(image)
    brightness = calculate_brightness(image)
    
    # Determine quality issues
    issues = []
    
    if blur_score < 100:
        issues.append("Image is blurry")
    
    if brightness < 50:
        issues.append("Image is too dark")
    elif brightness > 200:
        issues.append("Image is too bright")
    
    # Calculate overall quality score (0-1)
    blur_quality = min(1.0, blur_score / 500)  # Normalize blur score
    
    # Brightness quality (optimal around 128)
    brightness_diff = abs(brightness - 128)
    brightness_quality = max(0, 1 - (brightness_diff / 128))
    
    overall_quality = (blur_quality + brightness_quality) / 2
    
    return {
        'blur_score': blur_score,
        'brightness': brightness,
        'quality_score': overall_quality,
        'is_acceptable': overall_quality >= config.MIN_REGISTRATION_QUALITY,
        'issues': issues
    }


def resize_image(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image


def normalize_face_orientation(image: np.ndarray, landmarks: Dict) -> np.ndarray:
    """
    Align face to canonical orientation using eye positions
    
    Args:
        image: Input image
        landmarks: Facial landmarks dictionary
    
    Returns:
        Aligned face image
    """
    if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
        return image
    
    # Get eye centers
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    
    # Calculate angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Get center point between eyes
    eye_center = ((left_eye[0] + right_eye[0]) / 2, 
                  (left_eye[1] + right_eye[1]) / 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    
    # Apply rotation
    height, width = image.shape[:2]
    aligned = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return aligned


if __name__ == "__main__":
    print("Image processing utilities loaded successfully")
