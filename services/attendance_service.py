"""
Attendance Service - Handles attendance tracking via face authentication
"""

import cv2
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple, Any

import config
from core import face_detector, face_recognizer
from utils import image_processing, camera
from models import database


class AttendanceResult:
    """Result of attendance operation"""
    
    def __init__(self, success: bool, user_id: int = None, user_name: str = None,
                 message: str = "", attendance_id: int = None, 
                 distance: float = None, confidence: str = None):
        self.success = success
        self.user_id = user_id
        self.user_name = user_name
        self.message = message
        self.attendance_id = attendance_id
        self.distance = distance
        self.confidence = confidence


def authenticate_face(frame: np.ndarray, tolerance: float = None) -> Tuple[Optional[int], float, str]:
    """
    Authenticate face from a single frame
    
    Args:
        frame: Input frame (BGR format)
        tolerance: Recognition tolerance (default from config)
    
    Returns:
        Tuple of (user_id, distance, confidence_level)
        Returns (None, inf, "No Match") if no match found
    """
    tolerance = tolerance or config.RECOGNITION_TOLERANCE
    
    # Preprocess frame for better recognition
    processed_frame = image_processing.preprocess_image(frame)
    
    # Detect face
    face_location = face_detector.get_largest_face(processed_frame)
    
    if face_location is None:
        return None, float('inf'), "No Face Detected"
    
    # Validate face quality
    quality = face_detector.validate_face_quality(processed_frame, face_location)
    
    if not quality['valid']:
        issues = ", ".join(quality['issues'])
        return None, float('inf'), f"Poor Quality: {issues}"
    
    # Generate face encoding
    face_encoding = face_recognizer.encode_face(processed_frame, face_location)
    
    if face_encoding is None:
        return None, float('inf'), "Encoding Failed"
    
    # Get all user encodings from database
    user_encodings = database.get_all_user_encodings()
    
    if not user_encodings:
        return None, float('inf'), "No Registered Users"
    
    # Find best match
    user_id, distance = face_recognizer.find_best_match(
        face_encoding, user_encodings, tolerance
    )
    
    if user_id is None:
        return None, distance, "No Match"
    
    # Get confidence level
    confidence = face_recognizer.get_confidence_level(distance)
    
    return user_id, distance, confidence


def detect_and_recognize_face(frame: np.ndarray, tolerance: float = None) -> Dict[str, Any]:
    """
    Detect and recognize face in frame with detailed results
    
    Args:
        frame: Input frame (BGR format)
        tolerance: Recognition tolerance
    
    Returns:
        Dict with recognition results: {
            'recognized': bool,
            'user_id': int or None,
            'user_name': str or None,
            'employee_id': str or None,
            'distance': float,
            'confidence': str,
            'similarity': float,
            'face_location': tuple or None,
            'message': str
        }
    """
    result = {
        'recognized': False,
        'user_id': None,
        'user_name': None,
        'employee_id': None,
        'distance': float('inf'),
        'confidence': 'Unknown',
        'similarity': 0.0,
        'face_location': None,
        'message': ''
    }
    
    # Authenticate face
    user_id, distance, confidence = authenticate_face(frame, tolerance)
    
    result['distance'] = distance
    result['confidence'] = confidence
    
    # Detect face location for visualization
    face_location = face_detector.get_largest_face(frame)
    result['face_location'] = face_location
    
    if user_id is None:
        result['message'] = confidence  # Confidence field contains error message if no match
        return result
    
    # Get user details from database
    user = database.get_user_by_id(user_id)
    
    if user is None:
        result['message'] = "User not found in database"
        return result
    
    # Calculate similarity percentage
    similarity = face_recognizer.calculate_similarity_percentage(distance)
    
    # Fill result
    result['recognized'] = True
    result['user_id'] = user_id
    result['user_name'] = user['name']
    result['employee_id'] = user['employee_id']
    result['similarity'] = similarity
    result['message'] = f"Recognized: {user['name']}"
    
    return result


def can_punch_in(user_id: int) -> Tuple[bool, str]:
    """
    Check if user can punch in
    
    Args:
        user_id: User ID
    
    Returns:
        Tuple of (can_punch_in, message)
    """
    today = date.today()
    
    # Get today's attendance records
    records = database.get_attendance_by_user_date(user_id, today)
    
    if not records:
        return True, "Ready to punch in"
    
    # Check latest record
    latest = records[0]  # Ordered by punch_in_time DESC
    
    # If latest record has no punch_out, user is already punched in
    if latest['punch_out_time'] is None:
        punch_in_time = datetime.fromisoformat(latest['punch_in_time'])
        return False, f"Already punched in at {punch_in_time.strftime('%H:%M:%S')}"
    
    # Check cooldown period
    last_punch_out = datetime.fromisoformat(latest['punch_out_time'])
    cooldown_end = last_punch_out + timedelta(seconds=config.PUNCH_IN_COOLDOWN)
    
    if datetime.now() < cooldown_end:
        remaining = (cooldown_end - datetime.now()).seconds
        return False, f"Please wait {remaining}s before punching in again"
    
    return True, "Ready to punch in"


def can_punch_out(user_id: int) -> Tuple[bool, str, Optional[int]]:
    """
    Check if user can punch out
    
    Args:
        user_id: User ID
    
    Returns:
        Tuple of (can_punch_out, message, attendance_id)
    """
    today = date.today()
    
    # Get today's attendance records
    records = database.get_attendance_by_user_date(user_id, today)
    
    if not records:
        return False, "Not punched in yet", None
    
    # Check latest record
    latest = records[0]
    
    # If already punched out
    if latest['punch_out_time'] is not None:
        punch_out_time = datetime.fromisoformat(latest['punch_out_time'])
        return False, f"Already punched out at {punch_out_time.strftime('%H:%M:%S')}", None
    
    # Check minimum time since punch-in
    punch_in_time = datetime.fromisoformat(latest['punch_in_time'])
    
    if (datetime.now() - punch_in_time).seconds < config.PUNCH_OUT_COOLDOWN:
        return False, "Too soon after punch-in", None
    
    return True, "Ready to punch out", latest['id']


def punch_in(user_id: int, timestamp: datetime = None) -> AttendanceResult:
    """
    Record punch-in for user
    
    Args:
        user_id: User ID
        timestamp: Punch-in timestamp (defaults to now)
    
    Returns:
        AttendanceResult object
    """
    timestamp = timestamp or datetime.now()
    
    # Get user details
    user = database.get_user_by_id(user_id)
    if not user:
        return AttendanceResult(
            success=False,
            message="User not found"
        )
    
    # Check if can punch in
    can_punch, message = can_punch_in(user_id)
    
    if not can_punch:
        return AttendanceResult(
            success=False,
            user_id=user_id,
            user_name=user['name'],
            message=message
        )
    
    # Create attendance record
    try:
        attendance_id = database.create_attendance_record(
            user_id=user_id,
            punch_in_time=timestamp,
            attendance_date=timestamp.date()
        )
        
        return AttendanceResult(
            success=True,
            user_id=user_id,
            user_name=user['name'],
            message=f"Punch-in successful at {timestamp.strftime('%H:%M:%S')}",
            attendance_id=attendance_id
        )
        
    except Exception as e:
        return AttendanceResult(
            success=False,
            user_id=user_id,
            user_name=user['name'],
            message=f"Database error: {str(e)}"
        )


def punch_out(user_id: int, timestamp: datetime = None) -> AttendanceResult:
    """
    Record punch-out for user
    
    Args:
        user_id: User ID
        timestamp: Punch-out timestamp (defaults to now)
    
    Returns:
        AttendanceResult object
    """
    timestamp = timestamp or datetime.now()
    
    # Get user details
    user = database.get_user_by_id(user_id)
    if not user:
        return AttendanceResult(
            success=False,
            message="User not found"
        )
    
    # Check if can punch out
    can_punch, message, attendance_id = can_punch_out(user_id)
    
    if not can_punch:
        return AttendanceResult(
            success=False,
            user_id=user_id,
            user_name=user['name'],
            message=message
        )
    
    # Update attendance record
    try:
        success = database.update_punch_out(attendance_id, timestamp)
        
        if not success:
            return AttendanceResult(
                success=False,
                user_id=user_id,
                user_name=user['name'],
                message="Failed to update attendance record"
            )
        
        # Get updated record to show duration
        record = database.get_attendance_by_user_date(user_id, timestamp.date())[0]
        duration = record['duration']
        status = record['status']
        
        return AttendanceResult(
            success=True,
            user_id=user_id,
            user_name=user['name'],
            message=f"Punch-out successful. Duration: {duration:.2f}h ({status})",
            attendance_id=attendance_id
        )
        
    except Exception as e:
        return AttendanceResult(
            success=False,
            user_id=user_id,
            user_name=user['name'],
            message=f"Database error: {str(e)}"
        )


def authenticate_and_punch_in(frame: np.ndarray, tolerance: float = None) -> AttendanceResult:
    """
    Authenticate face and punch in if recognized
    
    Args:
        frame: Input frame
        tolerance: Recognition tolerance
    
    Returns:
        AttendanceResult object
    """
    # Recognize face
    recognition = detect_and_recognize_face(frame, tolerance)
    
    if not recognition['recognized']:
        return AttendanceResult(
            success=False,
            message=f"Authentication failed: {recognition['message']}"
        )
    
    # Punch in
    result = punch_in(recognition['user_id'])
    result.distance = recognition['distance']
    result.confidence = recognition['confidence']
    
    return result


def authenticate_and_punch_out(frame: np.ndarray, tolerance: float = None) -> AttendanceResult:
    """
    Authenticate face and punch out if recognized
    
    Args:
        frame: Input frame
        tolerance: Recognition tolerance
    
    Returns:
        AttendanceResult object
    """
    # Recognize face
    recognition = detect_and_recognize_face(frame, tolerance)
    
    if not recognition['recognized']:
        return AttendanceResult(
            success=False,
            message=f"Authentication failed: {recognition['message']}"
        )
    
    # Punch out
    result = punch_out(recognition['user_id'])
    result.distance = recognition['distance']
    result.confidence = recognition['confidence']
    
    return result


def get_attendance_summary(user_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Get attendance summary for user
    
    Args:
        user_id: User ID
        days: Number of days to look back
    
    Returns:
        Dict with attendance statistics
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    records = database.get_attendance_history(user_id, start_date, end_date)
    
    total_days = 0
    present_days = 0
    half_days = 0
    total_hours = 0.0
    
    for record in records:
        if record['status'] == 'present':
            present_days += 1
            total_days += 1
        elif record['status'] == 'half-day':
            half_days += 1
            total_days += 1
        
        if record['duration']:
            total_hours += record['duration']
    
    avg_hours = total_hours / total_days if total_days > 0 else 0
    
    return {
        'total_days': total_days,
        'present_days': present_days,
        'half_days': half_days,
        'total_hours': total_hours,
        'average_hours': avg_hours,
        'period_days': days
    }


if __name__ == "__main__":
    # Initialize database
    database.init_database()
    
    print("Attendance service loaded")
    print("Use authenticate_and_punch_in() or authenticate_and_punch_out()")
