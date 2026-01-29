"""
User Service - Handles user registration and management
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import config
from core import face_detector, face_recognizer
from utils import image_processing, camera
from models import database


class RegistrationResult:
    """Result of user registration"""
    
    def __init__(self, success: bool, user_id: int = None, message: str = "", 
                 captured_images: int = 0, issues: List[str] = None):
        self.success = success
        self.user_id = user_id
        self.message = message
        self.captured_images = captured_images
        self.issues = issues or []


def capture_face_images(user_id: str, name: str, num_images: int = None,
                       camera_index: int = 0, show_preview: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """
    Capture multiple face images for registration
    
    Args:
        user_id: Employee ID for the user
        name: User's full name
        num_images: Number of images to capture (default from config)
        camera_index: Camera device index
        show_preview: Whether to show live preview
    
    Returns:
        Tuple of (captured_images, issues_list)
    """
    num_images = num_images or config.REGISTRATION_SAMPLES
    captured_images = []
    issues = []
    
    print(f"\nStarting face capture for {name} (ID: {user_id})")
    print(f"Will capture {num_images} images with {config.CAPTURE_DELAY}s delay between each")
    print("Position your face in the camera and look straight ahead...")
    
    # Initialize camera
    cam = camera.Camera(camera_index)
    if not cam.initialize():
        issues.append("Failed to initialize camera")
        return captured_images, issues
    
    try:
        capture_count = 0
        last_capture_time = 0
        
        while capture_count < num_images:
            ret, frame = cam.read_frame()
            
            if not ret:
                issues.append("Failed to read frame from camera")
                break
            
            # Preprocess frame
            processed_frame = image_processing.preprocess_image(frame)
            
            # Detect face
            face_locations = face_detector.detect_faces(processed_frame)
            
            # Draw status on frame
            display_frame = frame.copy()
            current_time = time.time()
            
            if len(face_locations) == 0:
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif len(face_locations) > 1:
                cv2.putText(display_frame, "Multiple faces detected - Only one person allowed", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw all face boxes
                display_frame = face_detector.draw_face_boxes(display_frame, face_locations, 
                                                              color=(0, 0, 255))
            
            else:
                # Single face detected
                face_location = face_locations[0]
                
                # Validate face quality
                quality = face_detector.validate_face_quality(processed_frame, face_location)
                
                # Draw face box
                color = (0, 255, 0) if quality['valid'] else (0, 165, 255)
                display_frame = face_detector.draw_face_boxes(display_frame, [face_location], 
                                                              color=color)
                
                # Show quality status
                if quality['valid']:
                    # Check if enough time has passed since last capture
                    if current_time - last_capture_time >= config.CAPTURE_DELAY:
                        # Capture image
                        captured_images.append(processed_frame.copy())
                        capture_count += 1
                        last_capture_time = current_time
                        
                        cv2.putText(display_frame, 
                                   f"Captured {capture_count}/{num_images}!", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Flash effect
                        display_frame = cv2.addWeighted(display_frame, 0.7, 
                                                        np.ones_like(display_frame) * 255, 0.3, 0)
                    else:
                        # Waiting for delay
                        remaining = config.CAPTURE_DELAY - (current_time - last_capture_time)
                        cv2.putText(display_frame, 
                                   f"Ready - Next capture in {remaining:.1f}s ({capture_count}/{num_images})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Show quality issues
                    issue_text = ", ".join(quality['issues'])
                    cv2.putText(display_frame, f"Quality issue: {issue_text}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Show preview if enabled
            if show_preview:
                cv2.imshow('Face Registration', display_frame)
                
                # Allow early exit with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    issues.append("Registration cancelled by user")
                    break
    
    finally:
        cam.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    return captured_images, issues


def extract_face_embeddings(images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Extract face embeddings from multiple images and merge them
    
    Args:
        images: List of face images
    
    Returns:
        Tuple of (merged_encoding, issues_list)
    """
    issues = []
    encodings = []
    
    for i, image in enumerate(images):
        # Detect face
        face_location = face_detector.get_largest_face(image)
        
        if face_location is None:
            issues.append(f"No face detected in image {i+1}")
            continue
        
        # Generate encoding
        encoding = face_recognizer.encode_face(image, face_location)
        
        if encoding is None:
            issues.append(f"Failed to generate encoding for image {i+1}")
            continue
        
        # Validate encoding
        if not face_recognizer.validate_encoding(encoding):
            issues.append(f"Invalid encoding for image {i+1}")
            continue
        
        encodings.append(encoding)
    
    if not encodings:
        return None, issues
    
    # Merge encodings using average
    merged_encoding = face_recognizer.merge_encodings(encodings, method='average')
    
    return merged_encoding, issues


def save_embeddings_to_db(employee_id: str, name: str, face_encoding: np.ndarray,
                          email: str = None, department: str = None) -> int:
    """
    Save user and face embeddings to database
    
    Args:
        employee_id: Unique employee ID
        name: Full name
        face_encoding: 128D face encoding
        email: Email address (optional)
        department: Department name (optional)
    
    Returns:
        User ID from database
    
    Raises:
        sqlite3.IntegrityError: If employee_id or email already exists
    """
    user_id = database.create_user(
        employee_id=employee_id,
        name=name,
        face_encoding=face_encoding,
        email=email,
        department=department
    )
    
    return user_id


def register_user(employee_id: str, name: str, email: str = None, 
                 department: str = None, num_samples: int = None,
                 camera_index: int = 0, show_preview: bool = True) -> RegistrationResult:
    """
    Complete user registration workflow
    
    Args:
        employee_id: Unique employee ID
        name: Full name
        email: Email address (optional)
        department: Department (optional)
        num_samples: Number of face samples to capture
        camera_index: Camera device index
        show_preview: Show live preview during capture
    
    Returns:
        RegistrationResult object
    """
    print(f"\n{'='*60}")
    print(f"FACE REGISTRATION - {name}")
    print(f"{'='*60}\n")
    
    # Check if user already exists
    existing_user = database.get_user_by_employee_id(employee_id)
    if existing_user:
        return RegistrationResult(
            success=False,
            message=f"Employee ID {employee_id} already exists",
            issues=["Duplicate employee ID"]
        )
    
    # Step 1: Capture face images
    print("Step 1/3: Capturing face images...")
    captured_images, capture_issues = capture_face_images(
        employee_id, name, num_samples, camera_index, show_preview
    )
    
    if not captured_images:
        return RegistrationResult(
            success=False,
            message="No valid face images captured",
            captured_images=0,
            issues=capture_issues
        )
    
    print(f"✓ Captured {len(captured_images)} images")
    
    # Step 2: Extract face embeddings
    print("\nStep 2/3: Generating face embeddings...")
    face_encoding, encoding_issues = extract_face_embeddings(captured_images)
    
    if face_encoding is None:
        return RegistrationResult(
            success=False,
            message="Failed to generate face encoding",
            captured_images=len(captured_images),
            issues=capture_issues + encoding_issues
        )
    
    print(f"✓ Generated 128D face encoding")
    
    # Step 3: Save to database
    print("\nStep 3/3: Saving to database...")
    try:
        user_id = save_embeddings_to_db(
            employee_id=employee_id,
            name=name,
            face_encoding=face_encoding,
            email=email,
            department=department
        )
        
        print(f"✓ Registration successful! User ID: {user_id}")
        print(f"\n{'='*60}\n")
        
        return RegistrationResult(
            success=True,
            user_id=user_id,
            message=f"Successfully registered {name}",
            captured_images=len(captured_images),
            issues=capture_issues + encoding_issues
        )
        
    except Exception as e:
        return RegistrationResult(
            success=False,
            message=f"Database error: {str(e)}",
            captured_images=len(captured_images),
            issues=capture_issues + encoding_issues + [str(e)]
        )


def update_user_face(user_id: int, camera_index: int = 0, 
                    num_samples: int = None) -> RegistrationResult:
    """
    Update face encoding for an existing user
    
    Args:
        user_id: Database user ID
        camera_index: Camera device index
        num_samples: Number of new face samples
    
    Returns:
        RegistrationResult object
    """
    # Get existing user
    user = database.get_user_by_id(user_id)
    if not user:
        return RegistrationResult(
            success=False,
            message=f"User ID {user_id} not found"
        )
    
    print(f"\nUpdating face encoding for {user['name']}...")
    
    # Capture new images
    captured_images, capture_issues = capture_face_images(
        user['employee_id'], user['name'], num_samples, camera_index
    )
    
    if not captured_images:
        return RegistrationResult(
            success=False,
            message="No valid images captured",
            issues=capture_issues
        )
    
    # Generate new encoding
    face_encoding, encoding_issues = extract_face_embeddings(captured_images)
    
    if face_encoding is None:
        return RegistrationResult(
            success=False,
            message="Failed to generate face encoding",
            issues=capture_issues + encoding_issues
        )
    
    # Update database
    success = database.update_user(user_id, face_encoding=face_encoding)
    
    if success:
        return RegistrationResult(
            success=True,
            user_id=user_id,
            message=f"Successfully updated face encoding for {user['name']}",
            captured_images=len(captured_images)
        )
    else:
        return RegistrationResult(
            success=False,
            message="Failed to update database"
        )


if __name__ == "__main__":
    # Initialize database
    database.init_database()
    
    # Example usage
    print("User registration service loaded")
    print("Use register_user() to register new users")
