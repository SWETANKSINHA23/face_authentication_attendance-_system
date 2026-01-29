"""
Test Anti-Spoofing / Liveness Detection
Real-time demonstration of liveness detection techniques
"""

import sys
from pathlib import Path
import cv2
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.anti_spoofing import is_live_face, quick_liveness_check
from core.face_detector import detect_faces, get_face_landmarks
from utils.camera import Camera
import config


def collect_liveness_data(camera_index: int = 0, duration: float = 4.0):
    """
    Collect frames and landmarks for liveness detection
    
    Args:
        camera_index: Camera device index
        duration: Duration to collect data (seconds)
    
    Returns:
        Tuple of (frames, face_locations, face_landmarks_sequence)
    """
    frames = []
    face_locations = []
    face_landmarks_sequence = []
    
    cam = Camera(camera_index)
    if not cam.initialize():
        return None, None, None
    
    print(f"\nCollecting data for {duration} seconds...")
    print("Please:")
    print("  1. Look at the camera")
    print("  2. Blink naturally")
    print("  3. Move your head slightly\n")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cam.read_frame()
            
            if not ret:
                continue
            
            # Detect face
            face_locs = detect_faces(frame, model=config.FACE_DETECTION_MODEL)
            
            if face_locs:
                # Use first detected face
                face_location = face_locs[0]
                
                # Get landmarks
                landmarks = get_face_landmarks(frame, face_location)
                
                if landmarks:
                    frames.append(frame.copy())
                    face_locations.append(face_location)
                    face_landmarks_sequence.append(landmarks)
                    
                    frame_count += 1
                    
                    # Draw on frame
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Show countdown
                    remaining = duration - (time.time() - start_time)
                    cv2.putText(frame, f"Collecting: {remaining:.1f}s",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frames: {frame_count}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Liveness Detection - Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cam.release()
        cv2.destroyAllWindows()
    
    print(f"Collected {len(frames)} frames with face data\n")
    
    return frames, face_locations, face_landmarks_sequence


def display_liveness_result(result):
    """Display liveness detection result"""
    
    print("\n" + "="*70)
    print(" LIVENESS DETECTION RESULT")
    print("="*70)
    
    if result.is_live:
        print("\n✅ LIVE FACE DETECTED")
    else:
        print("\n❌ SPOOF DETECTED (Photo/Video)")
    
    print(f"\nOverall Confidence: {result.confidence:.2%}")
    print(f"Blink Detected: {'✅' if result.blink_detected else '❌'}")
    print(f"Movement Detected: {'✅' if result.movement_detected else '❌'}")
    print(f"Texture Score: {result.texture_score:.1f}")
    
    print("\nDetailed Analysis:")
    for i, reason in enumerate(result.reasons, 1):
        print(f"  {i}. {reason}")
    
    print("="*70 + "\n")


def test_real_time_liveness():
    """Test liveness detection with real-time capture"""
    
    print("\n" + "="*70)
    print(" ANTI-SPOOFING / LIVENESS DETECTION TEST")
    print("="*70)
    
    print("\nThis test will:")
    print("  1. Capture video for 4 seconds")
    print("  2. Analyze blinks and face movement")
    print("  3. Check texture and color patterns")
    print("  4. Determine if face is live or spoofed")
    
    input("\nPress ENTER to start...")
    
    # Collect data
    frames, face_locations, landmarks = collect_liveness_data(
        camera_index=config.CAMERA_INDEX,
        duration=config.LIVENESS_DETECTION_TIME
    )
    
    if not frames:
        print("❌ Failed to collect data")
        return
    
    if len(frames) < 10:
        print(f"⚠️  Only collected {len(frames)} frames, need at least 10")
        print("Please ensure:")
        print("  - Camera is working")
        print("  - Face is visible during entire duration")
        print("  - Lighting is adequate")
        return
    
    # Perform liveness detection
    print("Analyzing liveness...")
    result = is_live_face(frames, face_locations, landmarks)
    
    # Display result
    display_liveness_result(result)


def test_quick_check():
    """Test quick single-frame liveness check"""
    
    print("\n" + "="*70)
    print(" QUICK LIVENESS CHECK (Single Frame)")
    print("="*70)
    print("\nCapturing single frame...\n")
    
    cam = Camera(config.CAMERA_INDEX)
    if not cam.initialize():
        print("❌ Failed to initialize camera")
        return
    
    try:
        # Capture for 2 seconds to get good frame
        for _ in range(60):
            ret, frame = cam.read_frame()
            
            if ret:
                face_locs = detect_faces(frame)
                
                if face_locs:
                    top, right, bottom, left = face_locs[0]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to capture", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Quick Liveness Check', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    if face_locs:
                        # Perform quick check
                        is_live, confidence = quick_liveness_check(frame, face_locs[0])
                        
                        print(f"\nResult: {'LIVE' if is_live else 'SPOOF'}")
                        print(f"Confidence: {confidence:.2%}\n")
                        
                        # Show result on frame
                        color = (0, 255, 0) if is_live else (0, 0, 255)
                        label = "LIVE" if is_live else "SPOOF"
                        cv2.putText(frame, label, (left, top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                        cv2.imshow('Quick Liveness Check', frame)
                        cv2.waitKey(3000)
                    break
                elif key == ord('q'):
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def main():
    """Main test menu"""
    
    while True:
        print("\n" + "="*70)
        print(" ANTI-SPOOFING TEST MENU")
        print("="*70)
        print("\n1. Full Liveness Detection (4 seconds, multi-technique)")
        print("2. Quick Liveness Check (single frame, texture only)")
        print("3. Exit")
        print("-"*70)
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            test_real_time_liveness()
        elif choice == '2':
            test_quick_check()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
