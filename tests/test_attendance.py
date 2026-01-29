"""
Attendance System Demo - Complete attendance workflow
Combines face recognition with punch-in/punch-out
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.attendance_service import (
    detect_and_recognize_face,
    authenticate_and_punch_in,
    authenticate_and_punch_out,
    get_attendance_summary
)
from utils.camera import Camera
from models.database import init_database, get_all_active_users
import config


def capture_for_attendance(action: str = "punch-in") -> tuple:
    """
    Capture frame and recognize face for attendance
    
    Args:
        action: "punch-in" or "punch-out"
    
    Returns:
        Tuple of (frame, recognition_result)
    """
    print(f"\n{'='*60}")
    print(f" {action.upper().replace('-', ' ')}")
    print(f"{'='*60}")
    print("\nPosition your face in front of the camera...")
    print("Press SPACE to capture, ESC to cancel\n")
    
    cam = Camera(config.CAMERA_INDEX)
    if not cam.initialize():
        print("Error: Failed to initialize camera")
        return None, None
    
    try:
        while True:
            ret, frame = cam.read_frame()
            
            if not ret:
                print("Error: Failed to read frame")
                return None, None
            
            # Show live preview with recognition
            display_frame = frame.copy()
            
            # Try to recognize face in real-time
            result = detect_and_recognize_face(frame)
            
            # Draw results
            if result['face_location']:
                top, right, bottom, left = result['face_location']
                
                if result['recognized']:
                    color = (0, 255, 0)
                    label = f"{result['user_name']}"
                    status = f"Ready - Press SPACE"
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                    status = result['message']
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, label, (left, top - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_frame, status, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show instructions
            cv2.putText(display_frame, f"{action.upper()}: SPACE to capture, ESC to cancel",
                       (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Attendance System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                return frame, result
            elif key == 27:  # ESC
                return None, None
    
    finally:
        cam.release()
        cv2.destroyAllWindows()


def main():
    """Main attendance demo"""
    
    print("\n" + "="*70)
    print(" FACE AUTHENTICATION ATTENDANCE SYSTEM")
    print("="*70)
    
    # Initialize database
    init_database()
    
    # Check registered users
    users = get_all_active_users()
    
    if not users:
        print("\n⚠️  No users registered!")
        print("Please run 'python test_registration.py' first.\n")
        return
    
    print(f"\nRegistered Users: {len(users)}")
    for user in users:
        print(f"  - [{user['employee_id']}] {user['name']}")
    
    while True:
        print("\n" + "-"*70)
        print("MENU:")
        print("  1. Punch In")
        print("  2. Punch Out")
        print("  3. View Attendance Summary")
        print("  4. Real-time Recognition Demo")
        print("  5. Exit")
        print("-"*70)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            # Punch In
            frame, recognition = capture_for_attendance("punch-in")
            
            if frame is None:
                print("Cancelled")
                continue
            
            if not recognition['recognized']:
                print(f"\n❌ Authentication failed: {recognition['message']}")
                continue
            
            # Process punch-in
            result = authenticate_and_punch_in(frame)
            
            if result.success:
                print(f"\n✅ {result.message}")
                print(f"   User: {result.user_name}")
                print(f"   Confidence: {result.confidence} ({result.distance:.3f})")
            else:
                print(f"\n❌ {result.message}")
        
        elif choice == '2':
            # Punch Out
            frame, recognition = capture_for_attendance("punch-out")
            
            if frame is None:
                print("Cancelled")
                continue
            
            if not recognition['recognized']:
                print(f"\n❌ Authentication failed: {recognition['message']}")
                continue
            
            # Process punch-out
            result = authenticate_and_punch_out(frame)
            
            if result.success:
                print(f"\n✅ {result.message}")
                print(f"   User: {result.user_name}")
                print(f"   Confidence: {result.confidence} ({result.distance:.3f})")
            else:
                print(f"\n❌ {result.message}")
        
        elif choice == '3':
            # View Summary
            print("\nEnter Employee ID:")
            employee_id = input("> ").strip()
            
            from models.database import get_user_by_employee_id
            user = get_user_by_employee_id(employee_id)
            
            if not user:
                print(f"User {employee_id} not found")
                continue
            
            summary = get_attendance_summary(user['id'], days=30)
            
            print(f"\n{'='*60}")
            print(f" ATTENDANCE SUMMARY - {user['name']}")
            print(f"{'='*60}")
            print(f"  Period: Last {summary['period_days']} days")
            print(f"  Total Days Worked: {summary['total_days']}")
            print(f"  Present Days: {summary['present_days']}")
            print(f"  Half Days: {summary['half_days']}")
            print(f"  Total Hours: {summary['total_hours']:.2f}h")
            print(f"  Average Hours/Day: {summary['average_hours']:.2f}h")
            print(f"{'='*60}")
        
        elif choice == '4':
            # Real-time recognition
            print("\nLaunching real-time recognition demo...")
            print("(This will open in a new window)")
            
            import subprocess
            subprocess.Popen([sys.executable, "test_recognition.py"])
        
        elif choice == '5':
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
