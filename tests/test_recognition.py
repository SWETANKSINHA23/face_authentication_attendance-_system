"""
Real-time Face Recognition Demo
Continuous face recognition from camera feed with FPS optimization
"""

import sys
from pathlib import Path
import cv2
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.attendance_service import detect_and_recognize_face
from utils.camera import Camera
from models.database import init_database, get_all_active_users
import config


class RealTimeFaceRecognition:
    """Real-time face recognition with performance optimization"""
    
    def __init__(self, tolerance: float = None, process_every_n_frames: int = 3):
        """
        Initialize real-time recognition
        
        Args:
            tolerance: Recognition tolerance (default from config)
            process_every_n_frames: Process every Nth frame for performance
        """
        self.tolerance = tolerance or config.RECOGNITION_TOLERANCE
        self.process_every_n = process_every_n_frames
        self.frame_count = 0
        
        # Cache for last recognition result
        self.last_result = None
        self.last_face_location = None
        
        # FPS calculation
        self.fps = 0
        self.frame_times = []
        
        # Load registered users count
        self.registered_users_count = 0
        self.update_users_count()
    
    def update_users_count(self):
        """Update count of registered users"""
        users = get_all_active_users()
        self.registered_users_count = len(users)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame for face recognition
        
        Args:
            frame: Input frame
        
        Returns:
            Recognition result dict
        """
        self.frame_count += 1
        
        # Process every Nth frame for performance
        if self.frame_count % self.process_every_n == 0:
            result = detect_and_recognize_face(frame, self.tolerance)
            self.last_result = result
            self.last_face_location = result['face_location']
        
        return self.last_result
    
    def draw_results(self, frame: np.ndarray, result: dict = None) -> np.ndarray:
        """
        Draw recognition results on frame
        
        Args:
            frame: Input frame
            result: Recognition result (uses last result if None)
        
        Returns:
            Frame with drawn results
        """
        if result is None:
            result = self.last_result
        
        if result is None:
            return frame
        
        display_frame = frame.copy()
        
        # Draw face box if detected
        if result['face_location']:
            top, right, bottom, left = result['face_location']
            
            # Color based on recognition status
            if result['recognized']:
                color = (0, 255, 0)  # Green for recognized
                label = f"{result['user_name']} ({result['employee_id']})"
                confidence_text = f"{result['similarity']:.1f}% - {result['confidence']}"
            else:
                color = (0, 0, 255)  # Red for unknown/error
                label = "Unknown"
                confidence_text = result['message']
            
            # Draw rectangle
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label_height = 50
            cv2.rectangle(display_frame, (left, top - label_height), (right, top), color, -1)
            
            # Draw text
            cv2.putText(display_frame, label, (left + 6, top - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, confidence_text, (left + 6, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info panel
        self.draw_info_panel(display_frame)
        
        return display_frame
    
    def draw_info_panel(self, frame: np.ndarray):
        """Draw info panel with FPS and statistics"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Info text
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Registered Users: {self.registered_users_count}",
            f"Tolerance: {self.tolerance:.2f}",
            f"Press 'q' to quit"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frames
        self.frame_times = self.frame_times[-30:]
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            self.fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
    
    def run(self, camera_index: int = 0):
        """
        Run real-time recognition loop
        
        Args:
            camera_index: Camera device index
        """
        print("\n" + "="*70)
        print(" REAL-TIME FACE RECOGNITION")
        print("="*70)
        print(f"\nRegistered Users: {self.registered_users_count}")
        print(f"Recognition Tolerance: {self.tolerance}")
        print(f"Processing every {self.process_every_n} frames")
        print("\nPress 'q' to quit\n")
        
        # Initialize camera
        cam = Camera(camera_index)
        if not cam.initialize():
            print("Error: Failed to initialize camera")
            return
        
        try:
            while True:
                # Read frame
                ret, frame = cam.read_frame()
                
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Draw results
                display_frame = self.draw_results(frame, result)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('Face Recognition - Attendance System', display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Refresh user count
                    self.update_users_count()
                    print(f"Refreshed: {self.registered_users_count} users")
        
        finally:
            cam.release()
            cv2.destroyAllWindows()
            print("\nRecognition stopped")


def main():
    """Main function"""
    
    # Initialize database
    init_database()
    
    # Check if users are registered
    users = get_all_active_users()
    
    if not users:
        print("\n⚠️  WARNING: No users registered!")
        print("Please run 'python test_registration.py' to register users first.\n")
        
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Create and run recognition system
    recognizer = RealTimeFaceRecognition(
        tolerance=config.RECOGNITION_TOLERANCE,
        process_every_n_frames=3  # Process every 3rd frame for better FPS
    )
    
    recognizer.run(camera_index=config.CAMERA_INDEX)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
