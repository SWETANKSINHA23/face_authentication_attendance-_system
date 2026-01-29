"""
Camera utilities for video capture and frame processing
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import config


class Camera:
    """Camera handler for video capture"""
    
    def __init__(self, camera_index: int = None, width: int = None, height: int = None):
        """
        Initialize camera
        
        Args:
            camera_index: Camera device index (default from config)
            width: Frame width (default from config)
            height: Frame height (default from config)
        """
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width or config.CAMERA_WIDTH
        self.height = height or config.CAMERA_HEIGHT
        self.cap = None
        self.is_opened = False
    
    def initialize(self) -> bool:
        """
        Initialize and open camera
        
        Returns:
            True if camera opened successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        return ret, frame
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def get_properties(self) -> dict:
        """
        Get current camera properties
        
        Returns:
            Dict of camera properties
        """
        if not self.is_opened or self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
        }


def test_camera(camera_index: int = 0, duration: int = 5) -> bool:
    """
    Test if camera is working
    
    Args:
        camera_index: Camera device index
        duration: Test duration in seconds
    
    Returns:
        True if camera works
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return False
    
    print(f"Camera {camera_index} opened successfully")
    print(f"Testing for {duration} seconds...")
    
    frame_count = 0
    
    try:
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                cv2.imshow('Camera Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to read frame")
                break
        
        print(f"Captured {frame_count} frames")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return frame_count > 0


def list_available_cameras(max_index: int = 5) -> list:
    """
    List all available cameras
    
    Args:
        max_index: Maximum camera index to check
    
    Returns:
        List of available camera indices
    """
    available = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


def capture_image_from_camera(camera_index: int = 0, 
                              save_path: str = None) -> Optional[np.ndarray]:
    """
    Capture a single image from camera
    
    Args:
        camera_index: Camera device index
        save_path: Optional path to save image
    
    Returns:
        Captured image or None
    """
    with Camera(camera_index) as cam:
        ret, frame = cam.read_frame()
        
        if ret and save_path:
            cv2.imwrite(save_path, frame)
        
        return frame if ret else None


if __name__ == "__main__":
    # Test camera functionality
    print("Camera utilities loaded")
    print(f"Default camera index: {config.CAMERA_INDEX}")
    print(f"Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    
    available = list_available_cameras()
    print(f"Available cameras: {available}")
