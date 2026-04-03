import os
import requests
from pathlib import Path

def download_models():
    """
    Download OpenCV YuNet (Detection) and SFace (Recognition) ONNX models
    """
    # Use the current directory (project root)
    models_dir = Path(__file__).parent / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Official OpenCV Zoo URLs (Updated to 'main' branch)
    model_urls = {
        "face_detection_yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    }

    for name, url in model_urls.items():
        dst_path = models_dir / name
        if not dst_path.exists():
            print(f"Downloading {name}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(dst_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {name} to {dst_path}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            print(f"{name} already exists.")

if __name__ == "__main__":
    download_models()
