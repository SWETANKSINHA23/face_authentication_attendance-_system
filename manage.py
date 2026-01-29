
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_server():
    """Run the Streamlit application"""
    print("Starting Face Authentication Attendance System...")
    cmd = ["streamlit", "run", "app.py"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to start the application.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopping application...")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    # Upgrade pip
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install dlib wheel first
    print("Installing dlib (pre-built wheel)...")
    dlib_url = "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.0-cp310-cp310-win_amd64.whl"
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", dlib_url], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to install pre-built dlib wheel. Trying standard install...")
        pass

    # Install other requirements
    print("Installing remaining dependencies...")
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        print("Error: requirements.txt not found.")
        sys.exit(1)
        
    # Initialize database
    print("Initializing database...")
    try:
        from models.database import init_database
        init_database()
        print("Database initialized successfully.")
    except ImportError:
        print("Warning: Could not import database module. Make sure you are running from the project root.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage Face Authentication Attendance System")
    parser.add_argument('command', choices=['run', 'install'], help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_server()
    elif args.command == 'install':
        install_dependencies()

if __name__ == "__main__":
    main()
