"""
Mark Attendance Page
Punch-in and punch-out with face authentication
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.attendance_service import (
    detect_and_recognize_face,
    authenticate_and_punch_in,
    authenticate_and_punch_out,
    can_punch_in,
    can_punch_out
)
from utils.camera import Camera
from models.database import get_all_active_users, get_attendance_by_user_date
import config

st.title("✅ Mark Attendance")

st.markdown("""
Authenticate using face recognition to punch in or punch out.
Position your face in front of the camera and click the appropriate button.
""")

# Check if any users registered
users = get_all_active_users()

if not users:
    st.warning("""
    ⚠️ **No Users Registered**
    
    Please register at least one user before marking attendance.
    Navigate to "Register User" page to get started.
    """)
    st.stop()

# Camera selection & Configuration
if config.DEPLOYMENT_ENVIRONMENT == "local":
    camera_index = st.sidebar.number_input("Camera Index", 0, 5, config.CAMERA_INDEX, 
                                           help="Select camera device (usually 0 for built-in webcam)")
    st.sidebar.markdown("---")
    show_live_feed = st.sidebar.checkbox("Show Live Recognition Feed", value=False,
                                         help="Show continuous camera feed with real-time recognition")
else:
    # Cloud Config
    camera_index = 0
    show_live_feed = False
    st.sidebar.info("☁️ **Cloud Mode Active**")
    st.sidebar.caption("Using browser camera for capture.")

# Main attendance interface
# Main attendance interface
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Status")
    status_placeholder = st.empty()
    result_placeholder = st.empty()

with col1:
    st.subheader("📷 Camera Capture")
    
    if config.DEPLOYMENT_ENVIRONMENT == "cloud":
        # Cloud Mode UI
        action = st.radio("Select Action:", ["🟢 Punch In", "🔴 Punch Out"], horizontal=True)
        
        st.info(f"Ready to {action.split(' ')[1]} {action.split(' ')[2]}. Capture photo below.")
        
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Authenticate based on action
            if "Punch In" in action:
                with status_placeholder:
                    st.info("🔍 Authenticating Punch In...")
                result = authenticate_and_punch_in(frame, config.RECOGNITION_TOLERANCE)
            else:
                with status_placeholder:
                    st.info("🔍 Authenticating Punch Out...")
                result = authenticate_and_punch_out(frame, config.RECOGNITION_TOLERANCE)
            
            # Display result
            with result_placeholder:
                if result.success:
                    st.success(f"""
                    ✅ **{action.split(' ')[1]} {action.split(' ')[2]} Successful!**
                    
                    - **User**: {result.user_name}
                    - **Time**: {datetime.now().strftime('%H:%M:%S')}
                    - **Confidence**: {result.confidence}
                    
                    {result.message}
                    """)
                else:
                    st.error(f"""
                    ❌ **Authentication Failed**
                    
                    {result.message}
                    """)
                    if result.user_name:
                        st.info(f"User: {result.user_name}")

    else:
        # Local Mode UI (Original)
        # Placeholder for camera feed
        camera_placeholder = st.empty()
        
        # Capture buttons
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            punch_in_button = st.button("🟢 Punch In", use_container_width=True, type="primary")
        
        with button_col2:
            punch_out_button = st.button("🔴 Punch Out", use_container_width=True)

        # Live Feed Logic (Restored)
        if show_live_feed:
            live_feed_placeholder = st.empty()
            
            # We can't do a full infinite loop here easily inside Streamlit without rerun quirks,
            # but for now we follow the pattern of checking it. 
            # Actually, standard Streamlit pattern is to put the loop outside or handling it differently.
            # But let's just make sure the BUTTONS work first.
            pass

        # Handle punch in (Local)
        if punch_in_button:
            with status_placeholder:
                st.info("📷 Capturing image for authentication...")
            
            # Capture frame
            cam = Camera(camera_index)
            if not cam.initialize():
                with result_placeholder:
                    st.error("❌ Failed to initialize camera")
            else:
                # Read frame
                ret, frame = cam.read_frame()
                cam.release()
                
                if not ret:
                    with result_placeholder:
                        st.error("❌ Failed to capture image")
                else:
                    # Show captured image
                    with camera_placeholder:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption="Captured Image", width=700)
                    
                    # Authenticate and punch in
                    with status_placeholder:
                        st.info("🔍 Authenticating...")
                    
                    result = authenticate_and_punch_in(frame, config.RECOGNITION_TOLERANCE)
                    
                    # Display result
                    with status_placeholder:
                        st.empty()
                    
                    with result_placeholder:
                        if result.success:
                            st.success(f"""
                            ✅ **Punch-In Successful!**
                            
                            - **User**: {result.user_name}
                            - **Time**: {datetime.now().strftime('%H:%M:%S')}
                            - **Confidence**: {result.confidence}
                            - **Distance**: {result.distance:.3f}
                            
                            {result.message}
                            """)
                        else:
                            st.error(f"""
                            ❌ **Punch-In Failed**
                            
                            {result.message}
                            """)
                            
                            if result.user_name:
                                st.info(f"User: {result.user_name}")

        # Handle punch out (Local)
        if punch_out_button:
            with status_placeholder:
                st.info("📷 Capturing image for authentication...")
            
            # Capture frame
            cam = Camera(camera_index)
            if not cam.initialize():
                with result_placeholder:
                    st.error("❌ Failed to initialize camera")
            else:
                # Read frame
                ret, frame = cam.read_frame()
                cam.release()
                
                if not ret:
                    with result_placeholder:
                        st.error("❌ Failed to capture image")
                else:
                    # Show captured image
                    with camera_placeholder:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption="Captured Image", width=700)
                    
                    # Authenticate and punch out
                    with status_placeholder:
                        st.info("🔍 Authenticating...")
                    
                    result = authenticate_and_punch_out(frame, config.RECOGNITION_TOLERANCE)
                    
                    # Display result
                    with status_placeholder:
                        st.empty()
                    
                    with result_placeholder:
                        if result.success:
                            st.success(f"""
                            ✅ **Punch-Out Successful!**
                            
                            - **User**: {result.user_name}
                            - **Time**: {datetime.now().strftime('%H:%M:%S')}
                            - **Confidence**: {result.confidence}
                            - **Distance**: {result.distance:.3f}
                            
                            {result.message}
                            """)
                        else:
                            st.error(f"""
                            ❌ **Punch-Out Failed**
                            
                            {result.message}
                            """)
                            
                            if result.user_name:
                                st.info(f"User: {result.user_name}")

# Today's attendance
st.markdown("---")
st.subheader("📅 Today's Attendance")

today = date.today()
attendance_data = []

for user in users:
    records = get_attendance_by_user_date(user['id'], today)
    
    if records:
        latest = records[0]
        attendance_data.append({
            "Employee ID": user['employee_id'],
            "Name": user['name'],
            "Punch In": latest['punch_in_time'],
            "Punch Out": latest['punch_out_time'] or "Active",
            "Duration": f"{latest['duration']:.2f}h" if latest['duration'] else "In Progress",
            "Status": latest['status'].title()
        })

if attendance_data:
    st.dataframe(attendance_data, use_container_width=True, hide_index=True)
else:
    st.info("No attendance records for today yet.")

# Instructions
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    **How to Mark Attendance:**
    
    1. **Position yourself**: Ensure your face is clearly visible in the camera
    2. **Select action**: Click either "Punch In" or "Punch Out"
    3. **Wait for authentication**: System will capture and analyze your face
    4. **Check result**: Success or failure message will be displayed
    
    **Tips for Best Results:**
    - Ensure good lighting
    - Face the camera directly
    - Remove glasses if possible
    - Keep still during capture
    - Wait for confirmation before moving
    
    **Punch-In Rules:**
    - Can only punch in once per session
    - 5-minute cooldown between punch-in attempts
    - Must authenticate successfully
    
    **Punch-Out Rules:**
    - Must be punched in to punch out
    - Minimum 1 minute work time required
    - Work duration calculated automatically
    """)
