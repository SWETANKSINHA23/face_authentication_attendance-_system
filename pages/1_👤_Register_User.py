"""
User Registration Page
Register new users with face capture and quality validation
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.user_service import extract_face_embeddings, save_embeddings_to_db
from models.database import get_user_by_employee_id, get_all_active_users
from core import face_detector
import config

st.title("👤 User Registration")

st.markdown("""
Register new users with face capture for accurate recognition.
""")

# Registration form
with st.form("registration_form"):
    st.subheader("User Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_id = st.text_input("Employee ID *", placeholder="e.g., EMP001")
        name = st.text_input("Full Name *", placeholder="e.g., John Doe")
    
    with col2:
        email = st.text_input("Email (optional)", placeholder="john.doe@company.com")
        department = st.text_input("Department (optional)", placeholder="e.g., Engineering")
    
    submit_info = st.form_submit_button("✅ Save Info & Proceed to Camera", use_container_width=True)

# Store user info in session state
if submit_info:
    if not employee_id or not name:
        st.error("❌ Employee ID and Name are required!")
    else:
        # Check if user already exists
        existing_user = get_user_by_employee_id(employee_id)
        
        if existing_user:
            st.error(f"❌ Employee ID '{employee_id}' already exists!")
            st.info(f"Existing user: {existing_user['name']}")
        else:
            # Store in session
            st.session_state.reg_employee_id = employee_id
            st.session_state.reg_name = name
            st.session_state.reg_email = email
            st.session_state.reg_department = department
            st.session_state.info_saved = True
            st.success(f"✅ Info saved for {name}! Now capture your face below.")

# Camera capture section
st.markdown("---")
st.subheader("📸 Face Capture")

if not st.session_state.get('info_saved', False):
    st.info("👆 Please fill in user information above first")
else:
    st.success(f"Registering: **{st.session_state.reg_name}** ({st.session_state.reg_employee_id})")
    
    # Camera Logic based on Environment
    if config.DEPLOYMENT_ENVIRONMENT == "cloud":
        st.info("☁️ **Cloud Mode**: Using browser camera")
        
        st.markdown("""
        1. Click "Take a photo" below
        2. Allow camera access if prompted
        3. Position your face in the frame
        4. Click the camera button to capture
        5. Click "Register with this photo" to complete
        """)
        
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Show preview
            st.image(camera_photo, caption="Captured Image", width=300)
            
            # Validate face
            with st.spinner("Analyzing face..."):
                face_locations = face_detector.detect_faces(image)
            
            if len(face_locations) == 0:
                st.error("❌ No face detected. Please retake.")
            elif len(face_locations) > 1:
                st.warning(f"⚠️ Multiple faces detected ({len(face_locations)}). Please ensure only you are in the frame.")
            else:
                face_location = face_locations[0]
                quality = face_detector.validate_face_quality(image, face_location)
                
                if not quality['valid']:
                    st.warning(f"⚠️ Image quality issues: {', '.join(quality['issues'])}")
                    st.info("Try retaking with better lighting and positioning")
                else:
                    st.success("✅ Face detected with good quality!")
                
                # Register button
                if st.button("🚀 Register with this photo", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Extract embeddings from single image
                        encoding, issues = extract_face_embeddings([image])
                        
                        if encoding is None:
                            st.error("❌ Failed to generate face encoding")
                            if issues:
                                for issue in issues:
                                    st.write(f"  - {issue}")
                        else:
                            st.success("✅ Face encoding generated")
                            
                            # Save to database
                            try:
                                user_id = save_embeddings_to_db(
                                    employee_id=st.session_state.reg_employee_id,
                                    name=st.session_state.reg_name,
                                    face_encoding=encoding,
                                    email=st.session_state.reg_email or None,
                                    department=st.session_state.reg_department or None
                                )
                                
                                st.balloons()
                                st.success(f"""
                                ✅ **Registration Successful!**
                                
                                - Employee ID: {st.session_state.reg_employee_id}
                                - Name: {st.session_state.reg_name}
                                - User ID: {user_id}
                                - Registration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                """)
                                
                                # Clear session
                                st.session_state.info_saved = False
                                
                                st.info("""
                                **Next Steps:**
                                1. Navigate to "Mark Attendance" to test face recognition
                                2. View attendance in "View Records"
                                3. Check statistics in "Admin Dashboard"
                                """)
                                
                            except Exception as e:
                                st.error(f"❌ Database error: {str(e)}")

    else:
        # Local Mode - Show Tabs
        tab1, tab2 = st.tabs(["📷 Streamlit Camera (Recommended)", "🎥 OpenCV Multi-Sample"])
        
        with tab1:
            st.markdown("""
            **Simple and reliable camera capture**
            
            1. Click "Take a photo" below
            2. Allow camera access if prompted
            3. Position your face in the frame
            4. Click the camera button to capture
            5. Click "Register with this photo" to complete
            """)
            
            camera_photo = st.camera_input("Take a photo")
            
            if camera_photo is not None:
                # Convert to OpenCV format
                file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Show preview
                st.image(camera_photo, caption="Captured Image", width=300)
                
                # Validate face
                face_locations = face_detector.detect_faces(image)
                
                if len(face_locations) == 0:
                    st.error("❌ No face detected in the image. Please retake.")
                elif len(face_locations) > 1:
                    st.warning(f"⚠️ Multiple faces detected ({len(face_locations)}). Please ensure only you are in the frame.")
                else:
                    face_location = face_locations[0]
                    quality = face_detector.validate_face_quality(image, face_location)
                    
                    if not quality['valid']:
                        st.warning(f"⚠️ Image quality issues: {', '.join(quality['issues'])}")
                        st.info("Try retaking with better lighting and positioning")
                    else:
                        st.success("✅ Face detected with good quality!")
                    
                    # Register button
                    if st.button("🚀 Register with this photo", type="primary", use_container_width=True):
                        with st.spinner("Processing..."):
                            # Extract embeddings from single image
                            encoding, issues = extract_face_embeddings([image])
                            
                            if encoding is None:
                                st.error("❌ Failed to generate face encoding")
                                if issues:
                                    for issue in issues:
                                        st.write(f"  - {issue}")
                            else:
                                st.success("✅ Face encoding generated")
                                
                                # Save to database
                                try:
                                    user_id = save_embeddings_to_db(
                                        employee_id=st.session_state.reg_employee_id,
                                        name=st.session_state.reg_name,
                                        face_encoding=encoding,
                                        email=st.session_state.reg_email or None,
                                        department=st.session_state.reg_department or None
                                    )
                                    
                                    st.balloons()
                                    st.success(f"""
                                    ✅ **Registration Successful!**
                                    
                                    - Employee ID: {st.session_state.reg_employee_id}
                                    - Name: {st.session_state.reg_name}
                                    - User ID: {user_id}
                                    - Registration Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                    """)
                                    
                                    # Clear session
                                    st.session_state.info_saved = False
                                    
                                    st.info("""
                                    **Next Steps:**
                                    1. Navigate to "Mark Attendance" to test face recognition
                                    2. View attendance in "View Records"
                                    3. Check statistics in "Admin Dashboard"
                                    """)
                                    
                                except Exception as e:
                                    st.error(f"❌ Database error: {str(e)}")
        
        with tab2:
            st.markdown("""
            **Advanced: Capture 5 samples for better accuracy**
            
            ⚠️ **Note**: This method uses OpenCV which may have camera access issues.
            If it fails, please use the "Streamlit Camera" tab instead.
            """)
            
            num_samples = st.slider("Number of Samples", 3, 10, 5)
            camera_index = st.number_input("Camera Index", 0, 5, 0)
            
            if st.button("📸 Start Multi-Sample Capture"):
                st.info("📷 This feature requires OpenCV camera access...")
                st.warning("""
                ⚠️ **Camera Access Issues?**
                
                If you see "Failed to read frame from camera":
                1. Close other apps using camera (Zoom, Teams, etc.)
                2. Check Windows Privacy Settings → Camera
                3. Try different Camera Index (0, 1, or 2)
                4. **Or use the Streamlit Camera tab instead** (recommended)
                
                See CAMERA_TROUBLESHOOTING.md for detailed solutions.
                """)
                
                st.info("Multi-sample capture is not fully implemented in Streamlit mode. Please use the Streamlit Camera tab for reliable registration.")

# Show existing users
st.markdown("---")
st.subheader("📋 Registered Users")

users = get_all_active_users()

if users:
    st.write(f"Total registered users: **{len(users)}**")
    
    # Display as table
    user_data = []
    for user in users:
        user_data.append({
            "Employee ID": user['employee_id'],
            "Name": user['name'],
            "Department": user['department'] or "N/A",
            "Registered": user['registered_date']
        })
    
    st.dataframe(user_data, use_container_width=True, hide_index=True)
else:
    st.info("No users registered yet. Register the first user above!")

# Help section
with st.expander("❓ Need Help?"):
    st.markdown("""
    ### Registration Tips:
    
    **For best results:**
    - Ensure good lighting (not too bright or dark)
    - Face the camera directly
    - Remove glasses if possible
    - Keep a neutral expression
    - Ensure only your face is in frame
    
    **Camera Issues?**
    - Use the "Streamlit Camera" tab (most reliable)
    - Check camera permissions in Windows Settings
    - Close other apps using the camera
    - See `CAMERA_TROUBLESHOOTING.md` for detailed help
    
    **Single vs Multi-Sample:**
    - Single photo: Quick and easy, works reliably
    - Multi-sample: Better accuracy but may have camera issues
    - Recommendation: Start with single photo
    """)
