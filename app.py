import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# --- Page Config ---
st.set_page_config(
    page_title="Object Tracking App",
    layout="wide",
    page_icon=":guardsman:"
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.header("Object Tracking App")
    st.markdown(
        "Upload a video file to detect and track moving objects using background subtraction. "
        "Results are displayed side by side for easy comparison."
    )
    st.markdown("---")
    st.info("Supported formats: mp4, avi, mov")
    st.markdown("---")
    st.subheader("Detection Box Color")
    box_color = st.color_picker("Pick a color for the detection box", "#FF0000")
    st.subheader("Video Speed")
    speed = st.slider("Playback speed (frames/sec)", min_value=1, max_value=60, value=15)
    st.subheader("Background Learning")
    learning_frames = st.slider("Learning frames for background", min_value=10, max_value=100, value=30)

# --- Main Title & Description ---
st.title("🎯 Object Tracking with Background Subtraction")
st.markdown(
    """
    <style>
    .big-font {font-size:18px !important;}
    </style>
    <div class="big-font">
    This application allows you to upload a video file and perform object tracking using background subtraction.<br>
    <b>Original</b> and <b>Processed</b> frames are displayed side by side.
    </div>
    """,
    unsafe_allow_html=True
)

def convert_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

uploaded_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Error opening video file.")
    else:
        st.success("✅ Video uploaded successfully!")
        
        # Video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        st.info(f"📊 Video Info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Controls
        col_control1, col_control2, col_control3 = st.columns(3)
        with col_control1:
            process_video = st.button("🎬 Start Processing", type="primary")
        with col_control2:
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            stop_processing = st.button("⏹️ Stop")
        with col_control3:
            reset_video = st.button("🔄 Reset")
            
        if reset_video:
            st.session_state.processing = False
            st.rerun()
            
        if stop_processing:
            st.session_state.processing = False
            
        if process_video or st.session_state.get('processing', False):
            st.session_state.processing = True
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Frame")
                stframe = st.empty()
            with col2:
                st.subheader("Detected Objects")
                stframe2 = st.empty()

            back_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            progress = st.progress(0)
            status_text = st.empty()

            # Convert hex color to BGR for OpenCV
            hex_color = box_color.lstrip('#')
            box_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Calculate sleep time for proper frame rate
            target_fps = min(speed, fps) if fps > 0 else speed
            sleep_time = 1.0 / target_fps
            
            while cap.isOpened() and st.session_state.get('processing', False):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Create a copy for processing
                processed_frame = frame.copy()
                fg_mask = back_subtractor.apply(processed_frame)
                
                # Only start detecting after learning period
                if frame_idx > learning_frames:
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    object_count = 0
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 300:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), box_bgr, 2)
                            object_count += 1
                    
                    status_text.text(f"Frame {frame_idx}/{frame_count} - Objects detected: {object_count}")
                else:
                    status_text.text(f"Learning background... Frame {frame_idx}/{learning_frames}")

                stframe.image(convert_img(processed_frame), channels="RGB", use_column_width=True)
                stframe2.image(fg_mask, channels="GRAY", use_column_width=True)
                
                frame_idx += 1
                progress.progress(min(frame_idx / frame_count, 1.0))
                
                # Proper timing control - break into smaller sleeps to be more responsive
                sleep_remaining = sleep_time
                while sleep_remaining > 0 and st.session_state.get('processing', False):
                    time.sleep(min(0.1, sleep_remaining))
                    sleep_remaining -= 0.1
                
            cap.release()
            st.session_state.processing = False
            
            if frame_idx >= frame_count:
                st.success("🎉 Video processing completed!")
            else:
                st.info("⏸️ Processing stopped by user.")

# --- Footer ---
st.markdown(
    "<hr><center>Made with ❤️ using Streamlit & OpenCV | 2025</center>",
    unsafe_allow_html=True
)