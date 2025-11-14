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
    speed = st.slider("Playback speed (frames/sec)", min_value=1, max_value=60, value=10)
    st.subheader("Processing Optimization")
    frame_skip = st.slider("Process every N frames (1=all frames, 2=every 2nd frame)", min_value=1, max_value=5, value=2)
    resize_factor = st.slider("Resize factor for processing (0.5=half size, 1.0=full size)", min_value=0.3, max_value=1.0, value=0.7, step=0.1)
    st.subheader("Background Learning")
    learning_frames = st.slider("Learning frames for background", min_value=5, max_value=50, value=20)

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
                
                # Skip frames for performance if configured
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                    
                # Create a copy for processing
                processed_frame = frame.copy()
                
                # Resize frame for faster processing
                if resize_factor < 1.0:
                    height, width = processed_frame.shape[:2]
                    new_width = int(width * resize_factor)
                    new_height = int(height * resize_factor)
                    resized_frame = cv2.resize(processed_frame, (new_width, new_height))
                else:
                    resized_frame = processed_frame
                
                # Only process background subtraction for selected frames
                fg_mask = back_subtractor.apply(resized_frame)
                
                # Scale back coordinates if frame was resized
                scale_factor = 1.0 / resize_factor if resize_factor < 1.0 else 1.0
                
                # Only start detecting after learning period
                if frame_idx > learning_frames:
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    object_count = 0
                    for cnt in contours:
                        if cv2.contourArea(cnt) > (300 * resize_factor * resize_factor):
                            x, y, w, h = cv2.boundingRect(cnt)
                            # Scale coordinates back to original size
                            if resize_factor < 1.0:
                                x = int(x * scale_factor)
                                y = int(y * scale_factor)
                                w = int(w * scale_factor)
                                h = int(h * scale_factor)
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), box_bgr, 2)
                            object_count += 1
                    
                    status_text.text(f"Frame {frame_idx}/{frame_count} - Objects detected: {object_count}")
                else:
                    status_text.text(f"Learning background... Frame {frame_idx}/{learning_frames}")

                stframe.image(convert_img(processed_frame), channels="RGB", use_container_width=True)
                
                # Resize mask back to original size for display
                if resize_factor < 1.0:
                    fg_mask_display = cv2.resize(fg_mask, (processed_frame.shape[1], processed_frame.shape[0]))
                else:
                    fg_mask_display = fg_mask
                stframe2.image(fg_mask_display, channels="GRAY", use_container_width=True)
                
                frame_idx += 1
                progress.progress(min(frame_idx / frame_count, 1.0))
                
                # Much faster timing control
                time.sleep(max(0.02, sleep_time / frame_skip))
                
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