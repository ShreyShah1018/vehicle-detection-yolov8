import streamlit as st
import os
import time
from PIL import Image
from utils.youtube_stream import open_youtube_stream

# ========== Project Paths ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "videos", "user_uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "user_output")
EVAL_DIR = os.path.join(BASE_DIR, "output", "eval")

# ========== State Initialization ==========
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False
if "show_images" not in st.session_state:
    st.session_state.show_images = False

# ========== Page Config ==========
st.set_page_config(layout="wide", page_title="Vehicle Detection and Analytics")
st.title("🚗 Real-Time Vehicle Detection, Tracking & Analytics")

# ========== Video Source Selection ==========
st.sidebar.title("📥 Input Source")
source_type = st.sidebar.radio("Choose video input", ["Upload MP4", "YouTube Stream"])

input_path = None
is_stream = False

# ========== Source: Upload ==========
if source_type == "Upload MP4":
    uploaded_video = st.sidebar.file_uploader("📤 Upload a .mp4 video", type=["mp4"])
    if uploaded_video:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        input_path = os.path.join(UPLOAD_DIR, uploaded_video.name)

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success(f"✅ Uploaded: {uploaded_video.name}")

# ========== Source: YouTube ==========
elif source_type == "YouTube Stream":
    youtube_url = st.sidebar.text_input("Paste YouTube Live URL", placeholder="https://www.youtube.com/watch?v=...")
    if youtube_url:
        if not youtube_url.startswith("http"):
            st.warning("Please enter a valid YouTube link (starting with http/https).")
        else:
            input_path = youtube_url
            is_stream = True
            st.success("✅ YouTube URL accepted.")

# ========== Run Button ==========
if st.button("🔄 Run Tracking + Analytics"):
    from utils.pipeline import run_full_pipeline

    if input_path is None:
        st.error("❗ Please upload a video or enter a YouTube URL before running.")
        st.stop()

    st.session_state.run_clicked = True
    st.info("⏳ Processing video...")

    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = [
        "Step 1/4: Initializing model...",
        "Step 2/4: Tracking vehicles...",
        "Step 3/4: Generating analytics...",
        "Step 4/4: Saving results..."
    ]
    for step in steps:
        st.write(step)
        time.sleep(1.5)

    # ✅ Now use input_path and is_stream correctly
    run_full_pipeline(input_path, OUTPUT_DIR, is_stream=is_stream, timeout_seconds=30)

    duration = time.time() - start_time
    st.success(f"✅ Processing complete in {int(duration)} seconds.")

# ========== Analytics Dashboard ==========
if st.session_state.run_clicked:
    st.header("📊 Vehicle Analytics Dashboard")

    # ==== 1. Heatmap and Timeline ====
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Vehicle Density Heatmap")
        heatmap_path = os.path.join(OUTPUT_DIR, "vehicle_heatmap.png")
        if os.path.exists(heatmap_path):
            st.image(Image.open(heatmap_path), use_column_width=True)
        else:
            st.warning("Heatmap not found.")

    with col2:
        st.subheader("📈 Vehicle Count per Frame")
        timeline_path = os.path.join(OUTPUT_DIR, "vehicle_count_timeline.png")
        if os.path.exists(timeline_path):
            st.image(Image.open(timeline_path), use_column_width=True)
        else:
            st.warning("Timeline graph not found.")

    st.divider()

    # ==== 2. Model Evaluation ====
    st.subheader("📑 Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        conf_path = os.path.join(EVAL_DIR, "eval_confusion_matrix.png")
        if os.path.exists(conf_path):
            st.image(Image.open(conf_path), caption="Confusion Matrix")
        else:
            st.warning("Confusion matrix not found.")

    with col2:
        metric_path = os.path.join(EVAL_DIR, "eval_metrics.txt")
        if os.path.exists(metric_path):
            with open(metric_path, "r") as f:
                st.text(f.read())
        else:
            st.warning("Evaluation metrics not found.")

    st.divider()

    # ==== 3. Tracked Video Output Toggle ====
    col_btn, col_slider = st.columns([1, 2])

    with col_btn:
        if st.button("🖼️ Show Tracked Frames"):
            st.session_state.show_images = not st.session_state.show_images

    with col_slider:
        frame_skip = st.slider("Frame Skip", 1, 10, 2)
        show_stats = st.checkbox("Show Tracking Stats", value=True)

    # ==== 4. Display Tracked Frames ====
    if st.session_state.show_images:
        st.subheader("📹 Tracked Video with Bounding Boxes")
        from utils.video_utils import play_video_frames
        video_path = os.path.join(OUTPUT_DIR, "tracked_video.mp4")

        if os.path.exists(video_path):
            play_video_frames(video_path, frame_skip=frame_skip, show_stats=show_stats)
        else:
            st.warning("Tracked video not found.")

