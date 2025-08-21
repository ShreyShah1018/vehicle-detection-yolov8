
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os


MODEL_PATH = r"C:\Users\sshre\OneDrive\Desktop\Resume_Projects\vehicle-detection-yolov8\models\yolov8n-vehicle3\weights\best.pt"
model = YOLO(MODEL_PATH)

import yt_dlp

def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict.get("url", None)

def play_video_frames(video_path, frame_skip=2, show_stats=False):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    if not cap.isOpened():
        st.error("❌ Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_skip == 0:
            # Run inference and annotate
            results = model.predict(source=frame, imgsz=640, conf=0.4, save=False, verbose=False)[0]
            annotated_frame = results.plot()

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(frame_rgb), caption=f"Frame {frame_num}", use_column_width=True)

            if show_stats:
                st.markdown(f"`Frame: {frame_num}` | `Detections: {len(results.boxes)}`")

        frame_num += 1
        if frame_num > 100:
            break  # Limit frame display to prevent Streamlit timeout

    cap.release()
