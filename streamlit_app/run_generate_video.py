import os
import sys
from utils.generate_annotated_video import generate_annotated_video

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_video = os.path.join(BASE_DIR, "..", "videos", "sample_video.mp4")
output_video = os.path.join(BASE_DIR, "..", "output", "annotated_output.mp4")
model_path = r"C:\Users\sshre\OneDrive\Desktop\Resume_Projects\vehicle-detection-yolov8\models\yolov8n-vehicle3\weights\best.pt"

generate_annotated_video(input_video, output_video, model_path)
