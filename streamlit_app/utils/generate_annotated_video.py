import cv2
from ultralytics import YOLO
import os

def generate_annotated_video(input_path, output_path, model_path, confidence=0.4):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open input video.")
        return

    width = 640
    height = 360
    fps = 15  # lower fps to reduce size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize first
        frame = cv2.resize(frame, (width, height))

        # Run YOLO
        results = model.predict(source=frame, imgsz=640, conf=confidence, save=False, verbose=False)[0]
        annotated_frame = results.plot()

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"✅ Optimized annotated video saved at: {output_path}")
