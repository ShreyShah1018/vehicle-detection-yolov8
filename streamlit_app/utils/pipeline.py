import sys
import os

# Dynamically add the project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



from tracker.inference_tracker import YOLODeepSORTTracker
from analytics.generate_zone_heatmap import generate_zone_vehicle_heatmap
from analytics.framewise_vehicle_count import generate_vehicle_count_timeline


import pandas as pd
import cv2
def run_full_pipeline(input_path, output_dir, is_stream=False, timeout_seconds=30):
    from datetime import datetime
    import pandas as pd
    import cv2

    tracker = YOLODeepSORTTracker()

    if is_stream:
        from utils.youtube_stream import open_youtube_stream
        cap = open_youtube_stream(input_path, max_height=720)
        if cap is None:
            raise RuntimeError("Failed to open YouTube stream. Try another URL or lower resolution.")
    else:
        cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if fps is 0
    output_video = os.path.join(output_dir, "tracked_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_data = []
    frame_no = 0
    start_time = datetime.now()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stop if timeout reached
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if is_stream and elapsed_time >= timeout_seconds:
            print(f"[INFO] Timeout reached: {timeout_seconds}s")
            break

        tracked_frame, tracks = tracker.track_video_frame(frame)
        out.write(tracked_frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            frame_data.append({
                "frame": frame_no,
                "track_id": track_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        frame_no += 1

    cap.release()
    out.release()

    # Save tracking log
    csv_path = os.path.join(output_dir, "tracking_log.csv")
    df = pd.DataFrame(frame_data)
    df.to_csv(csv_path, index=False)

    # Generate analytics
    heatmap_path = os.path.join(output_dir, "vehicle_heatmap.png")
    timeline_path = os.path.join(output_dir, "vehicle_count_timeline.png")

    generate_zone_vehicle_heatmap(csv_path, heatmap_path)
    generate_vehicle_count_timeline(csv_path, timeline_path)
