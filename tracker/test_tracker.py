import cv2
import os
from inference_tracker import YOLODeepSORTTracker
import pandas as pd


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, 'videos', 'sample_video.mp4')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'tracked_video_2.mp4')

# Create tracker
tracker = YOLODeepSORTTracker()

# Video IO
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_data = []
frame_no = 0


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Frame-by-frame tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tracked_frame, tracks = tracker.track_video_frame(frame)

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

    out.write(tracked_frame)

    cv2.imshow("Tracking Output", tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

df = pd.DataFrame(frame_data)
log_path = os.path.join(BASE_DIR, "output", "tracking_log.csv")
df.to_csv(log_path, index=False)
print(f"✔️ Tracking log saved to {log_path}")

out.release()
cv2.destroyAllWindows()
print(f"✔️ Video saved to {OUTPUT_PATH}")
