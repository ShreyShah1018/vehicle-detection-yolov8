import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analytics.generate_zone_heatmap import generate_zone_vehicle_heatmap

# Set paths
csv_path = os.path.join("output", "tracking_log.csv")
output_path = os.path.join("output", "zone_vehicle_heatmap.png")

# Set frame size (match your video resolution)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

generate_zone_vehicle_heatmap(
    csv_path=csv_path,
    output_path=output_path,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    grid_rows=6,
    grid_cols=6
)
