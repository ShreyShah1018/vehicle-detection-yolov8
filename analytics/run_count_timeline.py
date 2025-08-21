import os
from framewise_vehicle_count import generate_vehicle_count_timeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "output", "tracking_log.csv")
OUTPUT_IMG = os.path.join(BASE_DIR, "output", "vehicle_count_timeline.png")

generate_vehicle_count_timeline(CSV_PATH, OUTPUT_IMG)
