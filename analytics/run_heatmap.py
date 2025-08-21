import os
from heatmap_generator import generate_vehicle_heatmap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "output", "tracking_log.csv")
OUTPUT_IMG = os.path.join(BASE_DIR, "output", "vehicle_heatmap.png")

generate_vehicle_heatmap(CSV_PATH, OUTPUT_IMG)
