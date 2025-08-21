import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_vehicle_count_timeline(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Count unique vehicle IDs per frame
    frame_counts = df.groupby("frame")["track_id"].nunique().reset_index()
    frame_counts.columns = ["frame", "vehicle_count"]

    # Plot timeline
    plt.figure(figsize=(10, 6))
    plt.plot(frame_counts["frame"], frame_counts["vehicle_count"], color="blue", linewidth=2)
    plt.title("Vehicle Count Per Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Active Vehicles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Vehicle count timeline saved to: {output_path}")
