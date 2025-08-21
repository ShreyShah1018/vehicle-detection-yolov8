import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_zone_vehicle_heatmap(csv_path, output_path,
                                   frame_width=1280, frame_height=720,
                                   grid_rows=6, grid_cols=6):
    df = pd.read_csv(csv_path)

    # Compute center of each bounding box
    df["center_x"] = (df["x1"] + df["x2"]) / 2
    df["center_y"] = (df["y1"] + df["y2"]) / 2

    # Assign zone based on position
    zone_width = frame_width / grid_cols
    zone_height = frame_height / grid_rows

    df["zone_x"] = (df["center_x"] // zone_width).astype(int)
    df["zone_y"] = (df["center_y"] // zone_height).astype(int)

    # Clamp zone indices to avoid overflow
    df["zone_x"] = df["zone_x"].clip(0, grid_cols - 1)
    df["zone_y"] = df["zone_y"].clip(0, grid_rows - 1)

    # Count vehicle appearances in each zone
    heatmap_data = np.zeros((grid_rows, grid_cols), dtype=int)
    for _, row in df.iterrows():
        heatmap_data[int(row["zone_y"]), int(row["zone_x"])] += 1

    # Plot zone-based heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="YlOrRd", cbar=True,
                xticklabels=[f"Z{c}" for c in range(grid_cols)],
                yticklabels=[f"Z{r}" for r in range(grid_rows)])

    plt.title("🧭 Zone-Based Vehicle Count Heatmap")
    plt.xlabel("Horizontal Zones")
    plt.ylabel("Vertical Zones")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Zone-based heatmap saved to: {output_path}")
