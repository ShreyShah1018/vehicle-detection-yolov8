
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

def generate_vehicle_heatmap(csv_path, output_path, background_path=None):
    df = pd.read_csv(csv_path)

    # Compute center of bounding boxes
    df["center_x"] = (df["x1"] + df["x2"]) / 2
    df["center_y"] = (df["y1"] + df["y2"]) / 2

    # Compute track longevity (frames each ID appeared)
    track_duration = df.groupby("track_id")["frame"].nunique().to_dict()
    df["duration"] = df["track_id"].map(track_duration)

    # Normalize weights
    df["weight"] = df["duration"] / df["duration"].max()

    # Start plotting
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # ✅ Optionally overlay a background image (first frame)
    if background_path and os.path.exists(background_path):
        img = cv2.imread(background_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, alpha=0.7)  # slightly transparent

    # ✅ Plot the heatmap
    heatmap = sns.kdeplot(
        x=df["center_x"],
        y=df["center_y"],
        weights=df["weight"],
        cmap="coolwarm",
        fill=True,
        bw_adjust=0.5,
        thresh=0.05,
        alpha=0.6,
        ax=ax
    )

    # ✅ Plot decoration
    ax.invert_yaxis()
    ax.set_title("🚗 Vehicle Density Heatmap (Time Weighted)", fontsize=14, pad=15)
    ax.set_xlabel("Frame Width")
    ax.set_ylabel("Frame Height")

    # ✅ Add color bar
    mappable = heatmap.get_children()[0]
    plt.colorbar(mappable, ax=ax, label="Density (Weighted by Track Duration)")

    # ✅ Save in high resolution
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Heatmap saved at: {output_path}")
