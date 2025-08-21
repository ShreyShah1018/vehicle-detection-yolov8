import streamlit as st
from PIL import Image
import os

def show_heatmap(heatmap_path):
    if os.path.exists(heatmap_path):
        img = Image.open(heatmap_path)
        st.image(img, caption="Frame-Weighted Vehicle Heatmap", use_column_width=True)
    else:
        st.warning("⚠️ Heatmap not found.")

def show_count_timeline(timeline_path):
    if os.path.exists(timeline_path):
        img = Image.open(timeline_path)
        st.image(img, caption="Vehicle Count Over Time", use_column_width=True)
    else:
        st.warning("⚠️ Count timeline image not found.")
