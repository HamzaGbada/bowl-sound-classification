"""Streamlit app for bowel sound detection and classification."""
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import soundfile as sf
from pathlib import Path
from io import BytesIO

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import inference engine
import sys
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from inference import BowelSoundDetector

CLASS_COLORS = {"b": "rgba(76, 114, 176, 0.3)", "mb": "rgba(221, 132, 82, 0.3)",
                "h": "rgba(85, 168, 104, 0.3)"}
CLASS_COLORS_SOLID = {"b": "#4C72B0", "mb": "#DD8452", "h": "#55A868"}
CLASS_LABELS = {"b": "Single Burst", "mb": "Multiple Burst", "h": "Harmonic"}


@st.cache_resource
def load_detector():
    config_path = PROJECT_ROOT / "results" / "best_model_config.json"
    return BowelSoundDetector(config_path=str(config_path))


def main():
    st.set_page_config(page_title="Bowel Sound Detector", layout="wide")

    # --- SECTION 1: Header ---
    st.title("Bowel Sound Detector")
    st.markdown("*Automatic detection and classification of bowel sounds in abdominal "
                "audio recordings*")

    detector = load_detector()
    config = detector.config
    st.markdown(
        f"**Model:** `{config['model']}` &nbsp; | &nbsp; "
        f"**Macro-F1:** `{config['macro_f1']:.4f}` &nbsp; | &nbsp; "
        f"**Preprocessing:** bandpass={config['preprocessing_config'].get('use_bandpass', False)}, "
        f"normalise={config['preprocessing_config'].get('use_normalise', False)}"
    )
    st.divider()

    # --- SECTION 2: Upload ---
    uploaded = st.file_uploader("Upload a .wav audio file", type=["wav"])
    if uploaded is None:
        st.info("Upload a .wav file to get started.")
        return

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    info = sf.info(tmp_path)
    st.markdown(f"**File:** `{uploaded.name}` &nbsp; | &nbsp; "
                f"**Duration:** {info.duration:.2f}s &nbsp; | &nbsp; "
                f"**Sample rate:** {info.samplerate} Hz &nbsp; | &nbsp; "
                f"**Channels:** {info.channels}")

    # --- SECTION 3: Run detection ---
    if st.button("Detect bowel sounds", type="primary"):
        with st.spinner("Running detection..."):
            detections = detector.detect(tmp_path)
        st.session_state["detections"] = detections
        st.session_state["tmp_path"] = tmp_path
        st.session_state["file_name"] = uploaded.name

    if "detections" not in st.session_state:
        return

    detections = st.session_state["detections"]
    tmp_path = st.session_state["tmp_path"]

    if len(detections) == 0:
        st.warning("No bowel sounds detected above confidence threshold.")
        return

    n_b = len(detections[detections["class_label"] == "b"])
    n_mb = len(detections[detections["class_label"] == "mb"])
    n_h = len(detections[detections["class_label"] == "h"])
    st.success(f"Found **{len(detections)}** events "
               f"({n_b} single bursts, {n_mb} multiple bursts, {n_h} harmonics)")

    # --- SECTION 4: Waveform visualisation ---
    st.subheader("Waveform with Detected Events")
    import librosa
    y, _ = librosa.load(tmp_path, sr=22050, mono=True)
    t = np.linspace(0, len(y) / 22050, len(y))

    # Downsample waveform for plotting (every 10th point)
    step = max(1, len(y) // 50000)
    t_ds = t[::step]
    y_ds = y[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_ds, y=y_ds, mode="lines",
                             line=dict(color="grey", width=0.5),
                             name="Waveform", showlegend=True))

    # Add shaded regions for detections
    legend_added = set()
    for _, row in detections.iterrows():
        cls = row["class_label"]
        show_legend = cls not in legend_added
        legend_added.add(cls)
        fig.add_vrect(
            x0=row["start_s"], x1=row["end_s"],
            fillcolor=CLASS_COLORS[cls],
            line_width=0,
            annotation_text=cls if (row["end_s"] - row["start_s"]) > 0.5 else "",
            annotation_position="top left",
        )
        # Invisible trace for legend
        if show_legend:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=CLASS_COLORS_SOLID[cls]),
                name=f"{cls} ({CLASS_LABELS[cls]})"
            ))

    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        height=400, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- SECTION 5: Detection table ---
    st.subheader("Detection Table")
    display_df = detections.copy()
    display_df["Duration (s)"] = (display_df["end_s"] - display_df["start_s"]).round(3)
    display_df = display_df.rename(columns={
        "start_s": "Start (s)", "end_s": "End (s)",
        "class_label": "Class", "confidence": "Confidence"
    })
    display_df = display_df[["Start (s)", "End (s)", "Duration (s)", "Class", "Confidence"]]

    def color_class(val):
        colors = {"b": "color: #4C72B0; font-weight: bold",
                  "mb": "color: #DD8452; font-weight: bold",
                  "h": "color: #55A868; font-weight: bold"}
        return colors.get(val, "")

    styled = display_df.style.map(color_class, subset=["Class"])
    st.dataframe(styled, use_container_width=True, height=min(400, 40 + 35 * len(display_df)))

    # Totals
    total_dur = display_df["Duration (s)"].sum()
    st.markdown(f"**Totals:** {len(display_df)} events, {total_dur:.2f}s total duration")

    # --- SECTION 6: Export ---
    st.subheader("Export")
    col1, col2 = st.columns(2)

    with col1:
        csv_data = detections.to_csv(index=False)
        st.download_button("Download detections as CSV", csv_data,
                           file_name="detections.csv", mime="text/csv")

    with col2:
        report = {
            "file": st.session_state.get("file_name", "unknown"),
            "model": config["model"],
            "macro_f1": config["macro_f1"],
            "preprocessing": config["preprocessing_config"],
            "n_events": len(detections),
            "detections": detections.to_dict(orient="records"),
        }
        json_data = json.dumps(report, indent=2)
        st.download_button("Download report as JSON", json_data,
                           file_name="report.json", mime="application/json")


if __name__ == "__main__":
    main()