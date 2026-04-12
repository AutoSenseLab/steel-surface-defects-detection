import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.predict import load_model, preprocess_image, CLASS_NAMES

st.set_page_config(
    page_title="AI Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .defect  { background-color: #ffcccc; color: #cc0000; }
    .good    { background-color: #ccffcc; color: #006600; }
    .metric-label { font-size: 14px; color: gray; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception:
        return None

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.99, 0.7, 0.05)
    show_heatmap   = st.checkbox("Show GradCAM Heatmap", value=True)
    show_all_probs = st.checkbox("Show all class probabilities", value=True)
    heatmap_alpha  = st.slider("Heatmap Intensity", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**Dataset:** NEU Steel Surface Defect")
    st.markdown("**Model:** MobileNetV2 + GradCAM")
    st.markdown("**by:** Heart Khunpanuk")

st.title("🔍 AI Defect Detection")
st.markdown("Steel surface inspection — 6 defect classes")
st.markdown("---")

model = get_model()

if model is None:
    st.warning("No model found. Run `python src/train.py` first.")
    st.code("python src/train.py", language="bash")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_array, _ = preprocess_image(image)

    with st.spinner("Analyzing..."):
        probs    = model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = CLASS_NAMES[pred_idx]
        conf     = float(probs[pred_idx])

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
        st.caption(f"{image.size[0]}×{image.size[1]} px")

    with col2:
        st.subheader("Result")
        st.markdown(
            f'<div class="result-box defect">⚠️ {pred_cls}</div>',
            unsafe_allow_html=True
        )
        st.metric("Confidence", f"{conf*100:.1f}%")
        st.progress(conf)

        if show_all_probs:
            st.markdown("**All classes:**")
            for cls, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
                col_a, col_b = st.columns([3, 1])
                col_a.progress(float(p), text=cls)
                col_b.write(f"`{p*100:.1f}%`")

    with col3:
        if show_heatmap:
            st.subheader("GradCAM")
            with st.spinner("Generating heatmap..."):
                try:
                    heatmap = make_gradcam_heatmap(img_array, model)
                    overlaid, heatmap_only = overlay_heatmap(
                        np.array(image), heatmap, alpha=heatmap_alpha
                    )
                    st.image(overlaid, use_column_width=True)
                    st.caption("Red = where the model focused")
                    with st.expander("Heatmap only"):
                        st.image(heatmap_only, use_column_width=True)
                except Exception as e:
                    st.error(f"Heatmap failed: {e}")

    st.markdown("---")
    st.info(f"Model: EfficientNetB0  |  Dataset: NEU Steel  |  Predicted: **{pred_cls}** ({conf*100:.1f}%)")

else:
    st.markdown("""
    ### Upload an image to get started

    | # | Defect | Description |
    |---|---|---|
    | 1 | Crazing | fine cracks |
    | 2 | Inclusion | embedded foreign material |
    | 3 | Patches | blotchy surface |
    | 4 | Pitted Surface | small pits |
    | 5 | Rolled-in Scale | scale pressed in during rolling |
    | 6 | Scratches | scratch marks |
    """)
