"""Streamlit interface for comparing two images of the same place.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.change_detection import absolute_difference, change_mask, changed_fraction
from src.image_io import ImageValidationError, load_rgb_image, validate_same_shape
from src.visualization import difference_heatmap, mask_overlay

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"

st.set_page_config(page_title="Satellite Change Detector", layout="wide")
st.title("Satellite Change Detector")
st.caption("Phase 1: pixel-wise RGB differencing")

with st.sidebar:
    st.header("Input")
    source = st.radio("Images", ["Synthetic sample pair", "Upload my own"])
    if source == "Upload my own":
        before_file = st.file_uploader("Before image", type=["png", "jpg", "jpeg"])
        after_file = st.file_uploader("After image", type=["png", "jpg", "jpeg"])
    else:
        before_file = SAMPLE_DIR / "before.png"
        after_file = SAMPLE_DIR / "after.png"
        if not before_file.exists():
            st.error("Sample images not found. Run `python scripts/make_sample_data.py` first.")
            st.stop()

    st.header("Detection")
    threshold = st.slider(
        "Change threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="A pixel is flagged when its average color difference is above this value.",
    )

if before_file is None or after_file is None:
    st.info("Upload a 'before' and an 'after' image of the same area to get started.")
    st.stop()

try:
    before = load_rgb_image(before_file)
    after = load_rgb_image(after_file)
    validate_same_shape(before, after)
except ImageValidationError as exc:
    st.error(str(exc))
    st.stop()

difference = absolute_difference(before, after)
mask = change_mask(difference, threshold)
fraction = changed_fraction(mask)

col1, col2 = st.columns(2)
col1.image(before, caption="Before", width="stretch")
col2.image(after, caption="After", width="stretch")

col3, col4 = st.columns(2)
col3.image(
    difference_heatmap(difference),
    caption="Difference heatmap (dark = similar, bright = different)",
    width="stretch",
)
col4.image(
    mask_overlay(after, mask),
    caption=f"Pixels above threshold {threshold:.2f} (magenta)",
    width="stretch",
)

st.subheader("Statistics")
m1, m2, m3 = st.columns(3)
m1.metric("Flagged pixels", f"{fraction:.2%}")
m2.metric("Flagged / total", f"{int(mask.sum()):,} / {mask.size:,}")
m3.metric("Image size", f"{before.shape[1]} x {before.shape[0]} px")
st.caption(
    "These are pixels the algorithm *flagged* as different. That is a measurement of "
    "the images, not proof that anything changed on the ground. No area in m² is shown "
    "because plain PNG/JPEG files carry no information about pixel size."
)

with st.expander("Why does simple differencing give false positives?"):
    st.markdown(
        """
Image differencing flags **any** change in pixel color, not just real changes on the ground:

- **Lighting & atmosphere:** sun angle, haze and thin clouds change brightness everywhere.
  Try a low threshold on the sample pair: the whole image lights up because the
  "after" image is about 8% brighter.
- **Seasons:** a field looks green in June and brown in October without anything being
  cleared.
- **Clouds and shadows:** they show up as huge "changes" that are really just weather.
- **Misalignment:** if the images are shifted by even one pixel, every edge (roads,
  rivers, field boundaries) looks like it changed.
- **Different sensors or processing:** two cameras record the same color slightly
  differently.
- **Noise:** every sensor adds random noise, so the difference is never exactly zero.

The threshold trades off **false positives** (flagging things that didn't really change)
against **false negatives** (missing real changes). Later phases use spectral indices
like NDVI, which react to vegetation specifically and less to overall brightness.
"""
    )
