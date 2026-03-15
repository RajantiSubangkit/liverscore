import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import morphology, segmentation, feature, measure

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="iRATco Liver Cell Boundary", layout="wide")
st.title("iRATco Liver Cell Boundary")
st.caption("Upload histopathology liver image to generate hepatocyte boundaries with yellow overlay")

# =========================================================
# HELPERS
# =========================================================
def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))

def segment_hepatocytes(rgb, min_obj_size=80, peak_distance=12, blur_ksize=5):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = binary > 0
    mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_obj_size)
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(1))

    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    coords = feature.peak_local_max(
        dist,
        min_distance=peak_distance,
        labels=mask
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if len(coords) == 0:
        labeled = measure.label(mask)
    else:
        labeled = segmentation.watershed(-dist, markers, mask=mask)

    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    return labeled, boundaries

def make_overlay(rgb, boundaries, boundary_color=(255, 255, 0)):
    overlay = rgb.copy()
    overlay[boundaries] = boundary_color
    return overlay

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Segmentation settings")
    min_obj_size = st.slider("Minimum object size", 20, 500, 80, 5)
    peak_distance = st.slider("Peak min distance", 3, 40, 12, 1)
    blur_ksize = st.slider("Gaussian blur kernel", 3, 11, 5, 2)

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload liver histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    rgb = pil_to_rgb_array(Image.open(uploaded))

    labeled, boundaries = segment_hepatocytes(
        rgb,
        min_obj_size=min_obj_size,
        peak_distance=peak_distance,
        blur_ksize=blur_ksize
    )

    overlay = make_overlay(rgb, boundaries, boundary_color=(255, 255, 0))
    n_objects = int(labeled.max())

    st.success(f"Detected segmented objects: {n_objects}")
    st.image(overlay, caption="Overlay on original (yellow boundaries)", use_container_width=True)

else:
    st.info("Please upload a liver histopathology image first.")
