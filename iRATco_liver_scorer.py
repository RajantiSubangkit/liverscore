import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import morphology, segmentation, feature, measure, color
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="iRATco Liver Cell Boundary", layout="wide")
st.title("iRATco Liver Cell Boundary")
st.caption("Upload histopathology liver image to generate cell boundaries and watershed segmentation")

# =========================================================
# HELPERS
# =========================================================
def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))

def segment_hepatocytes(rgb, min_obj_size=80, peak_distance=12, blur_ksize=5):
    # grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # threshold inverse otsu
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = binary > 0

    # clean up
    mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_obj_size)
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(1))

    # distance transform
    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    # local maxima as seeds
    coords = feature.peak_local_max(
        dist,
        min_distance=peak_distance,
        labels=mask
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # if no peaks found, fallback to connected components
    if len(coords) == 0:
        labeled = measure.label(mask)
    else:
        labeled = segmentation.watershed(-dist, markers, mask=mask)

    # relabel
    labeled = measure.label(labeled > 0) if labeled.max() == 0 else measure.label(labeled)

    # boundaries
    boundaries = segmentation.find_boundaries(labeled, mode="outer")

    return gray, mask, dist, labeled, boundaries

def make_overlay(rgb, boundaries, boundary_color=(255, 0, 0)):
    overlay = rgb.copy()
    overlay[boundaries] = boundary_color
    return overlay

def make_watershed_color(labeled):
    if labeled.max() == 0:
        return np.zeros((labeled.shape[0], labeled.shape[1], 3), dtype=np.uint8)
    ws = color.label2rgb(labeled, bg_label=0)
    ws = (ws * 255).astype(np.uint8)
    return ws

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

    gray, mask, dist, labeled, boundaries = segment_hepatocytes(
        rgb,
        min_obj_size=min_obj_size,
        peak_distance=peak_distance,
        blur_ksize=blur_ksize
    )

    overlay = make_overlay(rgb, boundaries, boundary_color=(255, 0, 0))
    ws_color = make_watershed_color(labeled)

    n_objects = int(labeled.max())

    st.success(f"Detected segmented objects: {n_objects}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image")
        st.image(rgb, use_container_width=True)

    with col2:
        st.subheader("Binary mask")
        st.image((mask.astype(np.uint8) * 255), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Boundaries")
        boundary_img = np.zeros_like(rgb)
        boundary_img[boundaries] = [255, 255, 255]
        st.image(boundary_img, use_container_width=True)

    with col4:
        st.subheader("Overlay on original")
        st.image(overlay, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Watershed labels")
        st.image(ws_color, use_container_width=True)

    with col6:
        st.subheader("Distance transform")
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(dist, cmap="viridis")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Object size summary")

    props = measure.regionprops(labeled)
    if len(props) > 0:
        areas = [p.area for p in props]
        st.write(f"Number of objects: {len(areas)}")
        st.write(f"Mean area: {np.mean(areas):.2f} pixels")
        st.write(f"Min area: {np.min(areas):.2f} pixels")
        st.write(f"Max area: {np.max(areas):.2f} pixels")
    else:
        st.warning("No objects detected.")

else:
    st.info("Please upload a liver histopathology image first.")
