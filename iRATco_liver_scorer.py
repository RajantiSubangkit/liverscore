import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology, segmentation, feature, measure

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="iRATco Liver Cell Boundary", layout="wide")
st.title("iRATco Liver Cell Boundary")
st.caption("Upload histopathology liver image to generate hepatocyte boundaries and estimate vacuolization from bright area versus dense area in each segmented object")

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

def compute_bright_dense_metrics(rgb, labeled, bright_v_thresh, bright_s_thresh, dense_v_thresh, dense_s_thresh):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    props = measure.regionprops(labeled)
    records = []

    for prop in props:
        label_id = int(prop.label)
        area = int(prop.area)
        if area <= 0:
            continue

        minr, minc, maxr, maxc = prop.bbox
        obj_mask = (labeled[minr:maxr, minc:maxc] == label_id)

        patch_rgb = rgb[minr:maxr, minc:maxc]
        patch_hsv = hsv[minr:maxr, minc:maxc]
        patch_gray = gray[minr:maxr, minc:maxc]

        pix_rgb = patch_rgb[obj_mask]
        pix_hsv = patch_hsv[obj_mask]
        pix_gray = patch_gray[obj_mask]

        if len(pix_rgb) == 0:
            continue

        # Bright area: terang dan relatif kurang jenuh
        bright_mask = (
            (pix_hsv[:, 2] >= bright_v_thresh) &
            (pix_hsv[:, 1] <= bright_s_thresh)
        )

        # Dense area: lebih gelap / lebih padat warnanya
        dense_mask = (
            (pix_hsv[:, 2] <= dense_v_thresh) |
            (pix_hsv[:, 1] >= dense_s_thresh)
        )

        bright_pixels = int(np.sum(bright_mask))
        dense_pixels = int(np.sum(dense_mask))

        informative_pixels = bright_pixels + dense_pixels
        if informative_pixels == 0:
            vac_index = 0.0
        else:
            vac_index = 100.0 * bright_pixels / informative_pixels

        bright_fraction_percent = 100.0 * bright_pixels / len(pix_rgb)
        dense_fraction_percent = 100.0 * dense_pixels / len(pix_rgb)

        centroid_y, centroid_x = prop.centroid

        records.append({
            "label_id": label_id,
            "area_px": area,
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "total_pixels": int(len(pix_rgb)),
            "bright_pixels": bright_pixels,
            "dense_pixels": dense_pixels,
            "bright_fraction_percent": float(bright_fraction_percent),
            "dense_fraction_percent": float(dense_fraction_percent),
            "vacuolization_index_percent": float(vac_index),
            "mean_gray": float(np.mean(pix_gray)),
            "mean_r": float(np.mean(pix_rgb[:, 0])),
            "mean_g": float(np.mean(pix_rgb[:, 1])),
            "mean_b": float(np.mean(pix_rgb[:, 2]))
        })

    df = pd.DataFrame(records)
    mean_vac = 0.0 if df.empty else float(df["vacuolization_index_percent"].mean())
    return df, mean_vac

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Segmentation settings")
    min_obj_size = st.slider("Minimum object size", 20, 500, 80, 5)
    peak_distance = st.slider("Peak min distance", 3, 40, 12, 1)
    blur_ksize = st.slider("Gaussian blur kernel", 3, 11, 5, 2)

    st.markdown("---")
    st.header("Bright area settings")
    bright_v_thresh = st.slider("Bright threshold V", 120, 255, 185, 5)
    bright_s_thresh = st.slider("Bright threshold S", 0, 180, 95, 5)

    st.markdown("---")
    st.header("Dense area settings")
    dense_v_thresh = st.slider("Dense threshold V", 0, 180, 145, 5)
    dense_s_thresh = st.slider("Dense threshold S", 20, 255, 120, 5)

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

    analysis_df, mean_vacuolization = compute_bright_dense_metrics(
        rgb=rgb,
        labeled=labeled,
        bright_v_thresh=bright_v_thresh,
        bright_s_thresh=bright_s_thresh,
        dense_v_thresh=dense_v_thresh,
        dense_s_thresh=dense_s_thresh
    )

    st.success(f"Detected segmented objects: {n_objects}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Detected objects", f"{n_objects}")
    with m2:
        st.metric("Mean vacuolization total", f"{mean_vacuolization:.2f}%")
    with m3:
        mean_dense = 0.0 if analysis_df.empty else float(analysis_df["dense_fraction_percent"].mean())
        st.metric("Mean dense area", f"{mean_dense:.2f}%")

    st.image(
        overlay,
        caption="Overlay on original (yellow boundaries)",
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Per-object bright and dense area analysis")

    if analysis_df.empty:
        st.warning("No segmented objects detected.")
    else:
        st.dataframe(
            analysis_df[[
                "label_id",
                "area_px",
                "bright_pixels",
                "dense_pixels",
                "bright_fraction_percent",
                "dense_fraction_percent",
                "vacuolization_index_percent",
                "centroid_x",
                "centroid_y"
            ]],
            use_container_width=True
        )

        csv = analysis_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download vacuolization CSV",
            data=csv,
            file_name="liver_bright_dense_vacuolization.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a liver histopathology image first.")
