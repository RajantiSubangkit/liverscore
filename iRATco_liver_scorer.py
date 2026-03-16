import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology, segmentation, feature, measure

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="iRATco Liver Vacuolization", layout="wide")
st.title("iRATco Liver Vacuolization")
st.caption("Segment hepatocyte-like objects and estimate vacuolization from grayscale dark-versus-light cytoplasmic threshold")

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

def make_gray_threshold_preview(rgb, gray_threshold):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # gelap = dense
    dense_mask = gray <= gray_threshold

    preview = np.stack([gray, gray, gray], axis=-1)
    preview = preview.copy()
    preview[dense_mask] = [255, 0, 0]  # merah = dense / gelap
    return gray, dense_mask, preview

def compute_object_vacuolization_from_gray(rgb, labeled, gray_threshold):
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

        patch_gray = gray[minr:maxr, minc:maxc]
        pix_gray = patch_gray[obj_mask]

        if len(pix_gray) == 0:
            continue

        dense_mask = pix_gray <= gray_threshold

        total_area = int(len(pix_gray))
        dense_area = int(np.sum(dense_mask))
        vacuolated_area = int(total_area - dense_area)

        dense_percent = 100.0 * dense_area / total_area if total_area > 0 else 0.0
        vac_percent = 100.0 * vacuolated_area / total_area if total_area > 0 else 0.0

        centroid_y, centroid_x = prop.centroid

        records.append({
            "label_id": label_id,
            "area_px": area,
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "mean_gray": float(np.mean(pix_gray)),
            "dense_area_px": dense_area,
            "non_dense_area_px": vacuolated_area,
            "dense_percent": float(dense_percent),
            "vacuolization_percent": float(vac_percent)
        })

    df = pd.DataFrame(records)
    mean_vacuolization = 0.0 if df.empty else float(df["vacuolization_percent"].mean())
    mean_dense = 0.0 if df.empty else float(df["dense_percent"].mean())

    return df, mean_vacuolization, mean_dense

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Segmentation settings")
    min_obj_size = st.slider("Minimum object size", 20, 500, 80, 5)
    peak_distance = st.slider("Peak min distance", 3, 40, 12, 1)
    blur_ksize = st.slider("Gaussian blur kernel", 3, 11, 5, 2)

    st.markdown("---")
    st.header("Grayscale density threshold")
    gray_threshold = st.slider(
        "Dense threshold (gray intensity)",
        min_value=0,
        max_value=255,
        value=155,
        step=1
    )
    st.caption("Pixels darker than or equal to this threshold are considered dense cytoplasm.")

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
    gray_img, dense_mask_full, threshold_preview = make_gray_threshold_preview(
        rgb,
        gray_threshold=gray_threshold
    )

    analysis_df, mean_vacuolization, mean_dense = compute_object_vacuolization_from_gray(
        rgb=rgb,
        labeled=labeled,
        gray_threshold=gray_threshold
    )

    n_objects = int(labeled.max())

    st.success(f"Detected segmented objects: {n_objects}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Detected objects", f"{n_objects}")
    with m2:
        st.metric("Mean dense cytoplasm", f"{mean_dense:.2f}%")
    with m3:
        st.metric("Mean vacuolization total", f"{mean_vacuolization:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segmented image")
        st.image(
            overlay,
            caption="Original image with yellow cell boundaries",
            use_container_width=True
        )

    with col2:
        st.subheader("Threshold preview")
        st.image(
            threshold_preview,
            caption="Grayscale preview. Red = dense/dark area based on threshold",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Per-object vacuolization analysis")

    if analysis_df.empty:
        st.warning("No segmented objects detected.")
    else:
        st.dataframe(
            analysis_df[[
                "label_id",
                "area_px",
                "mean_gray",
                "dense_area_px",
                "non_dense_area_px",
                "dense_percent",
                "vacuolization_percent",
                "centroid_x",
                "centroid_y"
            ]],
            use_container_width=True
        )

        csv = analysis_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download vacuolization CSV",
            data=csv,
            file_name="liver_gray_threshold_vacuolization.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a liver histopathology image first.")
