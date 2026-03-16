import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology, segmentation, feature, measure
import matplotlib.pyplot as plt

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

def make_gray_threshold_preview(rgb, labeled, gray_threshold):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dense_mask = gray <= gray_threshold

    preview = np.stack([gray, gray, gray], axis=-1).copy()

    object_mask = labeled > 0
    dense_object_mask = dense_mask & object_mask

    preview[dense_object_mask] = [255, 0, 0]
    preview[~object_mask] = [255, 255, 255]

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
        mean_gray = float(np.mean(pix_gray))

        centroid_y, centroid_x = prop.centroid

        records.append({
            "label_id": label_id,
            "area_px": area,
            "mean_gray": mean_gray,
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "dense_area_px": dense_area,
            "non_dense_area_px": vacuolated_area,
            "dense_percent": float(dense_percent),
            "vacuolization_percent": float(vac_percent)
        })

    df = pd.DataFrame(records)

    mean_vacuolization = 0.0 if df.empty else float(df["vacuolization_percent"].mean())
    mean_intensity = 0.0 if df.empty else float(df["mean_gray"].mean())

    return df, mean_vacuolization, mean_intensity

def crop_single_segmented_object(rgb, labeled, row, gray_threshold, pad=10):
    minr = max(0, int(row["bbox_minr"]) - pad)
    minc = max(0, int(row["bbox_minc"]) - pad)
    maxr = min(rgb.shape[0], int(row["bbox_maxr"]) + pad)
    maxc = min(rgb.shape[1], int(row["bbox_maxc"]) + pad)

    crop_rgb = rgb[minr:maxr, minc:maxc].copy()
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_obj_mask = (labeled[minr:maxr, minc:maxc] == int(row["label_id"]))

    object_preview = np.ones_like(crop_rgb, dtype=np.uint8) * 255
    object_preview[crop_obj_mask] = crop_rgb[crop_obj_mask]

    threshold_preview = np.stack([crop_gray, crop_gray, crop_gray], axis=-1)
    threshold_preview[~crop_obj_mask] = [255, 255, 255]

    dense_mask = (crop_gray <= gray_threshold) & crop_obj_mask
    threshold_preview[dense_mask] = [255, 0, 0]

    boundary = segmentation.find_boundaries(crop_obj_mask, mode="outer")
    object_preview[boundary] = [255, 255, 0]
    threshold_preview[boundary] = [255, 255, 0]

    return object_preview, threshold_preview

def make_density_plot(values):
    fig, ax = plt.subplots(figsize=(7, 4))

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.hist(values, bins=25, density=True, alpha=0.35)

    if len(values) > 1 and np.std(values) > 0:
        xs = np.linspace(0, 100, 400)
        bw = max(np.std(values) * 0.25, 1.0)
        density = np.zeros_like(xs)

        for v in values:
            density += np.exp(-0.5 * ((xs - v) / bw) ** 2)

        density /= (len(values) * bw * np.sqrt(2 * np.pi))
        ax.plot(xs, density, linewidth=2)

    ax.set_title("Density plot of vacuolization (%)")
    ax.set_xlabel("Vacuolization (%)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 100)
    return fig

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
        labeled,
        gray_threshold=gray_threshold
    )

    analysis_df, mean_vacuolization, mean_intensity = compute_object_vacuolization_from_gray(
        rgb=rgb,
        labeled=labeled,
        gray_threshold=gray_threshold
    )

    n_objects = int(labeled.max())

    st.success(f"Detected segmented objects: {n_objects}")

    # =====================================================
    # THREE IMAGES IN ONE ROW
    # =====================================================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Raw file")
        st.image(
            rgb,
            caption="Original image",
            use_container_width=True
        )

    with c2:
        st.subheader("Segmented image")
        st.image(
            overlay,
            caption="Yellow boundaries",
            use_container_width=True
        )

    with c3:
        st.subheader("Threshold preview")
        st.image(
            threshold_preview,
            caption="Red = dense/dark area",
            use_container_width=True
        )

    # =====================================================
    # SINGLE OBJECT PREVIEW
    # =====================================================
    if not analysis_df.empty:
        st.markdown("---")
        st.subheader("Single segmented object for threshold tuning")

        object_index = st.slider(
            "Preview object index",
            min_value=0,
            max_value=len(analysis_df) - 1,
            value=0,
            step=1
        )

        selected_row = analysis_df.iloc[object_index]
        obj_img, obj_thresh = crop_single_segmented_object(
            rgb=rgb,
            labeled=labeled,
            row=selected_row,
            gray_threshold=gray_threshold,
            pad=10
        )

        s1, s2 = st.columns(2)
        with s1:
            st.image(
                obj_img,
                caption=f"Segmented object ID {int(selected_row['label_id'])}",
                use_container_width=True
            )
        with s2:
            st.image(
                obj_thresh,
                caption="Threshold preview for selected object",
                use_container_width=True
            )

        st.caption(
            f"Object ID {int(selected_row['label_id'])} | "
            f"Mean intensity = {selected_row['mean_gray']:.2f} gray units | "
            f"Vacuolization = {selected_row['vacuolization_percent']:.2f}%"
        )

    # =====================================================
    # DENSITY PLOT
    # =====================================================
    st.markdown("---")
    st.subheader("Density plot")

    if analysis_df.empty:
        st.warning("No segmented objects detected.")
    else:
        fig = make_density_plot(analysis_df["vacuolization_percent"].values)
        st.pyplot(fig)
        plt.close(fig)

    # =====================================================
    # ANALYSIS
    # =====================================================
    st.markdown("---")
    st.subheader("Analysis")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Detected objects", f"{n_objects}")
    with m2:
        st.metric("Mean intensity", f"{mean_intensity:.2f}")
    with m3:
        st.metric("Mean vacuolization total", f"{mean_vacuolization:.2f}%")

    csv = analysis_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download vacuolization CSV",
        data=csv,
        file_name="liver_gray_threshold_vacuolization.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a liver histopathology image first.")
