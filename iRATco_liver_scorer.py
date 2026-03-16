import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology, segmentation, feature, measure

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="iRATco Liver Object Vacuolization", layout="wide")
st.title("iRATco Liver Object Vacuolization")
st.caption("Preview one segmented hepatocyte object and estimate cytoplasmic density and vacuolization")

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

    labeled = measure.label(labeled > 0) if labeled.max() == 0 else labeled
    return labeled

def get_object_rows(labeled):
    props = measure.regionprops(labeled)
    rows = []
    for prop in props:
        if prop.area < 30:
            continue
        minr, minc, maxr, maxc = prop.bbox
        cy, cx = prop.centroid
        rows.append({
            "label_id": int(prop.label),
            "area": int(prop.area),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "centroid_x": float(cx),
            "centroid_y": float(cy)
        })
    return pd.DataFrame(rows)

def crop_single_object(rgb, labeled, row, pad=8):
    minr = max(0, int(row["bbox_minr"]) - pad)
    minc = max(0, int(row["bbox_minc"]) - pad)
    maxr = min(rgb.shape[0], int(row["bbox_maxr"]) + pad)
    maxc = min(rgb.shape[1], int(row["bbox_maxc"]) + pad)

    crop_rgb = rgb[minr:maxr, minc:maxc].copy()
    crop_mask = (labeled[minr:maxr, minc:maxc] == int(row["label_id"]))

    return crop_rgb, crop_mask

def analyze_object_density_and_vacuolization(crop_rgb, crop_mask, sat_thresh, val_thresh, red_ratio_thresh):
    """
    Dense cytoplasm = piksel dalam object mask yang cukup berwarna eosin / cukup padat:
    - saturation >= sat_thresh
    - value <= val_thresh   (tidak terlalu pucat)
    - red channel relatif dominan terhadap green/blue

    Vacuolization = area object yang tidak termasuk dense threshold
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    pix_hsv = hsv[crop_mask]
    pix_rgb = crop_rgb[crop_mask]

    if len(pix_rgb) == 0:
        return None

    r = pix_rgb[:, 0].astype(np.float32)
    g = pix_rgb[:, 1].astype(np.float32)
    b = pix_rgb[:, 2].astype(np.float32)

    dense_mask_flat = (
        (pix_hsv[:, 1] >= sat_thresh) &
        (pix_hsv[:, 2] <= val_thresh) &
        (r >= g * red_ratio_thresh) &
        (r >= b * red_ratio_thresh)
    )

    total_area = int(np.sum(crop_mask))
    dense_area = int(np.sum(dense_mask_flat))
    vacuolated_area = int(total_area - dense_area)

    dense_percent = 100.0 * dense_area / total_area if total_area > 0 else 0.0
    vac_percent = 100.0 * vacuolated_area / total_area if total_area > 0 else 0.0

    # rebuild dense mask to image shape
    dense_mask_img = np.zeros(crop_mask.shape, dtype=bool)
    dense_mask_img[crop_mask] = dense_mask_flat

    # white bg version
    object_only = np.ones_like(crop_rgb, dtype=np.uint8) * 255
    object_only[crop_mask] = crop_rgb[crop_mask]

    overlay = object_only.copy()
    overlay[dense_mask_img] = [255, 0, 0]  # merah = dense cytoplasm

    # outline object
    boundary = segmentation.find_boundaries(crop_mask, mode="outer")
    overlay[boundary] = [255, 255, 0]
    object_only[boundary] = [255, 255, 0]

    return {
        "object_only": object_only,
        "overlay": overlay,
        "dense_mask_img": dense_mask_img,
        "total_area": total_area,
        "dense_area": dense_area,
        "vacuolated_area": vacuolated_area,
        "dense_percent": dense_percent,
        "vacuolization_percent": vac_percent
    }

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Segmentation settings")
    min_obj_size = st.slider("Minimum object size", 20, 500, 80, 5)
    peak_distance = st.slider("Peak min distance", 3, 40, 12, 1)
    blur_ksize = st.slider("Gaussian blur kernel", 3, 11, 5, 2)

    st.markdown("---")
    st.header("Dense cytoplasm threshold")
    sat_thresh = st.slider("Minimum saturation (S)", 20, 255, 90, 5)
    val_thresh = st.slider("Maximum brightness (V)", 80, 255, 210, 5)
    red_ratio_thresh = st.slider("Red dominance ratio", 1.00, 1.50, 1.08, 0.01)

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload liver histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    rgb = pil_to_rgb_array(Image.open(uploaded))
    labeled = segment_hepatocytes(
        rgb,
        min_obj_size=min_obj_size,
        peak_distance=peak_distance,
        blur_ksize=blur_ksize
    )

    objects_df = get_object_rows(labeled)

    if objects_df.empty:
        st.warning("No objects detected.")
    else:
        st.success(f"Detected segmented objects: {len(objects_df)}")

        object_idx = st.slider(
            "Select detected object",
            min_value=0,
            max_value=len(objects_df) - 1,
            value=0,
            step=1
        )

        row = objects_df.iloc[object_idx]
        crop_rgb, crop_mask = crop_single_object(rgb, labeled, row, pad=10)

        result = analyze_object_density_and_vacuolization(
            crop_rgb=crop_rgb,
            crop_mask=crop_mask,
            sat_thresh=sat_thresh,
            val_thresh=val_thresh,
            red_ratio_thresh=red_ratio_thresh
        )

        if result is None:
            st.warning("Selected object could not be analyzed.")
        else:
            st.markdown("### Selected object analysis")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Label ID", int(row["label_id"]))
            with m2:
                st.metric("Total cytoplasm area", int(result["total_area"]))
            with m3:
                st.metric("Dense cytoplasm", f"{result['dense_percent']:.2f}%")
            with m4:
                st.metric("Vacuolization", f"{result['vacuolization_percent']:.2f}%")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Single detected object")
                st.image(result["object_only"], use_container_width=True)

            with col2:
                st.subheader("Threshold overlay")
                st.image(
                    result["overlay"],
                    caption="Red = dense cytoplasm, Yellow = object boundary",
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("Object data")

            out_df = pd.DataFrame([{
                "label_id": int(row["label_id"]),
                "total_cytoplasm_area": int(result["total_area"]),
                "dense_area": int(result["dense_area"]),
                "vacuolated_area": int(result["vacuolated_area"]),
                "dense_percent": float(result["dense_percent"]),
                "vacuolization_percent": float(result["vacuolization_percent"])
            }])

            st.dataframe(out_df, use_container_width=True)

else:
    st.info("Please upload a liver histopathology image first.")
