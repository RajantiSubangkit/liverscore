from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import measure, morphology, segmentation, feature, color
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="iRATco-Liver Histo Scorer",
    layout="wide"
)

st.title("iRATco-Liver Histo Scorer")
st.caption(
    "Semi-automatic liver histopathology scoring: nucleus, cytoplasm, and inflammation"
)

# =========================================================
# SESSION STATE
# =========================================================
TRAIN_CLASSES = [
    "Nucleus_Normal",
    "Nucleus_Pyknotic",
    "Cytoplasm",
    "Inflammation",
    "Exclude"
]

if "samples" not in st.session_state:
    st.session_state.samples = {cls: [] for cls in TRAIN_CLASSES}

if "objects_df" not in st.session_state:
    st.session_state.objects_df = None

if "labeled_mask" not in st.session_state:
    st.session_state.labeled_mask = None

if "preview_rgb" not in st.session_state:
    st.session_state.preview_rgb = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "trained" not in st.session_state:
    st.session_state.trained = False

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

# =========================================================
# HELPERS
# =========================================================
def reset_all():
    st.session_state.samples = {cls: [] for cls in TRAIN_CLASSES}
    st.session_state.objects_df = None
    st.session_state.labeled_mask = None
    st.session_state.preview_rgb = None
    st.session_state.result_df = None
    st.session_state.trained = False
    st.session_state.clicked_points = []


def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))


def make_display_image(rgb, max_width=850):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_and_segment(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # stain-aware dark object extraction
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_sat = cv2.GaussianBlur(sat, (5, 5), 0)
    blur_val = cv2.GaussianBlur(val, (5, 5), 0)

    # dark/purple/pink tissue objects
    mask1 = blur < np.percentile(blur, 72)
    mask2 = blur_sat > np.percentile(blur_sat, 35)
    mask3 = blur_val < np.percentile(blur_val, 82)

    binary_bool = (mask1 & mask3) | (mask1 & mask2)

    binary_bool = morphology.remove_small_objects(binary_bool, min_size=35)
    binary_bool = morphology.remove_small_holes(binary_bool, area_threshold=35)
    binary_bool = morphology.binary_opening(binary_bool, morphology.disk(1))
    binary_bool = morphology.binary_closing(binary_bool, morphology.disk(1))

    dist = cv2.distanceTransform((binary_bool.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    local_max = feature.peak_local_max(
        dist,
        min_distance=7,
        labels=binary_bool
    )

    markers = np.zeros(binary_bool.shape, dtype=np.int32)
    for i, (r, c) in enumerate(local_max, start=1):
        markers[r, c] = i

    if len(local_max) == 0:
        labeled = measure.label(binary_bool)
    else:
        markers = morphology.dilation(markers, morphology.disk(2))
        labeled = segmentation.watershed(-dist, markers, mask=binary_bool)

    labeled = morphology.remove_small_objects(labeled, min_size=35)
    labeled = measure.label(labeled > 0)

    return gray, binary_bool, labeled


def extract_object_features(rgb, gray, labeled):
    records = []
    props = measure.regionprops(labeled, intensity_image=gray)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(rgb)

    for prop in props:
        area = prop.area
        if area < 35:
            continue

        minr, minc, maxr, maxc = prop.bbox
        perimeter = prop.perimeter if prop.perimeter > 0 else np.nan
        major_axis = prop.major_axis_length if prop.major_axis_length > 0 else np.nan
        minor_axis = prop.minor_axis_length if prop.minor_axis_length > 0 else np.nan

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter and perimeter > 0 else np.nan
        )
        roundness = (
            4 * area / (np.pi * (major_axis ** 2))
            if major_axis and major_axis > 0 else np.nan
        )

        mask = labeled[minr:maxr, minc:maxc] == prop.label
        gray_patch = gray[minr:maxr, minc:maxc]
        rgb_patch = rgb[minr:maxr, minc:maxc]
        hsv_patch = hsv[minr:maxr, minc:maxc]
        lab_patch = lab[minr:maxr, minc:maxc]

        if np.sum(mask) == 0:
            continue

        pix_gray = gray_patch[mask]
        pix_rgb = rgb_patch[mask]
        pix_hsv = hsv_patch[mask]
        pix_lab = lab_patch[mask]

        mean_intensity = float(np.mean(pix_gray))
        std_intensity = float(np.std(pix_gray))

        mean_r = float(np.mean(pix_rgb[:, 0]))
        mean_g = float(np.mean(pix_rgb[:, 1]))
        mean_b = float(np.mean(pix_rgb[:, 2]))

        mean_h = float(np.mean(pix_hsv[:, 0]))
        mean_s = float(np.mean(pix_hsv[:, 1]))
        mean_v = float(np.mean(pix_hsv[:, 2]))

        mean_l = float(np.mean(pix_lab[:, 0]))
        mean_a = float(np.mean(pix_lab[:, 1]))
        mean_b_lab = float(np.mean(pix_lab[:, 2]))

        lap_var = float(np.var(cv2.Laplacian(gray_patch, cv2.CV_64F)))
        eccentricity = float(prop.eccentricity) if prop.eccentricity is not None else np.nan
        solidity = float(prop.solidity) if prop.solidity is not None else np.nan
        extent = float(prop.extent) if prop.extent is not None else np.nan
        aspect_ratio = float(major_axis / minor_axis) if minor_axis and minor_axis > 0 else np.nan

        # proxy white/empty vacuole pixels inside each segmented cytoplasm object
        white_mask_rgb = (
            (pix_rgb[:, 0] > 210) &
            (pix_rgb[:, 1] > 210) &
            (pix_rgb[:, 2] > 210)
        )
        white_mask_hsv = (
            (pix_hsv[:, 2] > 210) &
            (pix_hsv[:, 1] < 45)
        )
        white_mask_lab = pix_lab[:, 0] > 82

        vac_mask = white_mask_rgb & white_mask_hsv & white_mask_lab
        vacuole_fraction = float(np.mean(vac_mask)) if len(vac_mask) > 0 else 0.0

        # pink density proxy
        pink_mask = (
            (pix_rgb[:, 0] > pix_rgb[:, 1]) &
            (pix_rgb[:, 0] > pix_rgb[:, 2]) &
            (pix_lab[:, 1] > 8)
        )
        pink_density = float(np.mean(pink_mask)) if len(pink_mask) > 0 else 0.0

        # darkness / pyknosis proxy
        dark_fraction = float(np.mean(pix_gray < np.percentile(gray, 35))) if len(pix_gray) > 0 else 0.0

        cy, cx = prop.centroid

        records.append({
            "label_id": int(prop.label),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "area": float(area),
            "perimeter": float(perimeter) if not np.isnan(perimeter) else np.nan,
            "major_axis_length": float(major_axis) if not np.isnan(major_axis) else np.nan,
            "minor_axis_length": float(minor_axis) if not np.isnan(minor_axis) else np.nan,
            "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
            "roundness": float(roundness) if not np.isnan(roundness) else np.nan,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "extent": extent,
            "aspect_ratio": aspect_ratio,
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "mean_h": mean_h,
            "mean_s": mean_s,
            "mean_v": mean_v,
            "mean_l": mean_l,
            "mean_a": mean_a,
            "mean_b_lab": mean_b_lab,
            "granularity": lap_var,
            "vacuole_fraction": vacuole_fraction,
            "pink_density": pink_density,
            "dark_fraction": dark_fraction
        })

    df = pd.DataFrame(records)

    if not df.empty:
        median_area = df["area"].median()
        if median_area > 0:
            df["variety_index"] = (df["area"] - median_area).abs() / median_area
        else:
            df["variety_index"] = 0.0

        median_intensity = df["mean_intensity"].median()
        std_intensity_global = df["mean_intensity"].std() if len(df) > 1 else 1.0
        if std_intensity_global == 0 or np.isnan(std_intensity_global):
            std_intensity_global = 1.0

        df["intensity_variety_index"] = (
            np.abs(df["mean_intensity"] - median_intensity) / std_intensity_global
        )

        df["combined_nucleus_variety"] = (
            0.6 * df["variety_index"] + 0.4 * (df["intensity_variety_index"] / 3.0)
        )

        df["cytoplasm_vacuolization_percent"] = df["vacuole_fraction"] * 100.0
        df["cytoplasm_pink_density_percent"] = df["pink_density"] * 100.0
    else:
        df["variety_index"] = []
        df["intensity_variety_index"] = []
        df["combined_nucleus_variety"] = []
        df["cytoplasm_vacuolization_percent"] = []
        df["cytoplasm_pink_density_percent"] = []

    return df


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "circularity", "roundness", "eccentricity", "solidity",
        "extent", "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "granularity", "vacuole_fraction", "pink_density", "dark_fraction",
        "variety_index", "intensity_variety_index", "combined_nucleus_variety"
    ]


def find_clicked_object_from_label(x, y, labeled, objects_df=None, search_radius=25):
    h, w = labeled.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    label_here = int(labeled[y, x])
    if label_here > 0:
        return label_here

    best_label = None
    best_dist = 1e9

    y0 = max(0, y - search_radius)
    y1 = min(h, y + search_radius + 1)
    x0 = max(0, x - search_radius)
    x1 = min(w, x + search_radius + 1)

    patch = labeled[y0:y1, x0:x1]
    coords = np.argwhere(patch > 0)

    for rr, cc in coords:
        yy = y0 + rr
        xx = x0 + cc
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        if d < best_dist:
            best_dist = d
            best_label = int(labeled[yy, xx])

    return best_label


def build_training_table(objects_df, samples_dict):
    rows = []
    for cls_name, label_ids in samples_dict.items():
        if cls_name == "Exclude":
            continue
        for lid in label_ids:
            row = objects_df[objects_df["label_id"] == lid]
            if len(row) == 1:
                r = row.iloc[0].copy()
                r["target_class"] = cls_name
                rows.append(r)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def annotate_image(rgb, objects_df, result_df=None, sample_dict=None, clicked_points=None):
    out = rgb.copy()

    color_map = {
        "Nucleus_Normal": (0, 255, 0),
        "Nucleus_Pyknotic": (255, 0, 0),
        "Cytoplasm": (0, 255, 255),
        "Inflammation": (255, 0, 255),
        "Exclude": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            pred = row.get("predicted_class", "Exclude")
            color_val = color_map.get(pred, (255, 255, 255))
            minr = int(row["bbox_minr"])
            minc = int(row["bbox_minc"])
            maxr = int(row["bbox_maxr"])
            maxc = int(row["bbox_maxc"])
            cv2.rectangle(out, (minc, minr), (maxc, maxr), color_val, 1)

    if clicked_points is not None:
        for p in clicked_points:
            x = int(p["x"])
            y = int(p["y"])
            cls = p["class"]
            color_val = color_map.get(cls, (255, 255, 255))
            cv2.drawMarker(
                out,
                (x, y),
                color_val,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2
            )

    if sample_dict is not None and objects_df is not None:
        for cls, label_ids in sample_dict.items():
            color_val = color_map.get(cls, (255, 255, 255))
            for lid in label_ids:
                row = objects_df[objects_df["label_id"] == lid]
                if len(row) == 1:
                    cx = int(row.iloc[0]["centroid_x"])
                    cy = int(row.iloc[0]["centroid_y"])
                    cv2.circle(out, (cx, cy), 6, color_val, 2)

    return out


def make_colored_segmentation(rgb, labeled, result_df=None):
    seg_rgb = np.zeros_like(rgb, dtype=np.uint8)

    color_map = {
        "Nucleus_Normal": (0, 255, 0),
        "Nucleus_Pyknotic": (255, 0, 0),
        "Cytoplasm": (0, 255, 255),
        "Inflammation": (255, 0, 255),
        "Exclude": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty and "predicted_class" in result_df.columns:
        for _, row in result_df.iterrows():
            lid = int(row["label_id"])
            cls = row["predicted_class"]
            seg_rgb[labeled == lid] = color_map.get(cls, (255, 255, 255))
    else:
        rng = np.random.default_rng(42)
        for lid in np.unique(labeled):
            if lid == 0:
                continue
            seg_rgb[labeled == lid] = rng.integers(50, 255, size=3, dtype=np.uint8)

    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    seg_rgb[boundaries] = [255, 255, 255]

    return seg_rgb


def crop_object_thumbnail(rgb, labeled, row, pad=8, target_size=96):
    minr = max(int(row["bbox_minr"]) - pad, 0)
    minc = max(int(row["bbox_minc"]) - pad, 0)
    maxr = min(int(row["bbox_maxr"]) + pad, rgb.shape[0])
    maxc = min(int(row["bbox_maxc"]) + pad, rgb.shape[1])

    crop_rgb = rgb[minr:maxr, minc:maxc].copy()
    crop_mask = (labeled[minr:maxr, minc:maxc] == int(row["label_id"]))

    if crop_rgb.size == 0:
        return None

    white_bg = np.ones_like(crop_rgb, dtype=np.uint8) * 255
    crop_object = white_bg.copy()
    crop_object[crop_mask] = crop_rgb[crop_mask]

    boundary = segmentation.find_boundaries(crop_mask, mode="outer")
    crop_object[boundary] = [255, 0, 0]

    h, w = crop_object.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_size / max(h, 1), target_size / max(w, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop_object, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def show_class_gallery(result_df, rgb, labeled, class_name, n_cols=6):
    class_df = result_df[result_df["predicted_class"] == class_name].copy()

    if class_df.empty:
        st.info(f"No objects in class: {class_name}")
        return

    st.markdown(f"### {class_name} ({len(class_df)})")

    rows = [class_df.iloc[i:i+n_cols] for i in range(0, len(class_df), n_cols)]

    for row_chunk in rows:
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            with cols[col_idx]:
                if col_idx < len(row_chunk):
                    row = row_chunk.iloc[col_idx]
                    thumb = crop_object_thumbnail(rgb, labeled, row, pad=8, target_size=96)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True)
                        st.caption(f"ID {int(row['label_id'])} | conf {row['confidence']:.2f}")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    annotation_target = st.selectbox(
        "Annotation class",
        ["Nucleus_Normal", "Nucleus_Pyknotic", "Cytoplasm", "Inflammation", "Exclude"]
    )

    show_segmentation_preview = st.checkbox("Show segmentation overlay", value=False)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write("Training samples")
    for cls in TRAIN_CLASSES:
        st.write(f"**{cls}**: {len(st.session_state.samples.get(cls, []))}")


# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload histopathology liver image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    if st.session_state.last_uploaded_name != uploaded.name:
        reset_all()
        st.session_state.last_uploaded_name = uploaded.name

    pil_img = Image.open(uploaded)
    rgb = pil_to_rgb_array(pil_img)

    gray, binary_mask, labeled = preprocess_and_segment(rgb)
    objects_df = extract_object_features(rgb, gray, labeled)

    st.session_state.objects_df = objects_df
    st.session_state.labeled_mask = labeled
    st.session_state.preview_rgb = rgb

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Annotation")

        preview_img = annotate_image(
            rgb=rgb,
            objects_df=st.session_state.objects_df,
            result_df=st.session_state.result_df,
            sample_dict=st.session_state.samples,
            clicked_points=st.session_state.clicked_points
        )

        if show_segmentation_preview:
            boundary_overlay = segmentation.mark_boundaries(preview_img, labeled, color=(1, 1, 0))
            display_base = (boundary_overlay * 255).astype(np.uint8)
        else:
            display_base = preview_img

        display_img, scale = make_display_image(display_base, max_width=850)
        click = streamlit_image_coordinates(display_img, key="liver_click")

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)

            picked_id = find_clicked_object_from_label(
                real_x,
                real_y,
                st.session_state.labeled_mask,
                st.session_state.objects_df,
                search_radius=25
            )

            if picked_id is not None:
                already_used = any(
                    picked_id in st.session_state.samples[c]
                    for c in TRAIN_CLASSES
                )
                if not already_used:
                    st.session_state.samples[annotation_target].append(picked_id)
                    st.session_state.clicked_points.append({
                        "x": real_x,
                        "y": real_y,
                        "class": annotation_target,
                        "label_id": picked_id
                    })
                    st.rerun()

        st.caption("Klik objek untuk dijadikan training sample sesuai class yang dipilih.")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Undo last"):
                if len(st.session_state.samples[annotation_target]) > 0:
                    last_lid = st.session_state.samples[annotation_target].pop()

                    for i in range(len(st.session_state.clicked_points)-1, -1, -1):
                        if (
                            st.session_state.clicked_points[i]["class"] == annotation_target and
                            st.session_state.clicked_points[i]["label_id"] == last_lid
                        ):
                            st.session_state.clicked_points.pop(i)
                            break
                    st.rerun()

        with c2:
            if st.button("Clear selected class"):
                lids_to_remove = set(st.session_state.samples[annotation_target])
                st.session_state.samples[annotation_target] = []
                st.session_state.clicked_points = [
                    p for p in st.session_state.clicked_points
                    if not (p["class"] == annotation_target and p["label_id"] in lids_to_remove)
                ]
                st.rerun()

        with c3:
            if st.button("Show segmentation only"):
                seg_vis = make_colored_segmentation(
                    rgb=st.session_state.preview_rgb,
                    labeled=st.session_state.labeled_mask,
                    result_df=None
                )
                st.image(seg_vis, caption="Segmentation preview", use_container_width=True)

    with right:
        st.subheader("Training and analysis")

        training_df = build_training_table(
            st.session_state.objects_df,
            st.session_state.samples
        )

        st.write(f"Detected objects: {len(st.session_state.objects_df) if st.session_state.objects_df is not None else 0}")
        st.write(f"Training objects selected: {len(training_df)}")

        if not training_df.empty:
            st.dataframe(
                training_df[["label_id", "target_class"] + feature_columns()].head(30),
                use_container_width=True
            )

        if st.button("Run classification"):
            if training_df.empty:
                st.error("Belum ada training sample.")
            else:
                X_train = training_df[feature_columns()].copy()
                y_train = training_df["target_class"].copy()

                X_all = st.session_state.objects_df[feature_columns()].copy()

                medians = X_train.median(numeric_only=True)
                X_train = X_train.fillna(medians)
                X_all = X_all.fillna(medians)

                clf = RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced"
                )
                clf.fit(X_train, y_train)

                pred = clf.predict(X_all)
                proba = clf.predict_proba(X_all)
                max_proba = proba.max(axis=1)

                result_df = st.session_state.objects_df.copy()
                result_df["predicted_class"] = pred
                result_df["confidence"] = max_proba

                exclude_ids = set(st.session_state.samples.get("Exclude", []))
                if len(exclude_ids) > 0:
                    result_df.loc[result_df["label_id"].isin(exclude_ids), "predicted_class"] = "Exclude"
                    result_df.loc[result_df["label_id"].isin(exclude_ids), "confidence"] = 1.0

                # derived flags for liver scoring
                result_df["is_nucleus"] = result_df["predicted_class"].isin(["Nucleus_Normal", "Nucleus_Pyknotic"]).astype(int)
                result_df["is_cytoplasm"] = (result_df["predicted_class"] == "Cytoplasm").astype(int)
                result_df["is_inflammation"] = (result_df["predicted_class"] == "Inflammation").astype(int)
                result_df["pyknosis_flag"] = np.where(
                    result_df["predicted_class"] == "Nucleus_Pyknotic",
                    "Pyknotic",
                    np.where(result_df["predicted_class"] == "Nucleus_Normal", "Normal", "")
                )

                st.session_state.result_df = result_df
                st.session_state.trained = True

        if st.session_state.trained and st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Classification complete")

            nucleus_df = result_df[result_df["is_nucleus"] == 1].copy()
            cytoplasm_df = result_df[result_df["is_cytoplasm"] == 1].copy()
            inflam_df = result_df[result_df["is_inflammation"] == 1].copy()

            total_nuclei = len(nucleus_df)
            total_pyknotic = int((nucleus_df["predicted_class"] == "Nucleus_Pyknotic").sum())
            pyknosis_percent = 100 * total_pyknotic / total_nuclei if total_nuclei > 0 else 0.0

            mean_nucleus_variety = nucleus_df["combined_nucleus_variety"].mean() if len(nucleus_df) > 0 else 0.0
            mean_vacuolization = cytoplasm_df["cytoplasm_vacuolization_percent"].mean() if len(cytoplasm_df) > 0 else 0.0
            mean_pink_density = cytoplasm_df["cytoplasm_pink_density_percent"].mean() if len(cytoplasm_df) > 0 else 0.0

            total_area = result_df["area"].sum() if len(result_df) > 0 else 1.0
            inflam_area = inflam_df["area"].sum() if len(inflam_df) > 0 else 0.0
            inflammation_area_percent = 100 * inflam_area / total_area if total_area > 0 else 0.0

            total_fov_area = rgb.shape[0] * rgb.shape[1]
            inflammation_fov_percent = 100 * inflam_area / total_fov_area if total_fov_area > 0 else 0.0

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total nuclei", f"{total_nuclei}")
                st.metric("Pyknotic nuclei", f"{total_pyknotic}")
                st.metric("Pyknosis (%)", f"{pyknosis_percent:.1f}%")
            with m2:
                st.metric("Mean nucleus variety", f"{mean_nucleus_variety:.3f}")
                st.metric("Inflammatory objects", f"{len(inflam_df)}")
                st.metric("Inflammation / field", f"{inflammation_fov_percent:.2f}%")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Mean vacuolization", f"{mean_vacuolization:.1f}%")
            with m4:
                st.metric("Mean pink density", f"{mean_pink_density:.1f}%")

            st.write("### Nucleus analysis")
            if len(nucleus_df) > 0:
                nucleus_table = nucleus_df[[
                    "label_id",
                    "area",
                    "mean_intensity",
                    "dark_fraction",
                    "combined_nucleus_variety",
                    "predicted_class",
                    "pyknosis_flag",
                    "confidence"
                ]].copy()
                st.dataframe(nucleus_table, use_container_width=True)
            else:
                st.info("No nucleus objects detected.")

            st.write("### Cytoplasm analysis")
            if len(cytoplasm_df) > 0:
                cytoplasm_table = cytoplasm_df[[
                    "label_id",
                    "area",
                    "cytoplasm_vacuolization_percent",
                    "cytoplasm_pink_density_percent",
                    "confidence"
                ]].copy()
                st.dataframe(cytoplasm_table, use_container_width=True)
            else:
                st.info("No cytoplasm objects detected.")

            st.write("### Inflammation analysis")
            if len(inflam_df) > 0:
                inflam_table = inflam_df[[
                    "label_id",
                    "area",
                    "circularity",
                    "mean_intensity",
                    "confidence"
                ]].copy()
                st.dataframe(inflam_table, use_container_width=True)
            else:
                st.info("No inflammatory objects detected.")

            if len(nucleus_df) > 0:
                fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                vals = nucleus_df["predicted_class"].value_counts()
                ax1.bar(vals.index, vals.values)
                ax1.set_title("Nucleus class distribution")
                ax1.set_ylabel("Count")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig1)
                plt.close(fig1)

            if len(cytoplasm_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.hist(cytoplasm_df["cytoplasm_vacuolization_percent"], bins=20)
                ax2.set_title("Cytoplasm vacuolization (%)")
                ax2.set_xlabel("Percent white/empty area")
                ax2.set_ylabel("Frequency")
                st.pyplot(fig2)
                plt.close(fig2)

            if len(inflam_df) > 0:
                fig3, ax3 = plt.subplots(figsize=(5, 3.5))
                ax3.hist(inflam_df["area"], bins=20)
                ax3.set_title("Inflammatory object area")
                ax3.set_xlabel("Area")
                ax3.set_ylabel("Frequency")
                st.pyplot(fig3)
                plt.close(fig3)

            annotated = annotate_image(
                rgb=st.session_state.preview_rgb,
                objects_df=st.session_state.objects_df,
                result_df=result_df,
                sample_dict=st.session_state.samples,
                clicked_points=st.session_state.clicked_points
            )
            st.image(annotated, caption="Annotated image", use_container_width=True)

            seg_result = make_colored_segmentation(
                rgb=st.session_state.preview_rgb,
                labeled=st.session_state.labeled_mask,
                result_df=result_df
            )
            st.image(seg_result, caption="Class-colored segmentation", use_container_width=True)

            st.markdown("---")
            st.subheader("Object gallery by class")

            for cls in ["Nucleus_Normal", "Nucleus_Pyknotic", "Cytoplasm", "Inflammation"]:
                show_class_gallery(
                    result_df=result_df[result_df["predicted_class"] != "Exclude"].copy(),
                    rgb=st.session_state.preview_rgb,
                    labeled=st.session_state.labeled_mask,
                    class_name=cls,
                    n_cols=6
                )

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download liver scoring CSV",
                data=csv,
                file_name="liver_histopathology_scoring.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption(
    "Prototype only. Klik dipakai sebagai training sample object-class. "
    "Hasil terbaik didapat bila tiap class diberi beberapa contoh yang representatif."
)
