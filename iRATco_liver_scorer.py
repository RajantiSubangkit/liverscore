import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

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
    "Semi-automatic scoring of inflammation, nuclear variety/anisocytosis, "
    "and cytoplasmic vacuolization from liver histopathology images"
)

# =========================================================
# CLASSES
# =========================================================
INFLAM_CLASSES = ["Mild", "Moderate", "Severe"]
NUCLEUS_CLASSES = ["Low variety", "Moderate variety", "High variety"]
CYTOPLASM_CLASSES = ["Low vacuolization", "Moderate vacuolization", "High vacuolization"]

# =========================================================
# SESSION STATE
# =========================================================
if "inflam_samples" not in st.session_state:
    st.session_state.inflam_samples = {cls: [] for cls in INFLAM_CLASSES}

if "nucleus_samples" not in st.session_state:
    st.session_state.nucleus_samples = {cls: [] for cls in NUCLEUS_CLASSES}

if "cytoplasm_samples" not in st.session_state:
    st.session_state.cytoplasm_samples = {cls: [] for cls in CYTOPLASM_CLASSES}

if "objects_df" not in st.session_state:
    st.session_state.objects_df = None

if "labeled_mask" not in st.session_state:
    st.session_state.labeled_mask = None

if "preview_rgb" not in st.session_state:
    st.session_state.preview_rgb = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None


# =========================================================
# HELPERS
# =========================================================
def reset_all():
    st.session_state.inflam_samples = {cls: [] for cls in INFLAM_CLASSES}
    st.session_state.nucleus_samples = {cls: [] for cls in NUCLEUS_CLASSES}
    st.session_state.cytoplasm_samples = {cls: [] for cls in CYTOPLASM_CLASSES}
    st.session_state.objects_df = None
    st.session_state.labeled_mask = None
    st.session_state.preview_rgb = None
    st.session_state.result_df = None


def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))


def make_display_image(rgb, max_width=900):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (max_width, new_h))
    return resized, scale


def preprocess_and_segment(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary_bool = binary > 0
    binary_bool = morphology.remove_small_objects(binary_bool, min_size=50)
    binary_bool = morphology.remove_small_holes(binary_bool, area_threshold=50)

    dist = cv2.distanceTransform((binary_bool.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    local_max = feature.peak_local_max(
        dist,
        min_distance=8,
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

    return gray, binary_bool, labeled


def extract_object_features(rgb, gray, labeled):
    records = []
    props = measure.regionprops(labeled, intensity_image=gray)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(rgb)

    for prop in props:
        area = prop.area
        if area < 50:
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

        # proxy vakuolisasi:
        # piksel terang + saturasi rendah + area tampak pucat
        bright_low_sat = (pix_hsv[:, 2] > 170) & (pix_hsv[:, 1] < 80)
        pale_lab = pix_lab[:, 0] > 70
        vac_mask = bright_low_sat & pale_lab
        vacuole_fraction = float(np.mean(vac_mask)) if len(vac_mask) > 0 else 0.0

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
            "vacuole_fraction": vacuole_fraction
        })

    df = pd.DataFrame(records)

    if not df.empty:
        median_area = df["area"].median()
        if median_area > 0:
            df["variety_index"] = (df["area"] - median_area).abs() / median_area
        else:
            df["variety_index"] = 0.0

        df["anisocytosis_flag"] = np.where(df["variety_index"] > 0.5, "Anisocytosis", "Non-anisocytosis")
        df["cytoplasm_vacuolization_percent"] = df["vacuole_fraction"] * 100.0
    else:
        df["variety_index"] = []
        df["anisocytosis_flag"] = []
        df["cytoplasm_vacuolization_percent"] = []

    return df


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "circularity", "roundness", "eccentricity", "solidity",
        "extent", "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "granularity", "vacuole_fraction", "variety_index"
    ]


def find_nearest_object(x, y, objects_df, max_dist=35):
    if objects_df is None or objects_df.empty:
        return None

    dx = objects_df["centroid_x"] - x
    dy = objects_df["centroid_y"] - y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    idx = dist.idxmin()

    if dist.loc[idx] <= max_dist:
        return int(objects_df.loc[idx, "label_id"])
    return None


def build_training_table(objects_df, samples_dict, target_name):
    rows = []
    for cls_name, label_ids in samples_dict.items():
        for lid in label_ids:
            row = objects_df[objects_df["label_id"] == lid]
            if len(row) == 1:
                r = row.iloc[0].copy()
                r[target_name] = cls_name
                rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def annotate_image(rgb, result_df=None, mode="Nucleus"):
    out = rgb.copy()

    color_map = {
        "Mild": (80, 180, 255),
        "Moderate": (255, 200, 0),
        "Severe": (255, 0, 0),
        "Low variety": (180, 180, 180),
        "Moderate variety": (255, 200, 0),
        "High variety": (255, 0, 0),
        "Low vacuolization": (180, 180, 180),
        "Moderate vacuolization": (0, 200, 255),
        "High vacuolization": (0, 0, 255),
    }

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            if mode == "Inflammation":
                cls = row.get("inflammation_class", "Mild")
            elif mode == "Nucleus":
                cls = row.get("nucleus_class", "Low variety")
            else:
                cls = row.get("cytoplasm_class", "Low vacuolization")

            color_val = color_map.get(cls, (255, 255, 255))
            minr = int(row["bbox_minr"])
            minc = int(row["bbox_minc"])
            maxr = int(row["bbox_maxr"])
            maxc = int(row["bbox_maxc"])
            cv2.rectangle(out, (minc, minr), (maxc, maxr), color_val, 1)

    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    mode = st.radio(
        "Analysis panel",
        ["Inflammation", "Nucleus", "Cytoplasm"]
    )

    if mode == "Inflammation":
        active_class = st.selectbox("Active class", INFLAM_CLASSES)
    elif mode == "Nucleus":
        active_class = st.selectbox("Active class", NUCLEUS_CLASSES)
    else:
        active_class = st.selectbox("Active class", CYTOPLASM_CLASSES)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write("Current samples")

    st.write("**Inflammation**")
    for c in INFLAM_CLASSES:
        st.write(f"{c}: {len(st.session_state.inflam_samples[c])}")

    st.write("**Nucleus**")
    for c in NUCLEUS_CLASSES:
        st.write(f"{c}: {len(st.session_state.nucleus_samples[c])}")

    st.write("**Cytoplasm**")
    for c in CYTOPLASM_CLASSES:
        st.write(f"{c}: {len(st.session_state.cytoplasm_samples[c])}")


# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload histopathology liver image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    if st.session_state.last_uploaded_name != uploaded.name:
        st.session_state.last_uploaded_name = uploaded.name
        st.session_state.result_df = None
        st.session_state.objects_df = None
        st.session_state.labeled_mask = None
        st.session_state.preview_rgb = None

    pil_img = Image.open(uploaded)
    rgb = pil_to_rgb_array(pil_img)

    gray, binary_mask, labeled = preprocess_and_segment(rgb)
    objects_df = extract_object_features(rgb, gray, labeled)

    st.session_state.objects_df = objects_df
    st.session_state.labeled_mask = labeled
    st.session_state.preview_rgb = rgb

    colL, colR = st.columns([1.4, 1])

    with colL:
        st.subheader("Annotation")
        preview = annotate_image(
            rgb,
            result_df=st.session_state.result_df,
            mode=mode
        )
        display_img, scale = make_display_image(preview, max_width=850)
        click = streamlit_image_coordinates(display_img)

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)
            nearest_id = find_nearest_object(real_x, real_y, st.session_state.objects_df)

            if nearest_id is not None:
                if mode == "Inflammation":
                    target_dict = st.session_state.inflam_samples
                elif mode == "Nucleus":
                    target_dict = st.session_state.nucleus_samples
                else:
                    target_dict = st.session_state.cytoplasm_samples

                if nearest_id not in target_dict[active_class]:
                    already_used = any(nearest_id in ids for ids in target_dict.values())
                    if not already_used:
                        target_dict[active_class].append(nearest_id)
                        st.rerun()

        st.caption("Click near an object to assign the selected class.")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Show segmentation mask"):
                seg_vis = segmentation.mark_boundaries(rgb, labeled, color=(1, 1, 0))
                st.image(seg_vis, caption="Segmentation preview", use_container_width=True)

        with c2:
            if st.button("Show raw object table"):
                st.dataframe(st.session_state.objects_df, use_container_width=True)

    with colR:
        st.subheader("Training and scoring")

        inflam_df = build_training_table(
            st.session_state.objects_df,
            st.session_state.inflam_samples,
            "inflammation_class"
        )

        nucleus_df = build_training_table(
            st.session_state.objects_df,
            st.session_state.nucleus_samples,
            "nucleus_class"
        )

        cytoplasm_df = build_training_table(
            st.session_state.objects_df,
            st.session_state.cytoplasm_samples,
            "cytoplasm_class"
        )

        st.write(f"Inflammation samples: {len(inflam_df)}")
        st.write(f"Nucleus samples: {len(nucleus_df)}")
        st.write(f"Cytoplasm samples: {len(cytoplasm_df)}")

        if st.button("Run analysis"):
            result_df = st.session_state.objects_df.copy()
            X_all = result_df[feature_columns()].copy()

            # -------------------------
            # Inflammation
            # -------------------------
            if not inflam_df.empty and inflam_df["inflammation_class"].nunique() >= 2:
                X_train = inflam_df[feature_columns()].copy()
                y_train = inflam_df["inflammation_class"].copy()
                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_inflam = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_inflam.fit(X_train, y_train)
                result_df["inflammation_class"] = clf_inflam.predict(X_all_filled)
            else:
                # fallback sederhana: object kecil-gelap-cukup bulat condong sel radang
                inflam_score = (
                    (result_df["area"] < result_df["area"].median()).astype(int) +
                    (result_df["mean_intensity"] < result_df["mean_intensity"].median()).astype(int) +
                    (result_df["circularity"] > 0.5).fillna(False).astype(int)
                )
                result_df["inflammation_class"] = np.select(
                    [inflam_score <= 0, inflam_score == 1, inflam_score >= 2],
                    ["Mild", "Moderate", "Severe"],
                    default="Mild"
                )

            # -------------------------
            # Nucleus
            # -------------------------
            if not nucleus_df.empty and nucleus_df["nucleus_class"].nunique() >= 2:
                X_train = nucleus_df[feature_columns()].copy()
                y_train = nucleus_df["nucleus_class"].copy()
                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_nucleus = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_nucleus.fit(X_train, y_train)
                result_df["nucleus_class"] = clf_nucleus.predict(X_all_filled)
            else:
                result_df["nucleus_class"] = np.select(
                    [
                        result_df["variety_index"] < 0.25,
                        (result_df["variety_index"] >= 0.25) & (result_df["variety_index"] <= 0.50),
                        result_df["variety_index"] > 0.50
                    ],
                    ["Low variety", "Moderate variety", "High variety"],
                    default="Low variety"
                )

            result_df["anisocytosis_flag"] = np.where(
                result_df["variety_index"] > 0.5,
                "Anisocytosis",
                "Non-anisocytosis"
            )

            # -------------------------
            # Cytoplasm
            # -------------------------
            if not cytoplasm_df.empty and cytoplasm_df["cytoplasm_class"].nunique() >= 2:
                X_train = cytoplasm_df[feature_columns()].copy()
                y_train = cytoplasm_df["cytoplasm_class"].copy()
                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_cytoplasm = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_cytoplasm.fit(X_train, y_train)
                result_df["cytoplasm_class"] = clf_cytoplasm.predict(X_all_filled)
            else:
                result_df["cytoplasm_class"] = np.select(
                    [
                        result_df["cytoplasm_vacuolization_percent"] < 15,
                        (result_df["cytoplasm_vacuolization_percent"] >= 15) &
                        (result_df["cytoplasm_vacuolization_percent"] <= 35),
                        result_df["cytoplasm_vacuolization_percent"] > 35
                    ],
                    ["Low vacuolization", "Moderate vacuolization", "High vacuolization"],
                    default="Low vacuolization"
                )

            st.session_state.result_df = result_df

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Analysis complete")

            # =====================================================
            # METRICS
            # =====================================================
            inflammation_percent = 100 * (
                result_df["inflammation_class"].isin(["Moderate", "Severe"])
            ).mean()

            anisocytosis_percent = 100 * (
                result_df["anisocytosis_flag"] == "Anisocytosis"
            ).mean()

            mean_vacuolization_percent = result_df["cytoplasm_vacuolization_percent"].mean()

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Inflammation percent", f"{inflammation_percent:.1f}%")
            with m2:
                st.metric("Anisocytosis percent", f"{anisocytosis_percent:.1f}%")
            with m3:
                st.metric("Mean vacuolization", f"{mean_vacuolization_percent:.1f}%")

            # =====================================================
            # SUMMARY TABLES
            # =====================================================
            inflam_summary = (
                result_df["inflammation_class"]
                .value_counts()
                .rename_axis("Inflammation")
                .reset_index(name="Count")
            )

            nucleus_summary = (
                result_df["nucleus_class"]
                .value_counts()
                .rename_axis("Nucleus variety")
                .reset_index(name="Count")
            )

            cytoplasm_summary = (
                result_df["cytoplasm_class"]
                .value_counts()
                .rename_axis("Cytoplasm vacuolization")
                .reset_index(name="Count")
            )

            st.write("### Inflammation")
            st.dataframe(inflam_summary, use_container_width=True)
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            ax1.bar(inflam_summary["Inflammation"], inflam_summary["Count"])
            ax1.set_ylabel("Count")
            ax1.set_title("Inflammation")
            st.pyplot(fig1)
            plt.close(fig1)

            st.write("### Nucleus")
            st.dataframe(nucleus_summary, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.bar(nucleus_summary["Nucleus variety"], nucleus_summary["Count"])
            ax2.set_ylabel("Count")
            ax2.set_title("Nuclear variety")
            st.pyplot(fig2)
            plt.close(fig2)

            st.write("### Cytoplasm")
            st.dataframe(cytoplasm_summary, use_container_width=True)
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            ax3.bar(cytoplasm_summary["Cytoplasm vacuolization"], cytoplasm_summary["Count"])
            ax3.set_ylabel("Count")
            ax3.set_title("Cytoplasmic vacuolization")
            st.pyplot(fig3)
            plt.close(fig3)

            # =====================================================
            # EXTRA INDEX TABLE
            # =====================================================
            st.write("### Quantitative indices")
            quant_df = result_df[[
                "label_id",
                "variety_index",
                "anisocytosis_flag",
                "cytoplasm_vacuolization_percent",
                "inflammation_class",
                "nucleus_class",
                "cytoplasm_class"
            ]].copy()
            st.dataframe(quant_df, use_container_width=True)

            # =====================================================
            # IMAGE OUTPUT
            # =====================================================
            st.image(
                annotate_image(
                    st.session_state.preview_rgb,
                    result_df=result_df,
                    mode=mode
                ),
                caption=f"Annotated image - {mode}",
                use_container_width=True
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
    "Prototype only. Best used as a semi-automatic starting point. "
    "Segmentation, nucleus/cytoplasm separation, inflammation logic, and thresholds "
    "should be refined for stain, magnification, species, and tissue quality."
)
