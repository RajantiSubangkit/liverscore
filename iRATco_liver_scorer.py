import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import measure, morphology, segmentation, feature
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="iRATco-Liver Histo Scorer",
    layout="wide"
)

st.title("iRATco-Liver Histo Scorer")
st.caption("Semi-automatic scoring of hepatocyte nuclear anisocytosis, cytoplasmic vacuolization, and interstitial inflammatory cells")

# =========================================================
# CLASSES
# =========================================================
ANISO_CLASSES = ["Mild", "Moderate", "Severe"]
INFLAM_CLASSES = ["Mild", "Moderate", "Severe"]
ALL_VAC_CLASSES = ["Vacuolated", "Non-vacuolated"]

# =========================================================
# SESSION STATE
# =========================================================
if "hepatocyte_samples" not in st.session_state:
    st.session_state.hepatocyte_samples = {cls: [] for cls in ANISO_CLASSES}

if "inflam_samples" not in st.session_state:
    st.session_state.inflam_samples = {cls: [] for cls in INFLAM_CLASSES}

if "vacuole_samples" not in st.session_state:
    st.session_state.vacuole_samples = {cls: [] for cls in ALL_VAC_CLASSES}

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
    st.session_state.hepatocyte_samples = {cls: [] for cls in ANISO_CLASSES}
    st.session_state.inflam_samples = {cls: [] for cls in INFLAM_CLASSES}
    st.session_state.vacuole_samples = {cls: [] for cls in ALL_VAC_CLASSES}
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

        if np.sum(mask) == 0:
            continue

        pix_gray = gray_patch[mask]
        pix_rgb = rgb_patch[mask]
        pix_hsv = hsv_patch[mask]

        mean_intensity = float(np.mean(pix_gray))
        std_intensity = float(np.std(pix_gray))
        mean_r = float(np.mean(pix_rgb[:, 0]))
        mean_g = float(np.mean(pix_rgb[:, 1]))
        mean_b = float(np.mean(pix_rgb[:, 2]))
        mean_h = float(np.mean(pix_hsv[:, 0]))
        mean_s = float(np.mean(pix_hsv[:, 1]))
        mean_v = float(np.mean(pix_hsv[:, 2]))

        lap_var = float(np.var(cv2.Laplacian(gray_patch, cv2.CV_64F)))
        eccentricity = float(prop.eccentricity) if prop.eccentricity is not None else np.nan
        solidity = float(prop.solidity) if prop.solidity is not None else np.nan
        extent = float(prop.extent) if prop.extent is not None else np.nan
        aspect_ratio = float(major_axis / minor_axis) if minor_axis and minor_axis > 0 else np.nan

        # Vacuolization proxy: bright low-saturation pixels inside object
        bright_mask = (pix_hsv[:, 2] > 170) & (pix_hsv[:, 1] < 70)
        vacuole_fraction = float(np.mean(bright_mask)) if len(bright_mask) > 0 else 0.0

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
            "granularity": lap_var,
            "vacuole_fraction": vacuole_fraction
        })

    return pd.DataFrame(records)


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "circularity", "roundness", "eccentricity", "solidity",
        "extent", "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v",
        "granularity", "vacuole_fraction"
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


def annotate_image(rgb, objects_df, result_df=None):
    out = rgb.copy()
    color_map = {
        "Mild": (80, 180, 255),
        "Moderate": (255, 200, 0),
        "Severe": (255, 0, 0),
        "Vacuolated": (0, 255, 255),
        "Non-vacuolated": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            cls = row.get("anisocytosis_class", "Mild")
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
    st.header("Annotation mode")
    mode = st.radio(
        "Choose task",
        ["Hepatocyte nucleus anisocytosis", "Cytoplasmic vacuolization", "Interstitial inflammation"]
    )

    if mode == "Hepatocyte nucleus anisocytosis":
        active_class = st.selectbox("Active class", ANISO_CLASSES)
    elif mode == "Cytoplasmic vacuolization":
        active_class = st.selectbox("Active class", ALL_VAC_CLASSES)
    else:
        active_class = st.selectbox("Active class", INFLAM_CLASSES)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write("Current samples")
    st.write("**Anisocytosis**")
    for c in ANISO_CLASSES:
        st.write(f"{c}: {len(st.session_state.hepatocyte_samples[c])}")
    st.write("**Vacuolization**")
    for c in ALL_VAC_CLASSES:
        st.write(f"{c}: {len(st.session_state.vacuole_samples[c])}")
    st.write("**Interstitial inflammation**")
    for c in INFLAM_CLASSES:
        st.write(f"{c}: {len(st.session_state.inflam_samples[c])}")


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
        display_img, scale = make_display_image(rgb, max_width=850)
        click = streamlit_image_coordinates(display_img)

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)
            nearest_id = find_nearest_object(real_x, real_y, st.session_state.objects_df)

            if nearest_id is not None:
                if mode == "Hepatocyte nucleus anisocytosis":
                    target_dict = st.session_state.hepatocyte_samples
                elif mode == "Cytoplasmic vacuolization":
                    target_dict = st.session_state.vacuole_samples
                else:
                    target_dict = st.session_state.inflam_samples

                if nearest_id not in target_dict[active_class]:
                    already_used = any(nearest_id in ids for ids in target_dict.values())
                    if not already_used:
                        target_dict[active_class].append(nearest_id)
                        st.rerun()

        st.caption("Click near an object to assign the selected score/class.")

        if st.button("Show segmentation mask"):
            seg_vis = segmentation.mark_boundaries(rgb, labeled, color=(1, 1, 0))
            st.image(seg_vis, caption="Segmentation preview", use_container_width=True)

    with colR:
        st.subheader("Training and scoring")

        aniso_df = build_training_table(st.session_state.objects_df, st.session_state.hepatocyte_samples, "anisocytosis_class")
        vac_df = build_training_table(st.session_state.objects_df, st.session_state.vacuole_samples, "vacuole_class")
        inflam_df = build_training_table(st.session_state.objects_df, st.session_state.inflam_samples, "inflammation_class")

        st.write(f"Anisocytosis samples: {len(aniso_df)}")
        st.write(f"Vacuolization samples: {len(vac_df)}")
        st.write(f"Inflammation samples: {len(inflam_df)}")

        if st.button("Run analysis"):
            result_df = st.session_state.objects_df.copy()
            X_all = result_df[feature_columns()].copy()

            if not aniso_df.empty and aniso_df["anisocytosis_class"].nunique() >= 2:
                X_train = aniso_df[feature_columns()].copy()
                y_train = aniso_df["anisocytosis_class"].copy()
                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_aniso = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_aniso.fit(X_train, y_train)
                result_df["anisocytosis_class"] = clf_aniso.predict(X_all_filled)
            else:
                result_df["anisocytosis_class"] = "Mild"

            if not vac_df.empty and vac_df["vacuole_class"].nunique() >= 2:
                X_train = vac_df[feature_columns()].copy()
                y_train = vac_df["vacuole_class"].copy()
                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_vac = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_vac.fit(X_train, y_train)
                result_df["vacuole_class"] = clf_vac.predict(X_all_filled)
            else:
                result_df["vacuole_class"] = np.where(result_df["vacuole_fraction"] >= 0.15, "Vacuolated", "Non-vacuolated")

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
                result_df["inflammation_class"] = "Mild"

            st.session_state.result_df = result_df

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()

            st.success("Analysis complete")

            vac_pct = 100 * (result_df["vacuole_class"] == "Vacuolated").mean()
            st.metric("Estimated vacuolization", f"{vac_pct:.1f}%")

            anisocytosis_summary = (
                result_df["anisocytosis_class"]
                .value_counts()
                .rename_axis("Anisocytosis")
                .reset_index(name="Count")
            )
            inflammation_summary = (
                result_df["inflammation_class"]
                .value_counts()
                .rename_axis("Interstitial inflammation")
                .reset_index(name="Count")
            )

            st.write("### Hepatocyte nuclear anisocytosis")
            st.dataframe(anisocytosis_summary, use_container_width=True)

            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            ax1.bar(anisocytosis_summary["Anisocytosis"], anisocytosis_summary["Count"])
            ax1.set_ylabel("Count")
            ax1.set_title("Nuclear anisocytosis")
            st.pyplot(fig1)
            plt.close(fig1)

            st.write("### Interstitial inflammatory cells")
            st.dataframe(inflammation_summary, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.bar(inflammation_summary["Interstitial inflammation"], inflammation_summary["Count"])
            ax2.set_ylabel("Count")
            ax2.set_title("Interstitial inflammation")
            st.pyplot(fig2)
            plt.close(fig2)

            vac_summary = pd.DataFrame({
                "Category": ["Vacuolated", "Non-vacuolated"],
                "Count": [
                    int((result_df["vacuole_class"] == "Vacuolated").sum()),
                    int((result_df["vacuole_class"] == "Non-vacuolated").sum())
                ]
            })
            vac_summary["Percent"] = 100 * vac_summary["Count"] / vac_summary["Count"].sum()

            st.write("### Cytoplasmic vacuolization")
            st.dataframe(vac_summary, use_container_width=True)

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.pie(vac_summary["Count"], labels=vac_summary["Category"], autopct="%1.1f%%", startangle=90)
            ax3.set_title("Vacuolization")
            st.pyplot(fig3)
            plt.close(fig3)

            st.image(
                annotate_image(st.session_state.preview_rgb, st.session_state.objects_df, result_df=result_df),
                caption="Annotated image",
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
    "Prototype only. Best used as a semi-automatic starting point; thresholds, segmentation, and class logic should be refined for stain, magnification, and species."
)
