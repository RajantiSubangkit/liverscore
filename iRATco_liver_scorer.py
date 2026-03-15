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
    "Semi-automatic analysis of nuclei, cytoplasm, and inflammatory cells "
    "from liver histopathology images"
)

# =========================================================
# SESSION STATE
# =========================================================
if "nucleus_samples" not in st.session_state:
    st.session_state.nucleus_samples = []

if "cytoplasm_samples" not in st.session_state:
    st.session_state.cytoplasm_samples = []

if "inflam_samples" not in st.session_state:
    st.session_state.inflam_samples = []

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
    st.session_state.nucleus_samples = []
    st.session_state.cytoplasm_samples = []
    st.session_state.inflam_samples = []
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

        # vacuolization proxy
        bright_low_sat = (pix_hsv[:, 2] > 170) & (pix_hsv[:, 1] < 80)
        pale_lab = pix_lab[:, 0] > 70
        vac_mask = bright_low_sat & pale_lab
        vacuole_fraction = float(np.mean(vac_mask)) if len(vac_mask) > 0 else 0.0

        # color density proxy
        color_density = float(
            np.mean(
                np.sqrt(
                    (pix_rgb[:, 0] / 255.0) ** 2 +
                    (pix_rgb[:, 1] / 255.0) ** 2 +
                    (pix_rgb[:, 2] / 255.0) ** 2
                )
            )
        ) if len(pix_rgb) > 0 else 0.0

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
            "color_density": color_density
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

        df["anisocytosis_flag"] = np.where(
            df["combined_nucleus_variety"] > 0.5,
            "Anisocytosis",
            "Non-anisocytosis"
        )

        df["cytoplasm_vacuolization_percent"] = df["vacuole_fraction"] * 100.0
        df["cytoplasm_color_density_percent"] = df["color_density"] * 100.0
    else:
        df["variety_index"] = []
        df["intensity_variety_index"] = []
        df["combined_nucleus_variety"] = []
        df["anisocytosis_flag"] = []
        df["cytoplasm_vacuolization_percent"] = []
        df["cytoplasm_color_density_percent"] = []

    return df


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "circularity", "roundness", "eccentricity", "solidity",
        "extent", "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "granularity", "vacuole_fraction", "color_density",
        "variety_index", "intensity_variety_index", "combined_nucleus_variety"
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


def annotate_image(rgb, objects_df, labeled, nucleus_ids=None, cytoplasm_ids=None, inflam_ids=None, result_df=None, mode="Nucleus"):
    out = rgb.copy()

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            minr = int(row["bbox_minr"])
            minc = int(row["bbox_minc"])
            maxr = int(row["bbox_maxr"])
            maxc = int(row["bbox_maxc"])

            if mode == "Nucleus":
                color_val = (255, 0, 0) if row.get("anisocytosis_flag", "Non-anisocytosis") == "Anisocytosis" else (180, 180, 180)
            elif mode == "Cytoplasm":
                vac_pct = row.get("cytoplasm_vacuolization_percent", 0)
                if vac_pct > 35:
                    color_val = (0, 0, 255)
                elif vac_pct >= 15:
                    color_val = (0, 200, 255)
                else:
                    color_val = (180, 180, 180)
            else:
                infl = row.get("is_inflammation", 0)
                color_val = (255, 0, 0) if infl == 1 else (180, 180, 180)

            cv2.rectangle(out, (minc, minr), (maxc, maxr), color_val, 1)

    for lid in nucleus_ids or []:
        row = objects_df[objects_df["label_id"] == lid]
        if len(row) == 1:
            cx = int(row.iloc[0]["centroid_x"])
            cy = int(row.iloc[0]["centroid_y"])
            cv2.circle(out, (cx, cy), 7, (255, 255, 0), 2)

    for lid in cytoplasm_ids or []:
        row = objects_df[objects_df["label_id"] == lid]
        if len(row) == 1:
            cx = int(row.iloc[0]["centroid_x"])
            cy = int(row.iloc[0]["centroid_y"])
            cv2.circle(out, (cx, cy), 7, (0, 255, 255), 2)

    for lid in inflam_ids or []:
        row = objects_df[objects_df["label_id"] == lid]
        if len(row) == 1:
            cx = int(row.iloc[0]["centroid_x"])
            cy = int(row.iloc[0]["centroid_y"])
            cv2.circle(out, (cx, cy), 7, (255, 0, 255), 2)

    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    mode = st.radio(
        "Annotation mode",
        ["Inflammation", "Nucleus", "Cytoplasm"]
    )

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write(f"Nucleus samples: {len(st.session_state.nucleus_samples)}")
    st.write(f"Cytoplasm samples: {len(st.session_state.cytoplasm_samples)}")
    st.write(f"Inflammation samples: {len(st.session_state.inflam_samples)}")


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
            objects_df=st.session_state.objects_df,
            labeled=st.session_state.labeled_mask,
            nucleus_ids=st.session_state.nucleus_samples,
            cytoplasm_ids=st.session_state.cytoplasm_samples,
            inflam_ids=st.session_state.inflam_samples,
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
                if mode == "Nucleus":
                    target_list = st.session_state.nucleus_samples
                elif mode == "Cytoplasm":
                    target_list = st.session_state.cytoplasm_samples
                else:
                    target_list = st.session_state.inflam_samples

                if nearest_id not in target_list:
                    target_list.append(nearest_id)
                    st.rerun()

        st.caption("Click object examples according to the selected mode.")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Undo last"):
                if mode == "Nucleus" and len(st.session_state.nucleus_samples) > 0:
                    st.session_state.nucleus_samples.pop()
                    st.rerun()
                elif mode == "Cytoplasm" and len(st.session_state.cytoplasm_samples) > 0:
                    st.session_state.cytoplasm_samples.pop()
                    st.rerun()
                elif mode == "Inflammation" and len(st.session_state.inflam_samples) > 0:
                    st.session_state.inflam_samples.pop()
                    st.rerun()

        with c2:
            if st.button("Clear current mode"):
                if mode == "Nucleus":
                    st.session_state.nucleus_samples = []
                elif mode == "Cytoplasm":
                    st.session_state.cytoplasm_samples = []
                else:
                    st.session_state.inflam_samples = []
                st.rerun()

        with c3:
            if st.button("Show segmentation mask"):
                seg_vis = segmentation.mark_boundaries(rgb, labeled, color=(1, 1, 0))
                st.image(seg_vis, caption="Segmentation preview", use_container_width=True)

    with colR:
        st.subheader("Analysis")

        st.write(f"Nucleus annotated objects: {len(st.session_state.nucleus_samples)}")
        st.write(f"Cytoplasm annotated objects: {len(st.session_state.cytoplasm_samples)}")
        st.write(f"Inflammation annotated objects: {len(st.session_state.inflam_samples)}")

        if st.button("Run analysis"):
            result_df = st.session_state.objects_df.copy()
            X_all = result_df[feature_columns()].copy()

            # =========================================
            # NUCLEUS MODEL / SELECTION
            # =========================================
            result_df["is_nucleus"] = 0
            if len(st.session_state.nucleus_samples) >= 2:
                nucleus_train = result_df.copy()
                nucleus_train["target"] = nucleus_train["label_id"].isin(st.session_state.nucleus_samples).astype(int)

                y_train = nucleus_train["target"]
                X_train = nucleus_train[feature_columns()].copy()

                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_nucleus = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_nucleus.fit(X_train, y_train)
                pred_nucleus = clf_nucleus.predict(X_all_filled)
                result_df["is_nucleus"] = pred_nucleus
            else:
                result_df["is_nucleus"] = (
                    (result_df["area"] <= result_df["area"].quantile(0.6)) &
                    (result_df["mean_intensity"] <= result_df["mean_intensity"].quantile(0.6))
                ).astype(int)

            result_df["anisocytosis_flag"] = np.where(
                result_df["combined_nucleus_variety"] > 0.5,
                "Anisocytosis",
                "Non-anisocytosis"
            )

            # =========================================
            # CYTOPLASM MODEL / SELECTION
            # =========================================
            result_df["is_cytoplasm"] = 0
            if len(st.session_state.cytoplasm_samples) >= 2:
                cyt_train = result_df.copy()
                cyt_train["target"] = cyt_train["label_id"].isin(st.session_state.cytoplasm_samples).astype(int)

                y_train = cyt_train["target"]
                X_train = cyt_train[feature_columns()].copy()

                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_cyt = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_cyt.fit(X_train, y_train)
                pred_cyt = clf_cyt.predict(X_all_filled)
                result_df["is_cytoplasm"] = pred_cyt
            else:
                result_df["is_cytoplasm"] = (
                    result_df["area"] >= result_df["area"].quantile(0.4)
                ).astype(int)

            # =========================================
            # INFLAMMATION MODEL / SELECTION
            # =========================================
            result_df["is_inflammation"] = 0
            if len(st.session_state.inflam_samples) >= 2:
                infl_train = result_df.copy()
                infl_train["target"] = infl_train["label_id"].isin(st.session_state.inflam_samples).astype(int)

                y_train = infl_train["target"]
                X_train = infl_train[feature_columns()].copy()

                X_train = X_train.fillna(X_train.median(numeric_only=True))
                X_all_filled = X_all.fillna(X_train.median(numeric_only=True))

                clf_infl = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf_infl.fit(X_train, y_train)
                pred_infl = clf_infl.predict(X_all_filled)
                result_df["is_inflammation"] = pred_infl
            else:
                result_df["is_inflammation"] = (
                    (result_df["area"] < result_df["area"].median()) &
                    (result_df["circularity"] > 0.5).fillna(False) &
                    (result_df["mean_intensity"] < result_df["mean_intensity"].median())
                ).astype(int)

            st.session_state.result_df = result_df

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Analysis complete")

            nucleus_df = result_df[result_df["is_nucleus"] == 1].copy()
            cytoplasm_df = result_df[result_df["is_cytoplasm"] == 1].copy()
            inflam_df = result_df[result_df["is_inflammation"] == 1].copy()

            # =========================================
            # METRICS
            # =========================================
            total_area = result_df["area"].sum() if len(result_df) > 0 else 1.0
            inflam_area = inflam_df["area"].sum() if len(inflam_df) > 0 else 0.0
            inflammation_area_percent = 100 * inflam_area / total_area if total_area > 0 else 0.0

            total_fov_area = rgb.shape[0] * rgb.shape[1]
            inflammation_fov_percent = 100 * inflam_area / total_fov_area if total_fov_area > 0 else 0.0

            anisocytosis_percent = (
                100 * (nucleus_df["anisocytosis_flag"] == "Anisocytosis").mean()
                if len(nucleus_df) > 0 else 0.0
            )

            mean_nucleus_variety = nucleus_df["combined_nucleus_variety"].mean() if len(nucleus_df) > 0 else 0.0
            mean_vacuolization = cytoplasm_df["cytoplasm_vacuolization_percent"].mean() if len(cytoplasm_df) > 0 else 0.0
            mean_color_density = cytoplasm_df["cytoplasm_color_density_percent"].mean() if len(cytoplasm_df) > 0 else 0.0

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total inflammatory cells", f"{len(inflam_df)}")
                st.metric("Inflammation area / field", f"{inflammation_fov_percent:.2f}%")
            with m2:
                st.metric("Anisocytosis in nuclei", f"{anisocytosis_percent:.1f}%")
                st.metric("Mean nucleus variety", f"{mean_nucleus_variety:.3f}")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Mean cytoplasm vacuolization", f"{mean_vacuolization:.1f}%")
            with m4:
                st.metric("Mean cytoplasm color density", f"{mean_color_density:.1f}%")

            # =========================================
            # TABLES
            # =========================================
            st.write("### Nucleus analysis")
            if len(nucleus_df) > 0:
                nucleus_table = nucleus_df[[
                    "label_id",
                    "area",
                    "mean_intensity",
                    "variety_index",
                    "intensity_variety_index",
                    "combined_nucleus_variety",
                    "anisocytosis_flag"
                ]].copy()
                st.dataframe(nucleus_table, use_container_width=True)
            else:
                st.info("No nucleus objects detected.")

            st.write("### Cytoplasm analysis")
            if len(cytoplasm_df) > 0:
                cytoplasm_table = cytoplasm_df[[
                    "label_id",
                    "area",
                    "cytoplasm_color_density_percent",
                    "cytoplasm_vacuolization_percent"
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
                    "mean_intensity"
                ]].copy()
                st.dataframe(inflam_table, use_container_width=True)
            else:
                st.info("No inflammatory objects detected.")

            # =========================================
            # PLOTS
            # =========================================
            if len(nucleus_df) > 0:
                fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                vals = nucleus_df["anisocytosis_flag"].value_counts()
                ax1.bar(vals.index, vals.values)
                ax1.set_title("Nuclear anisocytosis")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)
                plt.close(fig1)

            if len(cytoplasm_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.hist(cytoplasm_df["cytoplasm_vacuolization_percent"], bins=20)
                ax2.set_title("Cytoplasm vacuolization (%)")
                ax2.set_xlabel("Percent")
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

            # =========================================
            # IMAGE OUTPUT
            # =========================================
            st.image(
                annotate_image(
                    st.session_state.preview_rgb,
                    objects_df=st.session_state.objects_df,
                    labeled=st.session_state.labeled_mask,
                    nucleus_ids=st.session_state.nucleus_samples,
                    cytoplasm_ids=st.session_state.cytoplasm_samples,
                    inflam_ids=st.session_state.inflam_samples,
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
    "Prototype only. This tool uses object-level proxy segmentation. "
    "For research-grade results, nucleus, cytoplasm, and inflammatory cells "
    "should ideally be segmented with dedicated models or stain-specific rules."
)
