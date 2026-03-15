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
    "Semi-automatic analysis of nuclei, cytoplasm vacuolization, and inflammatory cells "
    "from liver histopathology images"
)

# =========================================================
# SESSION STATE
# =========================================================
DEFAULT_KEYS = {
    "nucleus_normal_samples": [],
    "nucleus_pyknosis_samples": [],
    "cytoplasm_samples": [],
    "inflam_samples": [],
    "objects_df": None,
    "labeled_mask": None,
    "preview_rgb": None,
    "result_df": None,
    "last_uploaded_name": None,
    "binary_mask": None,
}

for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# HELPERS
# =========================================================
def reset_all():
    for k, v in DEFAULT_KEYS.items():
        st.session_state[k] = v


def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))


def make_display_image(rgb, max_width=900):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    resized = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_NEAREST)
    return resized, scale


def preprocess_and_segment(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # sedikit blur agar threshold stabil
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold umum
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary_bool = binary > 0
    binary_bool = morphology.remove_small_objects(binary_bool, min_size=40)
    binary_bool = morphology.remove_small_holes(binary_bool, area_threshold=40)

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
        if np.sum(mask) == 0:
            continue

        gray_patch = gray[minr:maxr, minc:maxc]
        rgb_patch = rgb[minr:maxr, minc:maxc]
        hsv_patch = hsv[minr:maxr, minc:maxc]
        lab_patch = lab[minr:maxr, minc:maxc]

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

        # -----------------------------
        # Vacuolization: putih/kosong vs total area segmen
        # -----------------------------
        # Untuk H&E area vakuola cenderung putih/pucat
        white_mask = (
            (pix_rgb[:, 0] > 205) &
            (pix_rgb[:, 1] > 205) &
            (pix_rgb[:, 2] > 205)
        )

        pale_mask = (
            (pix_lab[:, 0] > 78) &
            (np.abs(pix_lab[:, 1]) < 8) &
            (np.abs(pix_lab[:, 2]) < 8)
        )

        white_empty_mask = white_mask | pale_mask
        white_pixel_percent = float(np.mean(white_empty_mask) * 100.0) if len(white_empty_mask) > 0 else 0.0

        # -----------------------------
        # Pink density untuk sitoplasma
        # makin pink makin tinggi eosin density
        # -----------------------------
        pink_mask = (
            (pix_rgb[:, 0] > pix_rgb[:, 1]) &
            (pix_rgb[:, 0] > pix_rgb[:, 2]) &
            (pix_rgb[:, 0] > 140)
        )
        pink_density_percent = float(np.mean(pink_mask) * 100.0) if len(pink_mask) > 0 else 0.0

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
            "white_pixel_percent": white_pixel_percent,
            "pink_density_percent": pink_density_percent
        })

    df = pd.DataFrame(records)

    if not df.empty:
        median_area = df["area"].median()
        df["size_variety_index"] = (
            np.abs(df["area"] - median_area) / median_area
            if median_area > 0 else 0.0
        )

        median_intensity = df["mean_intensity"].median()
        std_intensity_global = df["mean_intensity"].std() if len(df) > 1 else 1.0
        if std_intensity_global == 0 or np.isnan(std_intensity_global):
            std_intensity_global = 1.0

        df["intensity_variety_index"] = (
            np.abs(df["mean_intensity"] - median_intensity) / std_intensity_global
        )

        # gabungan ukuran + intensitas untuk variasi nukleus
        df["nucleus_variety_index"] = (
            0.6 * df["size_variety_index"] +
            0.4 * (df["intensity_variety_index"] / 3.0)
        )
    else:
        df["size_variety_index"] = []
        df["intensity_variety_index"] = []
        df["nucleus_variety_index"] = []

    return df


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "circularity", "roundness", "eccentricity", "solidity",
        "extent", "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b",
        "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "granularity", "white_pixel_percent", "pink_density_percent",
        "size_variety_index", "intensity_variety_index", "nucleus_variety_index"
    ]


def get_label_from_click(x, y, labeled):
    h, w = labeled.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    label_id = int(labeled[y, x])
    if label_id > 0:
        return label_id

    # kalau kena garis batas / background, cari radius kecil di sekitar titik
    radius = 6
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    patch = labeled[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if len(vals) == 0:
        return None

    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def train_binary_object_finder(result_df, positive_ids, feature_cols, prob_threshold=0.35, negative_ratio=4):
    df = result_df.copy()

    if len(positive_ids) == 0:
        return np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=float)

    df["target"] = 0
    df.loc[df["label_id"].isin(positive_ids), "target"] = 1

    pos_df = df[df["target"] == 1].copy()
    neg_df = df[df["target"] == 0].copy()

    if len(pos_df) < 2:
        # fallback heuristik kalau sampel terlalu sedikit
        probs = np.zeros(len(df), dtype=float)
        probs[df["label_id"].isin(positive_ids)] = 1.0
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    n_neg = min(len(neg_df), max(len(pos_df) * negative_ratio, len(pos_df)))
    if n_neg > 0:
        neg_train = neg_df.sample(n=n_neg, random_state=42)
        train_df = pd.concat([pos_df, neg_train], axis=0).copy()
    else:
        train_df = pos_df.copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].copy()

    X_all = df[feature_cols].copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_all = X_all.fillna(med)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_all)[:, 1]
    preds = (probs >= prob_threshold).astype(int)

    # paksa yang dianotasi tetap positif
    preds[df["label_id"].isin(positive_ids)] = 1
    probs[df["label_id"].isin(positive_ids)] = 1.0

    return preds, probs


def train_nucleus_subclassifier(nucleus_df, normal_ids, pyknosis_ids, feature_cols, prob_threshold=0.5):
    df = nucleus_df.copy()

    normal_ids = list(set(normal_ids) & set(df["label_id"].tolist()))
    pyknosis_ids = list(set(pyknosis_ids) & set(df["label_id"].tolist()))

    if len(normal_ids) == 0 and len(pyknosis_ids) == 0:
        # fallback: nukleus kecil dan gelap condong pyknosis
        area_med = df["area"].median() if len(df) > 0 else 0
        inten_med = df["mean_intensity"].median() if len(df) > 0 else 0
        df["nucleus_type"] = np.where(
            (df["area"] < area_med) & (df["mean_intensity"] < inten_med),
            "Pyknosis",
            "Normal"
        )
        probs = np.where(df["nucleus_type"] == "Pyknosis", 0.7, 0.3)
        return df["nucleus_type"].values, probs

    if len(normal_ids) == 0 or len(pyknosis_ids) == 0:
        # fallback semi-supervised kalau hanya satu kelas ada
        area_med = df["area"].median() if len(df) > 0 else 0
        inten_med = df["mean_intensity"].median() if len(df) > 0 else 0
        df["nucleus_type"] = np.where(
            (df["area"] < area_med) & (df["mean_intensity"] < inten_med),
            "Pyknosis",
            "Normal"
        )

        if len(pyknosis_ids) > 0:
            df.loc[df["label_id"].isin(pyknosis_ids), "nucleus_type"] = "Pyknosis"
        if len(normal_ids) > 0:
            df.loc[df["label_id"].isin(normal_ids), "nucleus_type"] = "Normal"

        probs = np.where(df["nucleus_type"] == "Pyknosis", 0.7, 0.3).astype(float)
        probs[df["label_id"].isin(pyknosis_ids)] = 1.0
        return df["nucleus_type"].values, probs

    df["target"] = np.nan
    df.loc[df["label_id"].isin(normal_ids), "target"] = 0
    df.loc[df["label_id"].isin(pyknosis_ids), "target"] = 1

    train_df = df[df["target"].notna()].copy()
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].astype(int).copy()

    X_all = df[feature_cols].copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_all = X_all.fillna(med)

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_all)[:, 1]
    preds = (probs >= prob_threshold).astype(int)

    types = np.where(preds == 1, "Pyknosis", "Normal")

    types[df["label_id"].isin(pyknosis_ids)] = "Pyknosis"
    types[df["label_id"].isin(normal_ids)] = "Normal"
    probs[df["label_id"].isin(pyknosis_ids)] = 1.0
    probs[df["label_id"].isin(normal_ids)] = 0.0

    return types, probs


def draw_segment_overlay(rgb, labeled, selected_ids=None, result_df=None, mode="Nucleus"):
    out = rgb.copy()

    # garis batas segment
    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    out[boundaries] = [255, 255, 0]

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            lid = int(row["label_id"])
            seg_mask = labeled == lid

            if mode == "Nucleus":
                if row.get("is_nucleus", 0) == 1:
                    cls = row.get("nucleus_type", "Normal")
                    color_val = np.array([255, 0, 0], dtype=np.uint8) if cls == "Pyknosis" else np.array([0, 255, 255], dtype=np.uint8)
                    out[seg_mask] = (0.75 * out[seg_mask] + 0.25 * color_val).astype(np.uint8)

            elif mode == "Cytoplasm":
                if row.get("is_cytoplasm", 0) == 1:
                    vac = row.get("white_pixel_percent", 0.0)
                    if vac > 35:
                        color_val = np.array([255, 0, 0], dtype=np.uint8)
                    elif vac >= 15:
                        color_val = np.array([0, 200, 255], dtype=np.uint8)
                    else:
                        color_val = np.array([180, 180, 180], dtype=np.uint8)
                    out[seg_mask] = (0.75 * out[seg_mask] + 0.25 * color_val).astype(np.uint8)

            else:
                if row.get("is_inflammation", 0) == 1:
                    color_val = np.array([255, 0, 255], dtype=np.uint8)
                    out[seg_mask] = (0.75 * out[seg_mask] + 0.25 * color_val).astype(np.uint8)

    if selected_ids:
        for lid in selected_ids:
            mask = labeled == lid
            boundary = segmentation.find_boundaries(mask, mode="outer")
            out[boundary] = [0, 255, 0]

    return out


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    mode = st.radio(
        "Annotation mode",
        ["Nucleus", "Cytoplasm", "Inflammation"]
    )

    nucleus_label_mode = None
    if mode == "Nucleus":
        nucleus_label_mode = st.radio(
            "Nucleus label",
            ["Normal", "Pyknosis"]
        )

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write(f"Nucleus normal samples: {len(st.session_state.nucleus_normal_samples)}")
    st.write(f"Nucleus pyknosis samples: {len(st.session_state.nucleus_pyknosis_samples)}")
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
        st.session_state.binary_mask = None

    pil_img = Image.open(uploaded)
    rgb = pil_to_rgb_array(pil_img)

    gray, binary_mask, labeled = preprocess_and_segment(rgb)
    objects_df = extract_object_features(rgb, gray, labeled)

    valid_ids = set(objects_df["label_id"].tolist()) if not objects_df.empty else set()

    st.session_state.nucleus_normal_samples = [x for x in st.session_state.nucleus_normal_samples if x in valid_ids]
    st.session_state.nucleus_pyknosis_samples = [x for x in st.session_state.nucleus_pyknosis_samples if x in valid_ids]
    st.session_state.cytoplasm_samples = [x for x in st.session_state.cytoplasm_samples if x in valid_ids]
    st.session_state.inflam_samples = [x for x in st.session_state.inflam_samples if x in valid_ids]

    st.session_state.objects_df = objects_df
    st.session_state.labeled_mask = labeled
    st.session_state.preview_rgb = rgb
    st.session_state.binary_mask = binary_mask

    colL, colR = st.columns([1.45, 1])

    with colL:
        st.subheader("Annotation")

        if mode == "Nucleus":
            selected_ids = st.session_state.nucleus_normal_samples + st.session_state.nucleus_pyknosis_samples
        elif mode == "Cytoplasm":
            selected_ids = st.session_state.cytoplasm_samples
        else:
            selected_ids = st.session_state.inflam_samples

        preview = draw_segment_overlay(
            rgb=st.session_state.preview_rgb,
            labeled=st.session_state.labeled_mask,
            selected_ids=selected_ids,
            result_df=st.session_state.result_df,
            mode=mode
        )

        display_img, scale = make_display_image(preview, max_width=850)
        click = streamlit_image_coordinates(display_img, key=f"img_click_{mode}")

        if click is not None:
            real_x = int(round(click["x"] / scale))
            real_y = int(round(click["y"] / scale))

            picked_id = get_label_from_click(real_x, real_y, st.session_state.labeled_mask)

            if picked_id is not None and picked_id in valid_ids:
                if mode == "Nucleus":
                    if nucleus_label_mode == "Normal":
                        if picked_id not in st.session_state.nucleus_normal_samples:
                            st.session_state.nucleus_normal_samples.append(picked_id)
                        if picked_id in st.session_state.nucleus_pyknosis_samples:
                            st.session_state.nucleus_pyknosis_samples.remove(picked_id)
                    else:
                        if picked_id not in st.session_state.nucleus_pyknosis_samples:
                            st.session_state.nucleus_pyknosis_samples.append(picked_id)
                        if picked_id in st.session_state.nucleus_normal_samples:
                            st.session_state.nucleus_normal_samples.remove(picked_id)

                elif mode == "Cytoplasm":
                    if picked_id not in st.session_state.cytoplasm_samples:
                        st.session_state.cytoplasm_samples.append(picked_id)

                else:
                    if picked_id not in st.session_state.inflam_samples:
                        st.session_state.inflam_samples.append(picked_id)

                st.rerun()

        st.caption("Klik langsung pada segmen objek. Label akan diambil dari pixel segmen yang diklik, bukan centroid terdekat.")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Undo last"):
                if mode == "Nucleus":
                    if nucleus_label_mode == "Normal" and len(st.session_state.nucleus_normal_samples) > 0:
                        st.session_state.nucleus_normal_samples.pop()
                        st.rerun()
                    elif nucleus_label_mode == "Pyknosis" and len(st.session_state.nucleus_pyknosis_samples) > 0:
                        st.session_state.nucleus_pyknosis_samples.pop()
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
                    if nucleus_label_mode == "Normal":
                        st.session_state.nucleus_normal_samples = []
                    else:
                        st.session_state.nucleus_pyknosis_samples = []
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

        st.write(f"Nucleus normal annotated: {len(st.session_state.nucleus_normal_samples)}")
        st.write(f"Nucleus pyknosis annotated: {len(st.session_state.nucleus_pyknosis_samples)}")
        st.write(f"Cytoplasm annotated: {len(st.session_state.cytoplasm_samples)}")
        st.write(f"Inflammation annotated: {len(st.session_state.inflam_samples)}")

        if st.button("Run analysis"):
            result_df = st.session_state.objects_df.copy()
            feat_cols = feature_columns()

            # =====================================================
            # 1. FIND ALL NUCLEI-LIKE OBJECTS
            # =====================================================
            nucleus_seed_ids = list(set(
                st.session_state.nucleus_normal_samples + st.session_state.nucleus_pyknosis_samples
            ))

            nucleus_pred, nucleus_prob = train_binary_object_finder(
                result_df=result_df,
                positive_ids=nucleus_seed_ids,
                feature_cols=feat_cols,
                prob_threshold=0.35,
                negative_ratio=4
            )
            result_df["is_nucleus"] = nucleus_pred
            result_df["nucleus_probability"] = nucleus_prob

            # subclass nucleus into normal vs pyknosis
            nucleus_df_tmp = result_df[result_df["is_nucleus"] == 1].copy()
            if len(nucleus_df_tmp) > 0:
                nuc_types, nuc_pyknosis_prob = train_nucleus_subclassifier(
                    nucleus_df=nucleus_df_tmp,
                    normal_ids=st.session_state.nucleus_normal_samples,
                    pyknosis_ids=st.session_state.nucleus_pyknosis_samples,
                    feature_cols=feat_cols,
                    prob_threshold=0.5
                )
                nucleus_df_tmp["nucleus_type"] = nuc_types
                nucleus_df_tmp["pyknosis_probability"] = nuc_pyknosis_prob

                result_df["nucleus_type"] = "Normal"
                result_df["pyknosis_probability"] = 0.0
                result_df.loc[nucleus_df_tmp.index, "nucleus_type"] = nucleus_df_tmp["nucleus_type"].values
                result_df.loc[nucleus_df_tmp.index, "pyknosis_probability"] = nucleus_df_tmp["pyknosis_probability"].values
            else:
                result_df["nucleus_type"] = "Normal"
                result_df["pyknosis_probability"] = 0.0

            # =====================================================
            # 2. FIND ALL CYTOPLASM-LIKE OBJECTS
            # =====================================================
            cyt_pred, cyt_prob = train_binary_object_finder(
                result_df=result_df,
                positive_ids=st.session_state.cytoplasm_samples,
                feature_cols=feat_cols,
                prob_threshold=0.35,
                negative_ratio=4
            )
            result_df["is_cytoplasm"] = cyt_pred
            result_df["cytoplasm_probability"] = cyt_prob

            # =====================================================
            # 3. FIND ALL INFLAMMATION-LIKE OBJECTS
            # =====================================================
            infl_pred, infl_prob = train_binary_object_finder(
                result_df=result_df,
                positive_ids=st.session_state.inflam_samples,
                feature_cols=feat_cols,
                prob_threshold=0.30,
                negative_ratio=5
            )
            result_df["is_inflammation"] = infl_pred
            result_df["inflammation_probability"] = infl_prob

            st.session_state.result_df = result_df
            st.rerun()

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Analysis complete")

            nucleus_df = result_df[result_df["is_nucleus"] == 1].copy()
            cytoplasm_df = result_df[result_df["is_cytoplasm"] == 1].copy()
            inflam_df = result_df[result_df["is_inflammation"] == 1].copy()

            # =====================================================
            # METRICS
            # =====================================================
            total_fov_area = rgb.shape[0] * rgb.shape[1]

            inflam_area = inflam_df["area"].sum() if len(inflam_df) > 0 else 0.0
            inflammation_fov_percent = 100 * inflam_area / total_fov_area if total_fov_area > 0 else 0.0

            pyknosis_percent = (
                100 * (nucleus_df["nucleus_type"] == "Pyknosis").mean()
                if len(nucleus_df) > 0 else 0.0
            )

            mean_nucleus_variety = nucleus_df["nucleus_variety_index"].mean() if len(nucleus_df) > 0 else 0.0
            mean_vacuolization = cytoplasm_df["white_pixel_percent"].mean() if len(cytoplasm_df) > 0 else 0.0
            mean_pink_density = cytoplasm_df["pink_density_percent"].mean() if len(cytoplasm_df) > 0 else 0.0

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total inflammatory cells", f"{len(inflam_df)}")
                st.metric("Inflammation area / field", f"{inflammation_fov_percent:.2f}%")
            with m2:
                st.metric("Pyknosis / total nuclei", f"{pyknosis_percent:.1f}%")
                st.metric("Mean nucleus variety", f"{mean_nucleus_variety:.3f}")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Mean vacuolization", f"{mean_vacuolization:.1f}%")
            with m4:
                st.metric("Mean pink density", f"{mean_pink_density:.1f}%")

            # =====================================================
            # TABLES
            # =====================================================
            st.write("### Nucleus analysis")
            if len(nucleus_df) > 0:
                nucleus_table = nucleus_df[[
                    "label_id",
                    "area",
                    "mean_intensity",
                    "size_variety_index",
                    "intensity_variety_index",
                    "nucleus_variety_index",
                    "nucleus_type",
                    "nucleus_probability",
                    "pyknosis_probability"
                ]].copy()
                st.dataframe(nucleus_table, use_container_width=True)
            else:
                st.info("No nucleus-like objects detected.")

            st.write("### Cytoplasm analysis")
            if len(cytoplasm_df) > 0:
                cytoplasm_table = cytoplasm_df[[
                    "label_id",
                    "area",
                    "white_pixel_percent",
                    "pink_density_percent",
                    "cytoplasm_probability"
                ]].copy()
                cytoplasm_table = cytoplasm_table.rename(columns={
                    "white_pixel_percent": "vacuolization_percent"
                })
                st.dataframe(cytoplasm_table, use_container_width=True)
            else:
                st.info("No cytoplasm-like objects detected.")

            st.write("### Inflammation analysis")
            if len(inflam_df) > 0:
                inflam_table = inflam_df[[
                    "label_id",
                    "area",
                    "circularity",
                    "mean_intensity",
                    "inflammation_probability"
                ]].copy()
                st.dataframe(inflam_table, use_container_width=True)
            else:
                st.info("No inflammation-like objects detected.")

            # =====================================================
            # PLOTS
            # =====================================================
            if len(nucleus_df) > 0:
                fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                vals = nucleus_df["nucleus_type"].value_counts()
                ax1.bar(vals.index, vals.values)
                ax1.set_title("Nucleus type")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)
                plt.close(fig1)

            if len(cytoplasm_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.hist(cytoplasm_df["white_pixel_percent"], bins=20)
                ax2.set_title("Cytoplasm vacuolization (%)")
                ax2.set_xlabel("Percent white/empty pixels")
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

            # =====================================================
            # IMAGE OUTPUT
            # =====================================================
            st.image(
                draw_segment_overlay(
                    rgb=st.session_state.preview_rgb,
                    labeled=st.session_state.labeled_mask,
                    selected_ids=[],
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
    "Prototype only. Object detection depends on segmentation quality and annotation quality. "
    "For research-grade quantification, stain-specific calibration and dedicated segmentation models are still recommended."
)
