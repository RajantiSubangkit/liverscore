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
st.set_page_config(page_title="iRATco Liver Histo Trainer", layout="wide")
st.title("iRATco Liver Histo Trainer")
st.caption("Upload image → annotate example objects → classify all segmented objects")

# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    defaults = {
        "image_name": None,
        "rgb": None,
        "gray": None,
        "labels": None,
        "mask": None,
        "objects_df": None,
        "result_df": None,
        "train_nucleus": [],
        "train_cytoplasm": [],
        "train_inflammation": [],
        "click_nucleus": [],
        "click_cytoplasm": [],
        "click_inflammation": [],
        "click_nonce": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()

# =========================================================
# IMAGE / SEGMENTATION HELPERS
# =========================================================
def pil_to_rgb(pil_img):
    return np.array(pil_img.convert("RGB"))

def resize_for_display(rgb, max_width=900):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    nh = int(h * scale)
    disp = cv2.resize(rgb, (max_width, nh), interpolation=cv2.INTER_NEAREST)
    return disp, scale

def segment_objects(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # foreground extraction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = th > 0

    # cleanup
    mask = morphology.remove_small_objects(mask, min_size=35)
    mask = morphology.remove_small_holes(mask, area_threshold=35)

    # watershed split
    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    coords = feature.peak_local_max(dist, min_distance=7, labels=mask)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if len(coords) == 0:
        labels = measure.label(mask)
    else:
        markers = morphology.dilation(markers, morphology.disk(2))
        labels = segmentation.watershed(-dist, markers, mask=mask)

    return gray, mask, labels

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(rgb, gray, labels):
    props = measure.regionprops(labels, intensity_image=gray)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(rgb)

    rows = []

    for prop in props:
        if prop.area < 30:
            continue

        minr, minc, maxr, maxc = prop.bbox
        obj_mask = labels[minr:maxr, minc:maxc] == prop.label
        if obj_mask.sum() == 0:
            continue

        gray_patch = gray[minr:maxr, minc:maxc]
        rgb_patch = rgb[minr:maxr, minc:maxc]
        hsv_patch = hsv[minr:maxr, minc:maxc]
        lab_patch = lab[minr:maxr, minc:maxc]

        pix_gray = gray_patch[obj_mask]
        pix_rgb = rgb_patch[obj_mask]
        pix_hsv = hsv_patch[obj_mask]
        pix_lab = lab_patch[obj_mask]

        perimeter = prop.perimeter if prop.perimeter > 0 else np.nan
        major_axis = prop.major_axis_length if prop.major_axis_length > 0 else np.nan
        minor_axis = prop.minor_axis_length if prop.minor_axis_length > 0 else np.nan

        circularity = 4 * np.pi * prop.area / (perimeter ** 2) if perimeter and perimeter > 0 else np.nan
        roundness = 4 * prop.area / (np.pi * (major_axis ** 2)) if major_axis and major_axis > 0 else np.nan
        aspect_ratio = major_axis / minor_axis if minor_axis and minor_axis > 0 else np.nan

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

        # white empty / vacuole proxy
        white_mask = (
            (pix_rgb[:, 0] > 210) &
            (pix_rgb[:, 1] > 210) &
            (pix_rgb[:, 2] > 210)
        )
        white_fraction = float(np.mean(white_mask)) if len(white_mask) > 0 else 0.0

        # pink cytoplasm proxy
        pink_mask = (
            (pix_rgb[:, 0] > 150) &
            (pix_rgb[:, 1] > 80) &
            (pix_rgb[:, 2] > 90) &
            ((pix_rgb[:, 0] - pix_rgb[:, 2]) > 5)
        )
        pink_fraction = float(np.mean(pink_mask)) if len(pink_mask) > 0 else 0.0

        # purple/blue dense nucleus proxy
        purple_mask = (
            (pix_rgb[:, 2] > pix_rgb[:, 0] * 0.9) &
            (pix_rgb[:, 2] > pix_rgb[:, 1] * 0.9) &
            (pix_hsv[:, 1] > 40) &
            (pix_hsv[:, 2] < 190)
        )
        purple_fraction = float(np.mean(purple_mask)) if len(purple_mask) > 0 else 0.0

        cy, cx = prop.centroid

        rows.append({
            "label_id": int(prop.label),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "area": float(prop.area),
            "perimeter": float(perimeter) if not np.isnan(perimeter) else np.nan,
            "major_axis_length": float(major_axis) if not np.isnan(major_axis) else np.nan,
            "minor_axis_length": float(minor_axis) if not np.isnan(minor_axis) else np.nan,
            "eccentricity": float(prop.eccentricity) if prop.eccentricity is not None else np.nan,
            "solidity": float(prop.solidity) if prop.solidity is not None else np.nan,
            "extent": float(prop.extent) if prop.extent is not None else np.nan,
            "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
            "roundness": float(roundness) if not np.isnan(roundness) else np.nan,
            "aspect_ratio": float(aspect_ratio) if not np.isnan(aspect_ratio) else np.nan,
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
            "white_fraction": white_fraction,
            "pink_fraction": pink_fraction,
            "purple_fraction": purple_fraction
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        med_area = df["area"].median()
        med_int = df["mean_intensity"].median()
        std_int = df["mean_intensity"].std()
        if pd.isna(std_int) or std_int == 0:
            std_int = 1.0

        df["size_variety"] = np.abs(df["area"] - med_area) / med_area if med_area > 0 else 0.0
        df["intensity_variety"] = np.abs(df["mean_intensity"] - med_int) / std_int
        df["nucleus_variety_score"] = 0.6 * df["size_variety"] + 0.4 * (df["intensity_variety"] / 3.0)

        # piknosis: nucleus makin gelap/padat dan cenderung kecil
        area_q50 = df["area"].median()
        intensity_q40 = df["mean_intensity"].quantile(0.40)
        purple_q60 = df["purple_fraction"].quantile(0.60)

        df["nucleus_pyknosis_flag"] = np.where(
            (df["area"] < area_q50) &
            (df["mean_intensity"] < intensity_q40) &
            (df["purple_fraction"] >= purple_q60),
            "Pyknosis",
            "Normal"
        )

        df["cytoplasm_vacuolization_percent"] = df["white_fraction"] * 100.0
        df["cytoplasm_pink_density_percent"] = df["pink_fraction"] * 100.0
    else:
        df["size_variety"] = []
        df["intensity_variety"] = []
        df["nucleus_variety_score"] = []
        df["nucleus_pyknosis_flag"] = []
        df["cytoplasm_vacuolization_percent"] = []
        df["cytoplasm_pink_density_percent"] = []

    return df

def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "eccentricity", "solidity", "extent", "circularity", "roundness",
        "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b",
        "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "white_fraction", "pink_fraction", "purple_fraction",
        "size_variety", "intensity_variety", "nucleus_variety_score"
    ]

# =========================================================
# CLICK / ANNOTATION
# =========================================================
def get_label_from_click(x, y, labels):
    h, w = labels.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0

    val = int(labels[y, x])
    if val > 0:
        return val

    r = 8
    y0, y1 = max(0, y-r), min(h, y+r+1)
    x0, x1 = max(0, x-r), min(w, x+r+1)
    patch = labels[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if len(vals) == 0:
        return 0
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])

def draw_clicks(rgb):
    out = rgb.copy()

    for p in st.session_state.click_nucleus:
        cv2.drawMarker(out, (p["x"], p["y"]), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    for p in st.session_state.click_cytoplasm:
        cv2.drawMarker(out, (p["x"], p["y"]), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    for p in st.session_state.click_inflammation:
        cv2.drawMarker(out, (p["x"], p["y"]), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)

    return out

def draw_seg_boundaries(rgb, labels):
    vis = segmentation.mark_boundaries(rgb, labels, color=(1, 1, 0))
    vis = (vis * 255).astype(np.uint8) if vis.dtype != np.uint8 else vis
    return vis

def draw_result_overlay(rgb, labels, result_df):
    out = rgb.copy()
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    out[boundaries] = [255, 255, 0]

    if result_df is None or len(result_df) == 0:
        return out

    for _, row in result_df.iterrows():
        lid = int(row["label_id"])
        mask = labels == lid

        if row["pred_class"] == "Nucleus":
            fill = np.array([0, 255, 255], dtype=np.uint8)
        elif row["pred_class"] == "Cytoplasm":
            fill = np.array([0, 255, 0], dtype=np.uint8)
        elif row["pred_class"] == "Inflammation":
            fill = np.array([255, 0, 255], dtype=np.uint8)
        else:
            continue

        out[mask] = (0.78 * out[mask] + 0.22 * fill).astype(np.uint8)

    return out

# =========================================================
# CLASSIFICATION
# =========================================================
def apply_rule_based_classification(df):
    result = df.copy()
    result["pred_class"] = "Unclassified"
    result["pred_prob"] = 0.50

    if len(result) == 0:
        return result

    area_q35 = result["area"].quantile(0.35)
    area_q60 = result["area"].quantile(0.60)
    circ_q60 = result["circularity"].quantile(0.60) if result["circularity"].notna().any() else 0.5
    intensity_q45 = result["mean_intensity"].quantile(0.45)
    pink_q55 = result["pink_fraction"].quantile(0.55)
    purple_q55 = result["purple_fraction"].quantile(0.55)
    white_q60 = result["white_fraction"].quantile(0.60)

    # inflammation: kecil, bulat, gelap
    infl_mask = (
        (result["area"] <= area_q35) &
        (result["mean_intensity"] <= intensity_q45) &
        (result["circularity"].fillna(0) >= circ_q60)
    )

    # nucleus: dominan ungu/gelap, umumnya lebih kecil dari cytoplasm
    nuc_mask = (
        (result["purple_fraction"] >= purple_q55) &
        (result["area"] <= area_q60)
    ) & (~infl_mask)

    # cytoplasm: lebih besar, pink, dan/atau punya white empty space
    cyt_mask = (
        (result["area"] > area_q35) &
        (
            (result["pink_fraction"] >= pink_q55) |
            (result["white_fraction"] >= white_q60)
        )
    ) & (~infl_mask) & (~nuc_mask)

    result.loc[infl_mask, "pred_class"] = "Inflammation"
    result.loc[nuc_mask, "pred_class"] = "Nucleus"
    result.loc[cyt_mask, "pred_class"] = "Cytoplasm"

    return result

def apply_ml_classification(df, nuc_ids, cyt_ids, infl_ids):
    result = df.copy()

    train_rows = []
    for lid in nuc_ids:
        train_rows.append((lid, "Nucleus"))
    for lid in cyt_ids:
        train_rows.append((lid, "Cytoplasm"))
    for lid in infl_ids:
        train_rows.append((lid, "Inflammation"))

    if len(train_rows) == 0:
        return apply_rule_based_classification(result)

    train_df = pd.DataFrame(train_rows, columns=["label_id", "target"]).drop_duplicates("label_id")
    train_df = train_df.merge(result, on="label_id", how="inner")

    # fallback kalau kelas masih terlalu sedikit
    if len(train_df) < 3 or train_df["target"].nunique() < 2:
        rb = apply_rule_based_classification(result)

        # paksa label hasil anotasi user
        force_map = {lid: "Nucleus" for lid in nuc_ids}
        force_map.update({lid: "Cytoplasm" for lid in cyt_ids})
        force_map.update({lid: "Inflammation" for lid in infl_ids})

        for lid, cls in force_map.items():
            idx = rb.index[rb["label_id"] == lid]
            if len(idx) > 0:
                rb.loc[idx, "pred_class"] = cls
                rb.loc[idx, "pred_prob"] = 1.0
        return rb

    feat_cols = feature_columns()
    X_train = train_df[feat_cols].copy()
    y_train = train_df["target"].copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_all = result[feat_cols].copy().fillna(med)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_all)
    prob = clf.predict_proba(X_all).max(axis=1)

    result["pred_class"] = pred
    result["pred_prob"] = prob

    force_map = {lid: "Nucleus" for lid in nuc_ids}
    force_map.update({lid: "Cytoplasm" for lid in cyt_ids})
    force_map.update({lid: "Inflammation" for lid in infl_ids})

    for lid, cls in force_map.items():
        idx = result.index[result["label_id"] == lid]
        if len(idx) > 0:
            result.loc[idx, "pred_class"] = cls
            result.loc[idx, "pred_prob"] = 1.0

    return result

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Annotation class", ["Nucleus", "Cytoplasm", "Inflammation"])

    show_seg = st.checkbox("Show segmentation boundary", value=False)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    st.write(f"Nucleus train: {len(st.session_state.train_nucleus)}")
    st.write(f"Cytoplasm train: {len(st.session_state.train_cytoplasm)}")
    st.write(f"Inflammation train: {len(st.session_state.train_inflammation)}")

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    if st.session_state.image_name != uploaded.name:
        rgb = pil_to_rgb(Image.open(uploaded))
        gray, mask, labels = segment_objects(rgb)
        objects_df = extract_features(rgb, gray, labels)

        st.session_state.image_name = uploaded.name
        st.session_state.rgb = rgb
        st.session_state.gray = gray
        st.session_state.mask = mask
        st.session_state.labels = labels
        st.session_state.objects_df = objects_df
        st.session_state.result_df = None
        st.session_state.train_nucleus = []
        st.session_state.train_cytoplasm = []
        st.session_state.train_inflammation = []
        st.session_state.click_nucleus = []
        st.session_state.click_cytoplasm = []
        st.session_state.click_inflammation = []
        st.session_state.click_nonce = 0

    rgb = st.session_state.rgb
    labels = st.session_state.labels
    objects_df = st.session_state.objects_df

    colL, colR = st.columns([1.35, 1])

    with colL:
        st.subheader("Annotation")

        preview = draw_clicks(rgb)
        if show_seg:
            preview = draw_seg_boundaries(preview, labels)

        disp_img, scale = resize_for_display(preview, max_width=900)

        click = streamlit_image_coordinates(disp_img, key=f"img_click_{st.session_state.click_nonce}")

        if click is not None:
            real_x = int(round(click["x"] / scale))
            real_y = int(round(click["y"] / scale))
            lid = get_label_from_click(real_x, real_y, labels)

            if lid > 0:
                if mode == "Nucleus":
                    if lid not in st.session_state.train_nucleus:
                        st.session_state.train_nucleus.append(lid)
                    st.session_state.click_nucleus.append({"x": real_x, "y": real_y, "label_id": lid})

                elif mode == "Cytoplasm":
                    if lid not in st.session_state.train_cytoplasm:
                        st.session_state.train_cytoplasm.append(lid)
                    st.session_state.click_cytoplasm.append({"x": real_x, "y": real_y, "label_id": lid})

                else:
                    if lid not in st.session_state.train_inflammation:
                        st.session_state.train_inflammation.append(lid)
                    st.session_state.click_inflammation.append({"x": real_x, "y": real_y, "label_id": lid})

                st.session_state.click_nonce += 1
                st.rerun()

        st.caption("Klik titik objek untuk menjadikannya data training sesuai kelas anotasi.")

        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("Undo last"):
                if mode == "Nucleus" and len(st.session_state.click_nucleus) > 0:
                    last = st.session_state.click_nucleus.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_nucleus):
                        if lid in st.session_state.train_nucleus:
                            st.session_state.train_nucleus.remove(lid)
                    st.rerun()

                elif mode == "Cytoplasm" and len(st.session_state.click_cytoplasm) > 0:
                    last = st.session_state.click_cytoplasm.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_cytoplasm):
                        if lid in st.session_state.train_cytoplasm:
                            st.session_state.train_cytoplasm.remove(lid)
                    st.rerun()

                elif mode == "Inflammation" and len(st.session_state.click_inflammation) > 0:
                    last = st.session_state.click_inflammation.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_inflammation):
                        if lid in st.session_state.train_inflammation:
                            st.session_state.train_inflammation.remove(lid)
                    st.rerun()

        with b2:
            if st.button("Clear current class"):
                if mode == "Nucleus":
                    st.session_state.train_nucleus = []
                    st.session_state.click_nucleus = []
                elif mode == "Cytoplasm":
                    st.session_state.train_cytoplasm = []
                    st.session_state.click_cytoplasm = []
                else:
                    st.session_state.train_inflammation = []
                    st.session_state.click_inflammation = []
                st.rerun()

        with b3:
            if st.button("Run classification"):
                result_df = apply_ml_classification(
                    objects_df,
                    st.session_state.train_nucleus,
                    st.session_state.train_cytoplasm,
                    st.session_state.train_inflammation
                )
                st.session_state.result_df = result_df
                st.rerun()

    with colR:
        st.subheader("Analysis")
        st.write(f"Total segmented objects: {len(objects_df)}")

        if st.session_state.result_df is None:
            st.info("Tambahkan anotasi lalu klik Run classification.")
        else:
            result_df = st.session_state.result_df.copy()
            st.success("Classification complete")

            nucleus_df = result_df[result_df["pred_class"] == "Nucleus"].copy()
            cytoplasm_df = result_df[result_df["pred_class"] == "Cytoplasm"].copy()
            inflam_df = result_df[result_df["pred_class"] == "Inflammation"].copy()

            # nucleus metrics
            pyknosis_percent = (
                100 * (nucleus_df["nucleus_pyknosis_flag"] == "Pyknosis").mean()
                if len(nucleus_df) > 0 else 0.0
            )
            mean_nucleus_variety = nucleus_df["nucleus_variety_score"].mean() if len(nucleus_df) > 0 else 0.0

            # cytoplasm metrics
            mean_vacuolization = cytoplasm_df["cytoplasm_vacuolization_percent"].mean() if len(cytoplasm_df) > 0 else 0.0
            mean_pink_density = cytoplasm_df["cytoplasm_pink_density_percent"].mean() if len(cytoplasm_df) > 0 else 0.0

            # inflammation metrics
            total_area = result_df["area"].sum() if len(result_df) > 0 else 1.0
            inflam_area = inflam_df["area"].sum() if len(inflam_df) > 0 else 0.0
            inflam_percent = 100 * inflam_area / total_area if total_area > 0 else 0.0

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total nucleus", f"{len(nucleus_df)}")
                st.metric("Pyknotic nucleus", f"{pyknosis_percent:.1f}%")
            with m2:
                st.metric("Mean nucleus variety", f"{mean_nucleus_variety:.3f}")
                st.metric("Total inflammation area", f"{inflam_percent:.2f}%")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Mean vacuolization", f"{mean_vacuolization:.1f}%")
            with m4:
                st.metric("Mean pink density", f"{mean_pink_density:.1f}%")

            st.write("### Nucleus table")
            if len(nucleus_df) > 0:
                st.dataframe(
                    nucleus_df[[
                        "label_id", "area", "mean_intensity", "purple_fraction",
                        "nucleus_variety_score", "nucleus_pyknosis_flag"
                    ]],
                    use_container_width=True
                )
            else:
                st.info("No nucleus objects.")

            st.write("### Cytoplasm table")
            if len(cytoplasm_df) > 0:
                st.dataframe(
                    cytoplasm_df[[
                        "label_id", "area", "cytoplasm_vacuolization_percent",
                        "cytoplasm_pink_density_percent"
                    ]],
                    use_container_width=True
                )
            else:
                st.info("No cytoplasm objects.")

            st.write("### Inflammation table")
            if len(inflam_df) > 0:
                st.dataframe(
                    inflam_df[[
                        "label_id", "area", "circularity", "mean_intensity"
                    ]],
                    use_container_width=True
                )
            else:
                st.info("No inflammation objects.")

            st.write("### Result overlay")
            overlay = draw_result_overlay(rgb, labels, result_df)
            st.image(overlay, use_container_width=True)

            if len(nucleus_df) > 0:
                fig1, ax1 = plt.subplots(figsize=(5, 3.2))
                vals = nucleus_df["nucleus_pyknosis_flag"].value_counts()
                ax1.bar(vals.index, vals.values)
                ax1.set_title("Nucleus normal vs pyknosis")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)
                plt.close(fig1)

            if len(cytoplasm_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(5, 3.2))
                ax2.hist(cytoplasm_df["cytoplasm_vacuolization_percent"], bins=20)
                ax2.set_title("Cytoplasm vacuolization (%)")
                ax2.set_xlabel("Percent")
                ax2.set_ylabel("Frequency")
                st.pyplot(fig2)
                plt.close(fig2)

            if len(inflam_df) > 0:
                fig3, ax3 = plt.subplots(figsize=(5, 3.2))
                ax3.hist(inflam_df["area"], bins=20)
                ax3.set_title("Inflammation area")
                ax3.set_xlabel("Area")
                ax3.set_ylabel("Frequency")
                st.pyplot(fig3)
                plt.close(fig3)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="iratco_liver_histo_result.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption(
    "Prototype. Segmentasi berbasis objek dan klasifikasi semi-otomatis dari anotasi klik."
)
