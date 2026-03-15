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
st.set_page_config(page_title="Histopathology Object Trainer", layout="wide")

st.title("Histopathology Object Trainer")
st.caption("Upload image, annotate example objects, then classify all segmented objects.")

# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    defaults = {
        "image_name": None,
        "rgb": None,
        "gray": None,
        "binary": None,
        "labels": None,
        "objects_df": None,
        "train_nucleus": [],
        "train_cytoplasm": [],
        "train_inflammation": [],
        "click_points_nucleus": [],
        "click_points_cytoplasm": [],
        "click_points_inflammation": [],
        "result_df": None,
        "show_segmentation": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================================================
# HELPERS
# =========================================================
def reset_all():
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    init_state()


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

    # Threshold foreground
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = th > 0

    # Clean
    mask = morphology.remove_small_objects(mask, min_size=40)
    mask = morphology.remove_small_holes(mask, area_threshold=40)

    # Distance + watershed
    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    coords = feature.peak_local_max(
        dist,
        min_distance=8,
        labels=mask
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if len(coords) == 0:
        labels = measure.label(mask)
    else:
        markers = morphology.dilation(markers, morphology.disk(2))
        labels = segmentation.watershed(-dist, markers, mask=mask)

    return gray, mask, labels


def extract_features(rgb, gray, labels):
    props = measure.regionprops(labels, intensity_image=gray)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(rgb)

    rows = []

    for prop in props:
        if prop.area < 40:
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
        aspect_ratio = major_axis / minor_axis if minor_axis and minor_axis > 0 else np.nan
        roundness = 4 * prop.area / (np.pi * (major_axis ** 2)) if major_axis and major_axis > 0 else np.nan

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

        white_fraction = float(np.mean(
            (pix_rgb[:, 0] > 210) &
            (pix_rgb[:, 1] > 210) &
            (pix_rgb[:, 2] > 210)
        ))

        pink_fraction = float(np.mean(
            (pix_rgb[:, 0] > 150) &
            (pix_rgb[:, 1] > 80) &
            (pix_rgb[:, 2] > 80) &
            ((pix_rgb[:, 0] - pix_rgb[:, 2]) > 5)
        ))

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
            "pink_fraction": pink_fraction
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        med_area = df["area"].median()
        med_int = df["mean_intensity"].median()
        std_int = df["mean_intensity"].std() if len(df) > 1 else 1.0
        if std_int == 0 or np.isnan(std_int):
            std_int = 1.0

        df["size_variety"] = np.abs(df["area"] - med_area) / med_area if med_area > 0 else 0.0
        df["intensity_variety"] = np.abs(df["mean_intensity"] - med_int) / std_int
    else:
        df["size_variety"] = []
        df["intensity_variety"] = []

    return df


def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "eccentricity", "solidity", "extent", "circularity", "roundness",
        "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b",
        "mean_h", "mean_s", "mean_v",
        "mean_l", "mean_a", "mean_b_lab",
        "white_fraction", "pink_fraction",
        "size_variety", "intensity_variety"
    ]


def get_clicked_label(x, y, labels):
    h, w = labels.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0

    label_id = int(labels[y, x])
    if label_id > 0:
        return label_id

    r = 7
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)

    patch = labels[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if len(vals) == 0:
        return 0

    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def draw_points(rgb):
    out = rgb.copy()

    for p in st.session_state.click_points_nucleus:
        cv2.drawMarker(out, (p["x"], p["y"]), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

    for p in st.session_state.click_points_cytoplasm:
        cv2.drawMarker(out, (p["x"], p["y"]), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)

    for p in st.session_state.click_points_inflammation:
        cv2.drawMarker(out, (p["x"], p["y"]), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

    return out


def draw_segmentation_overlay(rgb, labels):
    vis = segmentation.mark_boundaries(rgb, labels, color=(1, 1, 0))
    vis = (vis * 255).astype(np.uint8) if vis.dtype != np.uint8 else vis
    return vis


def draw_classification_overlay(rgb, labels, result_df):
    out = rgb.copy()

    if result_df is None or result_df.empty:
        return out

    # base boundaries
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    out[boundaries] = [255, 255, 0]

    for _, row in result_df.iterrows():
        lid = int(row["label_id"])
        mask = labels == lid

        if row["pred_class"] == "Nucleus":
            color_fill = np.array([0, 255, 255], dtype=np.uint8)
        elif row["pred_class"] == "Cytoplasm":
            color_fill = np.array([0, 255, 0], dtype=np.uint8)
        elif row["pred_class"] == "Inflammation":
            color_fill = np.array([255, 0, 255], dtype=np.uint8)
        else:
            continue

        out[mask] = (0.75 * out[mask] + 0.25 * color_fill).astype(np.uint8)

    return out


def train_and_predict_multiclass(df, nuc_ids, cyt_ids, inf_ids):
    result_df = df.copy()
    result_df["pred_class"] = "Unclassified"
    result_df["pred_prob"] = 0.0

    train_rows = []

    for lid in nuc_ids:
        train_rows.append((lid, "Nucleus"))
    for lid in cyt_ids:
        train_rows.append((lid, "Cytoplasm"))
    for lid in inf_ids:
        train_rows.append((lid, "Inflammation"))

    if len(train_rows) < 3:
        return result_df

    train_df = pd.DataFrame(train_rows, columns=["label_id", "target"]).drop_duplicates("label_id")
    train_df = train_df.merge(result_df, on="label_id", how="inner")

    if train_df["target"].nunique() < 2:
        return result_df

    feat_cols = feature_columns()

    X_train = train_df[feat_cols].copy()
    y_train = train_df["target"].copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)

    X_all = result_df[feat_cols].copy().fillna(med)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_all)
    prob = clf.predict_proba(X_all).max(axis=1)

    result_df["pred_class"] = pred
    result_df["pred_prob"] = prob

    # force training labels to stay fixed
    force_map = {lid: "Nucleus" for lid in nuc_ids}
    force_map.update({lid: "Cytoplasm" for lid in cyt_ids})
    force_map.update({lid: "Inflammation" for lid in inf_ids})

    for lid, cls in force_map.items():
        idx = result_df.index[result_df["label_id"] == lid]
        if len(idx) > 0:
            result_df.loc[idx, "pred_class"] = cls
            result_df.loc[idx, "pred_prob"] = 1.0

    return result_df


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Annotation label", ["Nucleus", "Cytoplasm", "Inflammation"])

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.write(f"Nucleus train objects: {len(st.session_state.train_nucleus)}")
    st.write(f"Cytoplasm train objects: {len(st.session_state.train_cytoplasm)}")
    st.write(f"Inflammation train objects: {len(st.session_state.train_inflammation)}")

    st.session_state.show_segmentation = st.checkbox(
        "Show segmentation boundary on preview",
        value=st.session_state.show_segmentation
    )

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
        gray, binary, labels = segment_objects(rgb)
        objects_df = extract_features(rgb, gray, labels)

        st.session_state.image_name = uploaded.name
        st.session_state.rgb = rgb
        st.session_state.gray = gray
        st.session_state.binary = binary
        st.session_state.labels = labels
        st.session_state.objects_df = objects_df
        st.session_state.result_df = None

        st.session_state.train_nucleus = []
        st.session_state.train_cytoplasm = []
        st.session_state.train_inflammation = []
        st.session_state.click_points_nucleus = []
        st.session_state.click_points_cytoplasm = []
        st.session_state.click_points_inflammation = []

    rgb = st.session_state.rgb
    labels = st.session_state.labels
    objects_df = st.session_state.objects_df

    col1, col2 = st.columns([1.45, 1])

    with col1:
        st.subheader("Image annotation")

        preview = draw_points(rgb)
        if st.session_state.show_segmentation:
            preview = draw_segmentation_overlay(preview, labels)

        disp_img, scale = resize_for_display(preview, max_width=900)
        click = streamlit_image_coordinates(disp_img, key="main_image_click")

        if click is not None:
            real_x = int(round(click["x"] / scale))
            real_y = int(round(click["y"] / scale))
            label_id = get_clicked_label(real_x, real_y, labels)

            if label_id > 0:
                if mode == "Nucleus":
                    if label_id not in st.session_state.train_nucleus:
                        st.session_state.train_nucleus.append(label_id)
                    st.session_state.click_points_nucleus.append(
                        {"x": real_x, "y": real_y, "label_id": label_id}
                    )

                elif mode == "Cytoplasm":
                    if label_id not in st.session_state.train_cytoplasm:
                        st.session_state.train_cytoplasm.append(label_id)
                    st.session_state.click_points_cytoplasm.append(
                        {"x": real_x, "y": real_y, "label_id": label_id}
                    )

                else:
                    if label_id not in st.session_state.train_inflammation:
                        st.session_state.train_inflammation.append(label_id)
                    st.session_state.click_points_inflammation.append(
                        {"x": real_x, "y": real_y, "label_id": label_id}
                    )

                st.rerun()

        st.caption("Klik objek untuk memberi label training sesuai mode anotasi.")

        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("Undo last click"):
                if mode == "Nucleus" and len(st.session_state.click_points_nucleus) > 0:
                    last = st.session_state.click_points_nucleus.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_points_nucleus):
                        if lid in st.session_state.train_nucleus:
                            st.session_state.train_nucleus.remove(lid)
                    st.rerun()

                elif mode == "Cytoplasm" and len(st.session_state.click_points_cytoplasm) > 0:
                    last = st.session_state.click_points_cytoplasm.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_points_cytoplasm):
                        if lid in st.session_state.train_cytoplasm:
                            st.session_state.train_cytoplasm.remove(lid)
                    st.rerun()

                elif mode == "Inflammation" and len(st.session_state.click_points_inflammation) > 0:
                    last = st.session_state.click_points_inflammation.pop()
                    lid = last["label_id"]
                    if not any(p["label_id"] == lid for p in st.session_state.click_points_inflammation):
                        if lid in st.session_state.train_inflammation:
                            st.session_state.train_inflammation.remove(lid)
                    st.rerun()

        with b2:
            if st.button("Clear current label"):
                if mode == "Nucleus":
                    st.session_state.train_nucleus = []
                    st.session_state.click_points_nucleus = []
                elif mode == "Cytoplasm":
                    st.session_state.train_cytoplasm = []
                    st.session_state.click_points_cytoplasm = []
                else:
                    st.session_state.train_inflammation = []
                    st.session_state.click_points_inflammation = []
                st.rerun()

        with b3:
            if st.button("Run classification"):
                result_df = train_and_predict_multiclass(
                    objects_df,
                    st.session_state.train_nucleus,
                    st.session_state.train_cytoplasm,
                    st.session_state.train_inflammation
                )
                st.session_state.result_df = result_df
                st.rerun()

    with col2:
        st.subheader("Results")

        st.write(f"Total segmented objects: {len(objects_df)}")

        if st.session_state.result_df is None:
            st.info("Belum ada hasil klasifikasi. Tambahkan anotasi lalu klik Run classification.")
        else:
            result_df = st.session_state.result_df.copy()

            nuc_df = result_df[result_df["pred_class"] == "Nucleus"].copy()
            cyt_df = result_df[result_df["pred_class"] == "Cytoplasm"].copy()
            inf_df = result_df[result_df["pred_class"] == "Inflammation"].copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Nucleus", len(nuc_df))
            c2.metric("Cytoplasm", len(cyt_df))
            c3.metric("Inflammation", len(inf_df))

            st.write("### Classified objects")
            st.dataframe(
                result_df[[
                    "label_id", "area", "mean_intensity", "circularity",
                    "white_fraction", "pink_fraction", "pred_class", "pred_prob"
                ]],
                use_container_width=True
            )

            st.write("### Classification overlay")
            overlay = draw_classification_overlay(rgb, labels, result_df)
            st.image(overlay, use_container_width=True)

            fig, ax = plt.subplots(figsize=(5, 3.2))
            vc = result_df["pred_class"].value_counts()
            ax.bar(vc.index, vc.values)
            ax.set_title("Predicted classes")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close(fig)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download result CSV",
                data=csv,
                file_name="histopathology_classification.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Tip: beri beberapa contoh klik pada tiap kelas supaya model lebih stabil.")
