from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure, morphology, segmentation, feature
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Liver Score Trainer", layout="wide")
st.title("Liver Score Trainer")
st.caption("Upload image -> click objects as training labels -> classify all segmented objects")

# =========================================================
# SESSION STATE
# =========================================================
CLASSES = ["Nucleus", "Cytoplasm", "Inflammation", "Exclude"]

if "samples" not in st.session_state:
    st.session_state.samples = {c: [] for c in CLASSES}

if "click_marks" not in st.session_state:
    st.session_state.click_marks = []

if "objects_df" not in st.session_state:
    st.session_state.objects_df = None

if "labeled" not in st.session_state:
    st.session_state.labeled = None

if "rgb" not in st.session_state:
    st.session_state.rgb = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# =========================================================
# HELPERS
# =========================================================
def reset_all():
    st.session_state.samples = {c: [] for c in CLASSES}
    st.session_state.click_marks = []
    st.session_state.objects_df = None
    st.session_state.labeled = None
    st.session_state.rgb = None
    st.session_state.result_df = None
    st.session_state.last_file = None


def pil_to_rgb(img):
    return np.array(img.convert("RGB"))


def resize_for_display(rgb, max_width=900):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    disp = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return disp, scale


def segment_objects(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = binary > 0
    mask = morphology.remove_small_objects(mask, min_size=40)
    mask = morphology.remove_small_holes(mask, area_threshold=40)

    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    peaks = feature.peak_local_max(dist, min_distance=8, labels=mask)

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    if len(peaks) == 0:
        labeled = measure.label(mask)
    else:
        labeled = segmentation.watershed(-dist, markers, mask=mask)

    return gray, labeled


def extract_features(rgb, gray, labeled):
    rows = []
    props = measure.regionprops(labeled, intensity_image=gray)

    for p in props:
        if p.area < 40:
            continue

        minr, minc, maxr, maxc = p.bbox
        obj_mask = labeled[minr:maxr, minc:maxc] == p.label
        rgb_patch = rgb[minr:maxr, minc:maxc]
        gray_patch = gray[minr:maxr, minc:maxc]

        pix_rgb = rgb_patch[obj_mask]
        pix_gray = gray_patch[obj_mask]

        perimeter = p.perimeter if p.perimeter > 0 else np.nan
        major = p.major_axis_length if p.major_axis_length > 0 else np.nan
        minor = p.minor_axis_length if p.minor_axis_length > 0 else np.nan

        circularity = 4 * np.pi * p.area / (perimeter ** 2) if perimeter and perimeter > 0 else np.nan
        aspect_ratio = major / minor if minor and minor > 0 else np.nan

        cy, cx = p.centroid

        rows.append({
            "label_id": int(p.label),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "area": float(p.area),
            "perimeter": float(perimeter) if not np.isnan(perimeter) else np.nan,
            "major_axis_length": float(major) if not np.isnan(major) else np.nan,
            "minor_axis_length": float(minor) if not np.isnan(minor) else np.nan,
            "eccentricity": float(p.eccentricity) if p.eccentricity is not None else np.nan,
            "solidity": float(p.solidity) if p.solidity is not None else np.nan,
            "extent": float(p.extent) if p.extent is not None else np.nan,
            "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
            "aspect_ratio": float(aspect_ratio) if not np.isnan(aspect_ratio) else np.nan,
            "mean_intensity": float(np.mean(pix_gray)),
            "std_intensity": float(np.std(pix_gray)),
            "mean_r": float(np.mean(pix_rgb[:, 0])),
            "mean_g": float(np.mean(pix_rgb[:, 1])),
            "mean_b": float(np.mean(pix_rgb[:, 2]))
        })

    return pd.DataFrame(rows)


def feature_cols():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "eccentricity", "solidity", "extent", "circularity",
        "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b"
    ]


def get_clicked_label(x, y, labeled, radius=20):
    h, w = labeled.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return None

    direct = int(labeled[y, x])
    if direct > 0:
        return direct

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    patch = labeled[y0:y1, x0:x1]
    coords = np.argwhere(patch > 0)
    if len(coords) == 0:
        return None

    best_label = None
    best_dist = 1e9
    for rr, cc in coords:
        yy = y0 + rr
        xx = x0 + cc
        d = (xx - x) ** 2 + (yy - y) ** 2
        if d < best_dist:
            best_dist = d
            best_label = int(labeled[yy, xx])

    return best_label


def draw_preview(rgb, result_df=None, click_marks=None):
    out = rgb.copy()

    color_map = {
        "Nucleus": (0, 255, 0),
        "Cytoplasm": (255, 255, 0),
        "Inflammation": (255, 0, 255),
        "Exclude": (180, 180, 180)
    }

    if result_df is not None and not result_df.empty and "predicted_class" in result_df.columns:
        for _, row in result_df.iterrows():
            cls = row["predicted_class"]
            color = color_map.get(cls, (255, 255, 255))
            cv2.rectangle(
                out,
                (int(row["bbox_minc"]), int(row["bbox_minr"])),
                (int(row["bbox_maxc"]), int(row["bbox_maxr"])),
                color,
                1
            )

    if click_marks is not None:
        for p in click_marks:
            color = color_map.get(p["class"], (255, 255, 255))
            cv2.drawMarker(
                out,
                (int(p["x"]), int(p["y"])),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2
            )

    return out


def build_train_df(objects_df, samples):
    rows = []
    for cls, ids in samples.items():
        if cls == "Exclude":
            continue
        for lid in ids:
            r = objects_df[objects_df["label_id"] == lid]
            if len(r) == 1:
                x = r.iloc[0].copy()
                x["target_class"] = cls
                rows.append(x)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")
    active_class = st.selectbox("Annotation class", CLASSES)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    for c in CLASSES:
        st.write(f"{c}: {len(st.session_state.samples[c])}")

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload liver histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is not None:
    if st.session_state.last_file != uploaded.name:
        reset_all()
        st.session_state.last_file = uploaded.name

        rgb = pil_to_rgb(Image.open(uploaded))
        gray, labeled = segment_objects(rgb)
        objects_df = extract_features(rgb, gray, labeled)

        st.session_state.rgb = rgb
        st.session_state.labeled = labeled
        st.session_state.objects_df = objects_df

    rgb = st.session_state.rgb
    labeled = st.session_state.labeled
    objects_df = st.session_state.objects_df

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Image annotation")

        preview = draw_preview(
            rgb,
            result_df=st.session_state.result_df,
            click_marks=st.session_state.click_marks
        )

        display_img, scale = resize_for_display(preview, max_width=850)
        click = streamlit_image_coordinates(display_img, key="img_click")

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)

            lid = get_clicked_label(real_x, real_y, labeled, radius=25)

            if lid is not None:
                already = any(lid in st.session_state.samples[c] for c in CLASSES)
                if not already:
                    st.session_state.samples[active_class].append(lid)
                    st.session_state.click_marks.append({
                        "x": real_x,
                        "y": real_y,
                        "class": active_class,
                        "label_id": lid
                    })
                    st.rerun()

        st.caption("Klik objek untuk jadi training sample.")

        a1, a2 = st.columns(2)
        with a1:
            if st.button("Undo last"):
                if len(st.session_state.samples[active_class]) > 0:
                    last_id = st.session_state.samples[active_class].pop()
                    for i in range(len(st.session_state.click_marks) - 1, -1, -1):
                        if (
                            st.session_state.click_marks[i]["class"] == active_class
                            and st.session_state.click_marks[i]["label_id"] == last_id
                        ):
                            st.session_state.click_marks.pop(i)
                            break
                    st.rerun()

        with a2:
            if st.button("Clear class"):
                ids_to_remove = set(st.session_state.samples[active_class])
                st.session_state.samples[active_class] = []
                st.session_state.click_marks = [
                    p for p in st.session_state.click_marks
                    if not (p["class"] == active_class and p["label_id"] in ids_to_remove)
                ]
                st.rerun()

    with col2:
        st.subheader("Training and classification")

        st.write("Detected objects:", len(objects_df))

        train_df = build_train_df(objects_df, st.session_state.samples)
        st.write("Training samples:", len(train_df))

        if not train_df.empty:
            st.dataframe(
                train_df[["label_id", "target_class"] + feature_cols()].head(20),
                use_container_width=True
            )

        if st.button("Run classification"):
            if train_df.empty:
                st.error("Belum ada training sample.")
            else:
                X_train = train_df[feature_cols()].copy()
                y_train = train_df["target_class"].copy()

                X_all = objects_df[feature_cols()].copy()

                med = X_train.median(numeric_only=True)
                X_train = X_train.fillna(med)
                X_all = X_all.fillna(med)

                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                clf.fit(X_train, y_train)

                pred = clf.predict(X_all)
                prob = clf.predict_proba(X_all).max(axis=1)

                result_df = objects_df.copy()
                result_df["predicted_class"] = pred
                result_df["confidence"] = prob

                excluded = set(st.session_state.samples["Exclude"])
                if len(excluded) > 0:
                    result_df.loc[result_df["label_id"].isin(excluded), "predicted_class"] = "Exclude"
                    result_df.loc[result_df["label_id"].isin(excluded), "confidence"] = 1.0

                st.session_state.result_df = result_df

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Classification finished")

            show_df = result_df[result_df["predicted_class"] != "Exclude"].copy()

            if not show_df.empty:
                summary_df = (
                    show_df["predicted_class"]
                    .value_counts()
                    .rename_axis("Class")
                    .reset_index(name="Count")
                )
                summary_df["Percent"] = summary_df["Count"] / summary_df["Count"].sum() * 100
                st.dataframe(summary_df, use_container_width=True)

                st.dataframe(
                    show_df[["label_id", "predicted_class", "confidence", "area", "mean_intensity"]],
                    use_container_width=True
                )

                csv = show_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="liver_classification.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.caption("Versi dasar dulu: fokus supaya upload, klik anotasi, training, dan classify berjalan stabil.")
