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
st.set_page_config(page_title="iRATco Liver Score", layout="wide")
st.title("iRATco Liver Score")
st.caption("Annotate Nucleus and Inflammation only. Cytoplasm is assigned automatically from remaining segmented objects.")

# =========================================================
# CONSTANTS
# =========================================================
ANNOT_CLASSES = ["Nucleus", "Inflammation", "Exclude"]
FINAL_CLASSES = ["Nucleus", "Inflammation", "Cytoplasm", "Exclude"]

CLASS_COLORS = {
    "Nucleus": (0, 255, 0),
    "Inflammation": (255, 0, 255),
    "Cytoplasm": (255, 255, 0),
    "Exclude": (140, 140, 140)
}

# =========================================================
# SESSION STATE
# =========================================================
def init_state():
    if "samples" not in st.session_state:
        st.session_state.samples = {c: [] for c in ANNOT_CLASSES}
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
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None

init_state()

def reset_all():
    st.session_state.samples = {c: [] for c in ANNOT_CLASSES}
    st.session_state.click_marks = []
    st.session_state.objects_df = None
    st.session_state.labeled = None
    st.session_state.rgb = None
    st.session_state.result_df = None
    st.session_state.last_uploaded_name = None

# =========================================================
# BASIC HELPERS
# =========================================================
def pil_to_rgb_array(pil_img):
    return np.array(pil_img.convert("RGB"))

def make_display_image(rgb, max_width=850):
    h, w = rgb.shape[:2]
    if w <= max_width:
        return rgb.copy(), 1.0
    scale = max_width / w
    new_h = int(h * scale)
    disp = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return disp, scale

# =========================================================
# SEGMENTATION
# =========================================================
def preprocess_and_segment(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu dark object segmentation
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = binary > 0
    mask = morphology.remove_small_objects(mask, min_size=35)
    mask = morphology.remove_small_holes(mask, area_threshold=35)
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(1))

    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    peaks = feature.peak_local_max(
        dist,
        min_distance=7,
        labels=mask
    )

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i

    if len(peaks) == 0:
        labeled = measure.label(mask)
    else:
        labeled = segmentation.watershed(-dist, markers, mask=mask)

    labeled = measure.label(labeled > 0)
    return gray, labeled

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_object_features(rgb, gray, labeled):
    rows = []
    props = measure.regionprops(labeled, intensity_image=gray)

    for prop in props:
        if prop.area < 35:
            continue

        minr, minc, maxr, maxc = prop.bbox
        obj_mask = labeled[minr:maxr, minc:maxc] == prop.label
        rgb_patch = rgb[minr:maxr, minc:maxc]
        gray_patch = gray[minr:maxr, minc:maxc]

        pix_rgb = rgb_patch[obj_mask]
        pix_gray = gray_patch[obj_mask]

        if len(pix_gray) == 0:
            continue

        perimeter = prop.perimeter if prop.perimeter > 0 else np.nan
        major = prop.major_axis_length if prop.major_axis_length > 0 else np.nan
        minor = prop.minor_axis_length if prop.minor_axis_length > 0 else np.nan

        circularity = (
            4 * np.pi * prop.area / (perimeter ** 2)
            if perimeter and perimeter > 0 else np.nan
        )
        aspect_ratio = (
            major / minor if minor and minor > 0 else np.nan
        )

        # white/empty fraction inside segmented object
        white_fraction = float(np.mean(
            (pix_rgb[:, 0] > 210) &
            (pix_rgb[:, 1] > 210) &
            (pix_rgb[:, 2] > 210)
        ))

        # simple pink proxy
        pink_fraction = float(np.mean(
            (pix_rgb[:, 0] > pix_rgb[:, 1]) &
            (pix_rgb[:, 0] > pix_rgb[:, 2]) &
            (pix_rgb[:, 0] > 120)
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
            "major_axis_length": float(major) if not np.isnan(major) else np.nan,
            "minor_axis_length": float(minor) if not np.isnan(minor) else np.nan,
            "eccentricity": float(prop.eccentricity) if prop.eccentricity is not None else np.nan,
            "solidity": float(prop.solidity) if prop.solidity is not None else np.nan,
            "extent": float(prop.extent) if prop.extent is not None else np.nan,
            "circularity": float(circularity) if not np.isnan(circularity) else np.nan,
            "aspect_ratio": float(aspect_ratio) if not np.isnan(aspect_ratio) else np.nan,
            "mean_intensity": float(np.mean(pix_gray)),
            "std_intensity": float(np.std(pix_gray)),
            "mean_r": float(np.mean(pix_rgb[:, 0])),
            "mean_g": float(np.mean(pix_rgb[:, 1])),
            "mean_b": float(np.mean(pix_rgb[:, 2])),
            "white_fraction": white_fraction,
            "pink_fraction": pink_fraction
        })

    return pd.DataFrame(rows)

def feature_columns():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "eccentricity", "solidity", "extent", "circularity",
        "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b", "white_fraction", "pink_fraction"
    ]

# =========================================================
# CLICK TO OBJECT
# =========================================================
def get_object_from_click(x, y, labeled, search_radius=20):
    h, w = labeled.shape

    if not (0 <= x < w and 0 <= y < h):
        return None

    direct_label = int(labeled[y, x])
    if direct_label > 0:
        return direct_label

    y0 = max(0, y - search_radius)
    y1 = min(h, y + search_radius + 1)
    x0 = max(0, x - search_radius)
    x1 = min(w, x + search_radius + 1)

    patch = labeled[y0:y1, x0:x1]
    coords = np.argwhere(patch > 0)
    if len(coords) == 0:
        return None

    best_label = None
    best_dist = 1e18

    for rr, cc in coords:
        yy = y0 + rr
        xx = x0 + cc
        d = (xx - x) ** 2 + (yy - y) ** 2
        if d < best_dist:
            best_dist = d
            best_label = int(labeled[yy, xx])

    return best_label

# =========================================================
# TRAINING TABLE
# =========================================================
def build_training_table(objects_df, samples_dict):
    rows = []
    for cls, ids in samples_dict.items():
        if cls == "Exclude":
            continue
        for lid in ids:
            row = objects_df[objects_df["label_id"] == lid]
            if len(row) == 1:
                r = row.iloc[0].copy()
                r["target_class"] = cls
                rows.append(r)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# =========================================================
# VISUALIZATION
# =========================================================
def draw_annotation_preview(rgb, result_df=None, click_marks=None):
    out = rgb.copy()

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            cls = row["predicted_class"]
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(
                out,
                (int(row["bbox_minc"]), int(row["bbox_minr"])),
                (int(row["bbox_maxc"]), int(row["bbox_maxr"])),
                color,
                1
            )

    if click_marks is not None:
        for mark in click_marks:
            color = CLASS_COLORS.get(mark["class"], (255, 255, 255))
            cv2.drawMarker(
                out,
                (int(mark["x"]), int(mark["y"])),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2
            )

    return out

def make_colored_segmentation(labeled, result_df=None):
    h, w = labeled.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if result_df is not None and not result_df.empty:
        for _, row in result_df.iterrows():
            lid = int(row["label_id"])
            cls = row["predicted_class"]
            seg_rgb[labeled == lid] = CLASS_COLORS.get(cls, (255, 255, 255))
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

def show_cell_gallery(df_to_show, rgb, labeled, title, n_cols=6, max_cells=60):
    st.subheader(title)

    if df_to_show is None or df_to_show.empty:
        st.info("No objects to display.")
        return

    show_df = df_to_show.head(max_cells).copy()
    chunks = [show_df.iloc[i:i+n_cols] for i in range(0, len(show_df), n_cols)]

    for chunk in chunks:
        cols = st.columns(n_cols)
        for i in range(n_cols):
            with cols[i]:
                if i < len(chunk):
                    row = chunk.iloc[i]
                    thumb = crop_object_thumbnail(rgb, labeled, row, pad=8, target_size=96)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True)
                        cap = f"ID {int(row['label_id'])}"
                        if "predicted_class" in row.index:
                            cap += f" | {row['predicted_class']}"
                        st.caption(cap)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Controls")

    active_class = st.selectbox("Annotation class", ANNOT_CLASSES)
    show_seg_overlay = st.checkbox("Show segmentation overlay", value=True)
    show_gallery = st.checkbox("Show cell gallery", value=True)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    for c in ANNOT_CLASSES:
        st.write(f"**{c}**: {len(st.session_state.samples[c])}")

# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload liver histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded is not None:
    if st.session_state.last_uploaded_name != uploaded.name:
        reset_all()
        st.session_state.last_uploaded_name = uploaded.name

        rgb = pil_to_rgb_array(Image.open(uploaded))
        gray, labeled = preprocess_and_segment(rgb)
        objects_df = extract_object_features(rgb, gray, labeled)

        st.session_state.rgb = rgb
        st.session_state.labeled = labeled
        st.session_state.objects_df = objects_df

    rgb = st.session_state.rgb
    labeled = st.session_state.labeled
    objects_df = st.session_state.objects_df

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Image annotation")

        preview = draw_annotation_preview(
            rgb,
            result_df=st.session_state.result_df,
            click_marks=st.session_state.click_marks
        )

        display_img, scale = make_display_image(preview, max_width=850)
        click = streamlit_image_coordinates(display_img, key="main_click")

        if click is not None:
            real_x = int(click["x"] / scale)
            real_y = int(click["y"] / scale)

            label_id = get_object_from_click(real_x, real_y, labeled, search_radius=25)

            if label_id is not None:
                already_used = any(
                    label_id in st.session_state.samples[c]
                    for c in ANNOT_CLASSES
                )
                if not already_used:
                    st.session_state.samples[active_class].append(label_id)
                    st.session_state.click_marks.append({
                        "x": real_x,
                        "y": real_y,
                        "class": active_class,
                        "label_id": label_id
                    })
                    st.rerun()

        st.caption("Klik objek hasil segmentasi untuk anotasi Nucleus atau Inflammation.")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Undo last"):
                if len(st.session_state.samples[active_class]) > 0:
                    last_id = st.session_state.samples[active_class].pop()
                    for i in range(len(st.session_state.click_marks) - 1, -1, -1):
                        if (
                            st.session_state.click_marks[i]["class"] == active_class and
                            st.session_state.click_marks[i]["label_id"] == last_id
                        ):
                            st.session_state.click_marks.pop(i)
                            break
                    st.rerun()

        with c2:
            if st.button("Clear class"):
                ids_to_remove = set(st.session_state.samples[active_class])
                st.session_state.samples[active_class] = []
                st.session_state.click_marks = [
                    p for p in st.session_state.click_marks
                    if not (p["class"] == active_class and p["label_id"] in ids_to_remove)
                ]
                st.rerun()

    with right:
        st.subheader("Training and classification")

        st.write(f"Detected objects: {len(objects_df)}")

        train_df = build_training_table(objects_df, st.session_state.samples)
        st.write(f"Training samples: {len(train_df)}")

        if not train_df.empty:
            st.dataframe(
                train_df[["label_id", "target_class"] + feature_columns()].head(20),
                use_container_width=True
            )

        if st.button("Run classification"):
            n_nucleus = len(st.session_state.samples["Nucleus"])
            n_inflam = len(st.session_state.samples["Inflammation"])

            if n_nucleus == 0 and n_inflam == 0:
                st.error("Belum ada training sample.")
            else:
                result_df = objects_df.copy()
                result_df["predicted_class"] = "Cytoplasm"
                result_df["confidence"] = 0.5

                if n_nucleus > 0 and n_inflam > 0:
                    X_train = train_df[feature_columns()].copy()
                    y_train = train_df["target_class"].copy()
                    X_all = objects_df[feature_columns()].copy()

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
                    proba = clf.predict_proba(X_all).max(axis=1)

                    result_df["predicted_class"] = pred
                    result_df["confidence"] = proba

                # enforce manually clicked samples
                nucleus_ids = set(st.session_state.samples["Nucleus"])
                inflam_ids = set(st.session_state.samples["Inflammation"])
                exclude_ids = set(st.session_state.samples["Exclude"])

                if len(nucleus_ids) > 0:
                    result_df.loc[result_df["label_id"].isin(nucleus_ids), "predicted_class"] = "Nucleus"
                    result_df.loc[result_df["label_id"].isin(nucleus_ids), "confidence"] = 1.0

                if len(inflam_ids) > 0:
                    result_df.loc[result_df["label_id"].isin(inflam_ids), "predicted_class"] = "Inflammation"
                    result_df.loc[result_df["label_id"].isin(inflam_ids), "confidence"] = 1.0

                if len(exclude_ids) > 0:
                    result_df.loc[result_df["label_id"].isin(exclude_ids), "predicted_class"] = "Exclude"
                    result_df.loc[result_df["label_id"].isin(exclude_ids), "confidence"] = 1.0

                # everything else becomes cytoplasm
                other_mask = ~result_df["predicted_class"].isin(["Nucleus", "Inflammation", "Exclude"])
                result_df.loc[other_mask, "predicted_class"] = "Cytoplasm"

                st.session_state.result_df = result_df

        if st.session_state.result_df is not None:
            result_df = st.session_state.result_df.copy()
            st.success("Classification complete")

            display_df = result_df[result_df["predicted_class"] != "Exclude"].copy()

            if not display_df.empty:
                summary_df = (
                    display_df["predicted_class"]
                    .value_counts()
                    .rename_axis("Class")
                    .reset_index(name="Count")
                )
                summary_df["Percent"] = 100 * summary_df["Count"] / summary_df["Count"].sum()

                st.dataframe(summary_df, use_container_width=True)

                st.dataframe(
                    display_df[[
                        "label_id", "predicted_class", "confidence",
                        "area", "mean_intensity", "white_fraction", "pink_fraction"
                    ]],
                    use_container_width=True
                )

                nucleus_df = display_df[display_df["predicted_class"] == "Nucleus"].copy()
                inflam_df = display_df[display_df["predicted_class"] == "Inflammation"].copy()
                cyt_df = display_df[display_df["predicted_class"] == "Cytoplasm"].copy()

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Nucleus", len(nucleus_df))
                with m2:
                    st.metric("Inflammation", len(inflam_df))
                with m3:
                    st.metric("Cytoplasm", len(cyt_df))

                if len(cyt_df) > 0:
                    st.metric("Mean cytoplasm white fraction", f"{cyt_df['white_fraction'].mean() * 100:.1f}%")

    st.markdown("---")
    st.subheader("Segmentation result")

    if show_seg_overlay:
        seg_overlay = make_colored_segmentation(labeled, st.session_state.result_df)
        st.image(seg_overlay, caption="Segmented objects", use_container_width=True)

    if show_gallery:
        if st.session_state.result_df is not None:
            show_cell_gallery(
                st.session_state.result_df,
                rgb,
                labeled,
                title="Segmented cell gallery",
                n_cols=6,
                max_cells=60
            )
        else:
            show_cell_gallery(
                objects_df,
                rgb,
                labeled,
                title="Segmented cell gallery",
                n_cols=6,
                max_cells=60
            )

st.markdown("---")
st.caption("Versi stabil dasar: anotasi hanya nucleus dan inflammation, cytoplasm otomatis dari sisa segmen.")
