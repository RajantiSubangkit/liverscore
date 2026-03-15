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
st.caption(
    "Annotate only Nucleus and Inflammation. Cytoplasm is assigned automatically "
    "from remaining segmented objects."
)

# =========================================================
# SESSION STATE
# =========================================================
ANNOTATION_CLASSES = ["Nucleus", "Inflammation", "Exclude"]
FINAL_CLASSES = ["Nucleus", "Inflammation", "Cytoplasm", "Exclude"]

if "samples" not in st.session_state:
    st.session_state.samples = {c: [] for c in ANNOTATION_CLASSES}

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
    st.session_state.samples = {c: [] for c in ANNOTATION_CLASSES}
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
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(1))

    dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    peaks = feature.peak_local_max(
        dist,
        min_distance=8,
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

        if len(pix_gray) == 0:
            continue

        perimeter = p.perimeter if p.perimeter > 0 else np.nan
        major = p.major_axis_length if p.major_axis_length > 0 else np.nan
        minor = p.minor_axis_length if p.minor_axis_length > 0 else np.nan

        circularity = 4 * np.pi * p.area / (perimeter ** 2) if perimeter and perimeter > 0 else np.nan
        aspect_ratio = major / minor if minor and minor > 0 else np.nan

        white_fraction = float(np.mean(
            (pix_rgb[:, 0] > 210) &
            (pix_rgb[:, 1] > 210) &
            (pix_rgb[:, 2] > 210)
        ))

        pink_fraction = float(np.mean(
            (pix_rgb[:, 0] > pix_rgb[:, 1]) &
            (pix_rgb[:, 0] > pix_rgb[:, 2]) &
            (pix_rgb[:, 0] > 120)
        ))

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
            "mean_b": float(np.mean(pix_rgb[:, 2])),
            "white_fraction": white_fraction,
            "pink_fraction": pink_fraction
        })

    return pd.DataFrame(rows)


def feature_cols():
    return [
        "area", "perimeter", "major_axis_length", "minor_axis_length",
        "eccentricity", "solidity", "extent", "circularity",
        "aspect_ratio", "mean_intensity", "std_intensity",
        "mean_r", "mean_g", "mean_b",
        "white_fraction", "pink_fraction"
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
    best_dist = 1e18
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
        "Inflammation": (255, 0, 255),
        "Cytoplasm": (255, 255, 0),
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


def make_segmentation_overlay(labeled, result_df=None):
    h, w = labeled.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    color_map = {
        "Nucleus": (0, 255, 0),
        "Inflammation": (255, 0, 255),
        "Cytoplasm": (255, 255, 0),
        "Exclude": (120, 120, 120)
    }

    if result_df is not None and not result_df.empty and "predicted_class" in result_df.columns:
        for _, row in result_df.iterrows():
            lid = int(row["label_id"])
            cls = row["predicted_class"]
            out[labeled == lid] = color_map.get(cls, (255, 255, 255))
    else:
        rng = np.random.default_rng(123)
        for lid in np.unique(labeled):
            if lid == 0:
                continue
            out[labeled == lid] = rng.integers(50, 255, size=3, dtype=np.uint8)

    boundaries = segmentation.find_boundaries(labeled, mode="outer")
    out[boundaries] = [255, 255, 255]
    return out


def crop_single_cell(rgb, labeled, row, pad=8, target_size=96):
    lid = int(row["label_id"])

    minr = max(int(row["bbox_minr"]) - pad, 0)
    minc = max(int(row["bbox_minc"]) - pad, 0)
    maxr = min(int(row["bbox_maxr"]) + pad, rgb.shape[0])
    maxc = min(int(row["bbox_maxc"]) + pad, rgb.shape[1])

    crop_rgb = rgb[minr:maxr, minc:maxc].copy()
    crop_mask = labeled[minr:maxr, minc:maxc] == lid

    if crop_rgb.size == 0:
        return None

    canvas = np.ones_like(crop_rgb, dtype=np.uint8) * 255
    canvas[crop_mask] = crop_rgb[crop_mask]

    boundary = segmentation.find_boundaries(crop_mask, mode="outer")
    canvas[boundary] = [255, 0, 0]

    h, w = canvas.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_size / max(h, 1), target_size / max(w, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    final_canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return final_canvas


def show_cell_gallery(df_to_show, rgb, labeled, title, n_cols=6, max_cells=60):
    st.subheader(title)

    if df_to_show is None or df_to_show.empty:
        st.info("No objects to display.")
        return

    show_df = df_to_show.head(max_cells).copy()
    chunks = [show_df.iloc[i:i + n_cols] for i in range(0, len(show_df), n_cols)]

    for chunk in chunks:
        cols = st.columns(n_cols)
        for i in range(n_cols):
            with cols[i]:
                if i < len(chunk):
                    row = chunk.iloc[i]
                    thumb = crop_single_cell(rgb, labeled, row, pad=8, target_size=96)
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
    active_class = st.selectbox("Annotation class", ANNOTATION_CLASSES)
    show_gallery = st.checkbox("Show segmented cell gallery", value=True)
    show_seg_overlay = st.checkbox("Show segmentation overlay", value=True)

    if st.button("Reset all"):
        reset_all()
        st.rerun()

    st.markdown("---")
    for c in ANNOTATION_CLASSES:
        st.write(f"{c}: {len(st.session_state.samples[c])}")


# =========================================================
# UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload liver histopathology image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

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

    top_left, top_right = st.columns([1.4, 1])

    with top_left:
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
                already = any(lid in st.session_state.samples[c] for c in ANNOTATION_CLASSES)
                if not already:
                    st.session_state.samples[active_class].append(lid)
                    st.session_state.click_marks.append({
                        "x": real_x,
                        "y": real_y,
                        "class": active_class,
                        "label_id": lid
                    })
                    st.rerun()

        st.caption("Klik objek untuk jadi training sample nucleus atau inflammation.")

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

    with top_right:
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
            n_nucleus = len(st.session_state.samples["Nucleus"])
            n_inflam = len(st.session_state.samples["Inflammation"])

            if n_nucleus == 0 and n_inflam == 0:
                st.error("Belum ada training sample.")
            else:
                result_df = objects_df.copy()

                if n_nucleus > 0 and n_inflam > 0:
                    X_train = train_df[feature_cols()].copy()
                    y_train = train_df["target_class"].copy()

                    X_all = objects_df[feature_cols()].copy()

                    med = X_train.median(numeric_only=True)
                    X_train = X_train.fillna(med)
                    X_all = X_all.fillna(med)

                    clf = RandomForestClassifier(
                        n_estimators=250,
                        random_state=42,
                        class_weight="balanced"
                    )
                    clf.fit(X_train, y_train)

                    pred = clf.predict(X_all)
                    prob = clf.predict_proba(X_all).max(axis=1)

                    result_df["predicted_class"] = pred
                    result_df["confidence"] = prob

                else:
                    # fallback kalau baru ada 1 kelas manual
                    result_df["predicted_class"] = "Cytoplasm"
                    result_df["confidence"] = 0.5

                    if n_nucleus > 0:
                        nucleus_ids = set(st.session_state.samples["Nucleus"])
                        result_df.loc[result_df["label_id"].isin(nucleus_ids), "predicted_class"] = "Nucleus"
                        result_df.loc[result_df["label_id"].isin(nucleus_ids), "confidence"] = 1.0

                    if n_inflam > 0:
                        inflam_ids = set(st.session_state.samples["Inflammation"])
                        result_df.loc[result_df["label_id"].isin(inflam_ids), "predicted_class"] = "Inflammation"
                        result_df.loc[result_df["label_id"].isin(inflam_ids), "confidence"] = 1.0

                excluded = set(st.session_state.samples["Exclude"])
                if len(excluded) > 0:
                    result_df.loc[result_df["label_id"].isin(excluded), "predicted_class"] = "Exclude"
                    result_df.loc[result_df["label_id"].isin(excluded), "confidence"] = 1.0

                # semua sisa objek yang bukan nucleus / inflammation / exclude
                # otomatis dianggap cytoplasm
                mask_other = ~result_df["predicted_class"].isin(["Nucleus", "Inflammation", "Exclude"])
                result_df.loc[mask_other, "predicted_class"] = "Cytoplasm"

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
                    show_df[[
                        "label_id", "predicted_class", "confidence",
                        "area", "mean_intensity", "white_fraction", "pink_fraction"
                    ]],
                    use_container_width=True
                )

                nucleus_df = show_df[show_df["predicted_class"] == "Nucleus"].copy()
                inflam_df = show_df[show_df["predicted_class"] == "Inflammation"].copy()
                cyt_df = show_df[show_df["predicted_class"] == "Cytoplasm"].copy()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Nucleus", len(nucleus_df))
                with c2:
                    st.metric("Inflammation", len(inflam_df))
                with c3:
                    st.metric("Cytoplasm", len(cyt_df))

                if len(cyt_df) > 0:
                    st.metric(
                        "Mean cytoplasm white fraction (%)",
                        f"{cyt_df['white_fraction'].mean() * 100:.1f}"
                    )

                csv = show_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="liver_classification.csv",
                    mime="text/csv"
                )

    st.markdown("---")
    st.subheader("Segmentation result")

    if show_seg_overlay:
        seg_overlay = make_segmentation_overlay(labeled, st.session_state.result_df)
        st.image(seg_overlay, caption="Segmentation overlay per object", use_container_width=True)

    if show_gallery:
        if st.session_state.result_df is not None:
            show_cell_gallery(
                st.session_state.result_df,
                rgb,
                labeled,
                title="Segmented cells gallery",
                n_cols=6,
                max_cells=60
            )
        else:
            show_cell_gallery(
                objects_df,
                rgb,
                labeled,
                title="Segmented cells gallery",
                n_cols=6,
                max_cells=60
            )

st.markdown("---")
st.caption(
    "Cytoplasm is automatically assigned from segmented objects that are not classified "
    "as Nucleus, Inflammation, or Exclude."
)
