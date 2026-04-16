import os
from pathlib import Path
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from PIL import Image
from radiomics import featureextractor

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Chest X-ray Radiomics Classifier",
    page_icon="🫁",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lightgbm_central_crop_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder_central_crop.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns_central_crop.pkl"

# =========================================================
# HELPERS
# =========================================================
def check_artifacts():
    missing = []
    for p in [MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_COLUMNS_PATH]:
        if not p.exists():
            missing.append(str(p.name))
    return missing


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, label_encoder, feature_columns


@st.cache_resource
def get_radiomics_extractor():
    params = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "verbose": False
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableAllFeatures()
    return extractor


def preprocess_and_crop(img, output_size=(256, 256)):
    """
    Central chest ROI preprocessing:
    1. central crop
    2. CLAHE enhancement
    3. resize to fixed size
    """
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    h, w = img.shape

    y1 = int(0.10 * h)
    y2 = int(0.90 * h)
    x1 = int(0.15 * w)
    x2 = int(0.85 * w)

    cropped = img[y1:y2, x1:x2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped = clahe.apply(cropped)

    cropped = cv2.resize(cropped, output_size)
    return cropped


def create_roi_mask(img):
    return np.ones(img.shape, dtype=np.uint8)


def extract_features_from_image(img_array, extractor, expected_columns):
    """
    img_array: grayscale numpy array
    expected_columns: feature columns used during training
    """
    processed = preprocess_and_crop(img_array, output_size=(256, 256))
    mask = create_roi_mask(processed)

    img_sitk = sitk.GetImageFromArray(processed)
    mask_sitk = sitk.GetImageFromArray(mask)

    features = extractor.execute(img_sitk, mask_sitk)

    clean_features = {}
    for k, v in features.items():
        if not k.startswith("diagnostics"):
            clean_features[k] = v

    feature_df = pd.DataFrame([clean_features])

    # align exactly to training feature columns
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0.0

    feature_df = feature_df[expected_columns]

    return processed, feature_df


def interpret_prediction(predicted_label):
    if predicted_label.lower() == "covid":
        return (
            "This chest X-ray is most consistent with **COVID-like pulmonary involvement**."
        )
    elif predicted_label.lower() == "virus":
        return (
            "This chest X-ray shows features most consistent with **non-COVID viral lung involvement**."
        )
    elif predicted_label.lower() == "normal":
        return (
            "This chest X-ray appears **radiographically normal** based on the model output."
        )
    return "Prediction completed."


# =========================================================
# UI
# =========================================================
st.title("🫁 Chest X-ray Radiomics Classifier")
st.markdown(
    """
This app uses a **Central Chest ROI → Radiomics → LightGBM** pipeline to classify uploaded chest X-rays into:

- **COVID**
- **Virus**
- **Normal**
"""
)

st.info(
    "This tool is for decision support and research use. It is not a standalone medical diagnosis."
)

missing_files = check_artifacts()
if missing_files:
    st.error(
        "Missing required files in this folder: " + ", ".join(missing_files)
    )
    st.stop()

model, label_encoder, feature_columns = load_artifacts()
extractor = get_radiomics_extractor()

with st.sidebar:
    st.header("About")
    st.write("Model: LightGBM")
    st.write("Preprocessing: Central chest crop + CLAHE")
    st.write("Feature extraction: PyRadiomics (Original + Wavelet)")
    st.write(f"Expected radiomics features: {len(feature_columns)}")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("L")
        img_array = np.array(pil_image)

        processed_img, feature_df = extract_features_from_image(
            img_array,
            extractor,
            feature_columns
        )

        pred_encoded = model.predict(feature_df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_df)[0]
        else:
            probabilities = None

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(img_array, caption="Original uploaded X-ray", use_container_width=True)

        with col2:
            st.subheader("Processed ROI")
            st.image(processed_img, caption="Central chest ROI used for radiomics", use_container_width=True)

        st.markdown("---")
        st.subheader("Prediction")

        if pred_label.lower() == "covid":
            st.error(f"Predicted Class: {pred_label.upper()}")
        elif pred_label.lower() == "virus":
            st.warning(f"Predicted Class: {pred_label.upper()}")
        else:
            st.success(f"Predicted Class: {pred_label.upper()}")

        st.markdown(interpret_prediction(pred_label))

        if probabilities is not None:
            st.subheader("Class Probabilities")

            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)

            st.dataframe(
                prob_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True
            )

            for _, row in prob_df.iterrows():
                st.write(f"**{row['Class']}**")
                st.progress(float(row["Probability"]))

        with st.expander("Show extracted feature preview"):
            st.dataframe(feature_df.iloc[:, :20], use_container_width=True)

        st.markdown("---")
        st.caption(
            "Suggested interpretation wording: "
            "COVID-like pulmonary involvement, non-COVID viral lung involvement, or radiographically normal chest."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.write("Upload a chest X-ray image to begin.")
