import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Set page configuration
st.set_page_config(page_title="Garbage Segregator", page_icon="♻️", layout="centered")

# Apply custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6em 1.5em;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            margin-top: 10px;
        }
        .stFileUploader {
            margin-bottom: 1em;
        }
        .stImage {
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 1em;
        }
        h1 {
            color: #2E8B57;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("♻️ Garbage Segregator")
st.markdown("Upload an image to classify whether the waste is **Biodegradable** 🌱 or **Non-Biodegradable** 🗑️.")

# Download model if not present
model_path = "biodegradable_classifier.h5"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1Op3lmIpjkVWprlLlwdMD51o196CX2FO9",
            model_path,
            quiet=False
        )

# Load the trained model
model = load_model(model_path)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((150, 150))  # Match training image size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    # Display confidence
    st.write(f"**Model Confidence:** `{confidence:.2f}`")

    # Display class prediction
    if confidence > 0.5:
        st.error("🚯 **Predicted: Non-Biodegradable**")
    else:
        st.success("🌿 **Predicted: Biodegradable**")
