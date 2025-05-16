import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Title and description
st.title("Garbage Segregator")
st.write("Upload an image to classify it as **Biodegradable** or **Non-Biodegradable**.")

# Load model from Google Drive using gdown
model_path = "biodegradable_classifier.h5"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1Op3lmIpjkVWprlLlwdMD51o196CX2FO9",
            model_path,
            quiet=False
        )

# Load the model
model = load_model(model_path)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image (match training setup)
    img = img.resize((150, 150))  # Replace with your model's input size if different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    # Show results
    st.write(f"Model confidence: **{confidence:.2f}**")
    if confidence > 0.5:
        st.success("Predicted Class: **Non-Biodegradable**")
    else:
        st.success("Predicted Class: **Biodegradable**")
