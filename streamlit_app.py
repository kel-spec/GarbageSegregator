import streamlit as st
import gdown
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Download model from Google Drive (only once)
model_path = "biodegradable_classifier.h5"
file_id = "1Op3lmIpjkVWprlLlwdMD51o196CX2FO9"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Set up Streamlit UI
st.title("Biodegradable vs Non-Biodegradable Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((150, 150))  # Adjust to your model's input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = "Biodegradable" if prediction[0][0] > 0.5 else "Non-Biodegradable"
    st.write(f"### Prediction: {label}")
