import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Page configuration
st.set_page_config(page_title="Garbage Segregator", page_icon="♻️", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6em 1.5em;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            margin-top: 10px;
        }
        h1 {
            color: #2E8B57;
        }
    </style>
""", unsafe_allow_html=True)

# Load model from Google Drive (if not already downloaded)
model_path = "biodegradable_classifier.h5"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1Op3lmIpjkVWprlLlwdMD51o196CX2FO9",
            model_path,
            quiet=False
        )

model = load_model(model_path)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for history and educational content
with st.sidebar:
    st.header("📁 Navigation")
    nav = st.radio("Go to", ["🏠 Main", "📚 Education", "🕘 Prediction History"])

    if nav == "📚 Education":
        st.subheader("🧪 What is Biodegradable?")
        st.markdown("""
        **Biodegradable waste** is waste that can be broken down naturally by microorganisms.

        - 🟢 Examples: Food scraps, paper, leaves, cotton, wood.
        - ✅ Proper disposal helps create compost and reduce landfill waste.
        - ❌ Improper disposal can still contribute to pollution if mixed with plastics.

        ---

        **Non-Biodegradable waste** does *not* decompose naturally.

        - 🔴 Examples: Plastic, metal, glass, Styrofoam.
        - ✅ Can be recycled or repurposed.
        - ❌ Improper disposal causes long-term environmental damage, water and soil pollution.

        👉 *Segregating waste properly helps protect ecosystems and reduces pollution.*
        """)

    elif nav == "🕘 Prediction History":
        st.subheader("📜 Session Prediction History")
        if st.session_state.history:
            for idx, (label, conf) in enumerate(reversed(st.session_state.history), 1):
                st.write(f"**{idx}.** `{label}` (Confidence: `{conf:.2f}`)")
        else:
            st.write("No predictions yet.")

# Main content
if nav == "🏠 Main":
    st.title("♻️ Garbage Segregator")
    st.markdown("Upload an image to classify whether the waste is **Biodegradable** 🌱 or **Non-Biodegradable** 🗑️.")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess
        img = img.resize((150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        st.write(f"**Model Confidence:** `{confidence:.2f}`")

        # Interpret and display result
        if confidence > 0.5:
            result = "Non-Biodegradable"
            st.error("🚯 **Predicted: Non-Biodegradable**")
        else:
            result = "Biodegradable"
            st.success("🌿 **Predicted: Biodegradable**")

        # Store in session history
        st.session_state.history.append((result, confidence))
