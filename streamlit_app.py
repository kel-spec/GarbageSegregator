import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Page config
st.set_page_config(page_title="Garbage Segregator", page_icon="♻️", layout="centered")

# --- Custom Styling ---
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

# --- Load Model ---
model_path = "biodegradable_classifier.h5"
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1Op3lmIpjkVWprlLlwdMD51o196CX2FO9",
            model_path,
            quiet=False
        )
model = load_model(model_path)

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Segregator", "Want to learn more?", "About us"])

# --- Page: Segregator ---
if page == "Segregator":
    st.title("♻️ Garbage Segregator")
    st.markdown("Upload an image to classify whether the waste is **Biodegradable** 🌱 or **Non-Biodegradable** 🗑️.")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess image
        img = img.resize((150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        st.write(f"**Model Confidence:** `{confidence:.2f}`")

        if confidence > 0.5:
            result = "Non-Biodegradable"
            st.error("🚯 **Predicted: Non-Biodegradable**")
        else:
            result = "Biodegradable"
            st.success("🌿 **Predicted: Biodegradable**")

        # Save to history
        st.session_state.history.append((result, confidence))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕘 Prediction History")
    if st.session_state.history:
        for idx, (label, conf) in enumerate(reversed(st.session_state.history), 1):
            st.sidebar.write(f"**{idx}.** `{label}` (Confidence: `{conf:.2f}`)")
    else:
        st.sidebar.write("No predictions yet.")

# --- Page: Education ---
elif page == "Want to learn more?":
    st.title("📚 Learn About Waste Types")
    st.markdown("""
    ### 🧪 What is Biodegradable Waste?
    **Biodegradable waste** decomposes naturally by microorganisms.

    - 🟢 Examples: Food scraps, paper, leaves, cotton, wood.
    - ✅ Compostable and environmentally friendly.
    - ❌ If mixed with non-bio waste, composting fails.

    ### 🔴 What is Non-Biodegradable Waste?
    **Non-biodegradable waste** doesn't break down easily and can persist for hundreds of years.

    - 🔴 Examples: Plastic, glass, metal, batteries.
    - ✅ Recyclable in many cases.
    - ❌ Improper disposal harms nature and wildlife.

    ### 🌍 Why Segregation Matters
    - Reduces landfill burden
    - Promotes composting and recycling
    - Prevents pollution and supports sustainability
    """)

    st.subheader("🎥 Watch and Learn")
    st.video("https://www.youtube.com/watch?v=4ECxHTf_Co4")
    st.video("https://www.youtube.com/watch?v=V0lQ3ljjl40")

# --- Page: About Us ---
elif page == "About us":
    st.title("👥 About the Team")
    st.markdown("""
    ### 🧑‍💻 Team Members
    - Lagunday, Michael Luis
    - Salamanca, Lance
    - Sarcauga, Dexter
    - Bondoc, Christian
    - Roldan, Christian Cyril

    ### 🎯 The Problem
    Waste segregation remains a challenge in many communities, leading to improper disposal, pollution, and health hazards.

    ### 💡 Our Solution
    We created a simple image classification app that allows users to upload a photo of waste and instantly learn whether it’s biodegradable or not. This can assist in educating users and improving real-world disposal habits.

    **Built using:** Streamlit, TensorFlow, and Google Drive integration.
    """)
