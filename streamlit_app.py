import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Page config
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

# Load model from Google Drive
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

# Sidebar only for history
with st.sidebar:
    st.header("🕘 Prediction History")
    if st.session_state.history:
        for idx, (label, conf) in enumerate(reversed(st.session_state.history), 1):
            st.write(f"**{idx}.** `{label}` (Confidence: `{conf:.2f}`)")
    else:
        st.write("No predictions yet.")

# --- Main App ---
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

    # Display prediction
    if confidence > 0.5:
        result = "Non-Biodegradable"
        st.error("🚯 **Predicted: Non-Biodegradable**")
    else:
        result = "Biodegradable"
        st.success("🌿 **Predicted: Biodegradable**")

    # Save to history
    st.session_state.history.append((result, confidence))

# --- Education Section ---
st.markdown("---")
st.header("📚 Learn About Waste Types")
st.markdown("""
### 🧪 What is Biodegradable Waste?
**Biodegradable waste** refers to substances that decompose naturally and safely by microorganisms.

- 🟢 Examples: Food scraps, paper, leaves, cotton, wood.
- ✅ Proper disposal turns them into compost or natural fertilizers.
- ❌ If mixed with plastics, they lose their composting value.

---

### 🔴 What is Non-Biodegradable Waste?
**Non-biodegradable waste** doesn't decompose or takes hundreds of years to break down.

- 🔴 Examples: Plastic, metal, glass, batteries, Styrofoam.
- ✅ Can be recycled or reused.
- ❌ Improper disposal leads to pollution, harming marine and land ecosystems.

---

### 🌍 Why Segregation Matters
Proper segregation:
- Reduces landfill waste
- Promotes recycling and composting
- Lowers greenhouse gas emissions
""")

# --- Educational Videos ---
st.subheader("🎥 Educational Videos")
st.video("https://www.youtube.com/watch?v=4ECxHTf_Co4")  # Biodegradable vs non-biodegradable waste
st.video("https://www.youtube.com/watch?v=V0lQ3ljjl40")  # Waste management

st.markdown("*Tip: Scroll up to upload and test more waste images!*")
