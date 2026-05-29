import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import gdown

from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection using CNN",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #020617, #0f172a);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main-title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    color: #00E5FF;
    margin-top: 20px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

.success-card {
    background: green;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    color: white;
}

.error-card {
    background: red;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    color: white;
}

.warning-card {
    background: orange;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    color: black;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection using CNN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered MRI Classification System</div>',
    unsafe_allow_html=True
)

# =========================================================
# MODEL DOWNLOAD
# =========================================================

MODEL_PATH = "model.h5"

FILE_ID = "1RuIYg4imou1JzjRVowF-H4BJXgP2FIOF"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):

    st.warning("Downloading AI Model... Please Wait ⏳")

    gdown.download(URL, MODEL_PATH, quiet=False)

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model(MODEL_PATH, compile=False)

class_names = [
    "glioma",
    "meningioma",
    "notbrain",
    "notumor",
    "pituitary"
]

IMG_SIZE = 224

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "png", "jpeg"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded MRI", use_column_width=True)

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    prediction = model.predict(img_array)[0]

    class_index = np.argmax(prediction)

    class_name = class_names[class_index]

    confidence = np.max(prediction) * 100

    # INVALID IMAGE
    if class_name == "notbrain":

        st.markdown(
            """
            <div class="warning-card">
            ❌ Invalid Image <br>
            Please Upload Brain MRI
            </div>
            """,
            unsafe_allow_html=True
        )

    # NO TUMOR
    elif class_name == "notumor":

        st.markdown(
            """
            <div class="success-card">
            ✅ No Tumor Detected
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"### Confidence: {confidence:.2f}%")

    # TUMOR DETECTED
    else:

        st.markdown(
            f"""
            <div class="error-card">
            ⚠️ Brain Tumor Detected <br>
            Type: {class_name.capitalize()}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"### Confidence: {confidence:.2f}%")
