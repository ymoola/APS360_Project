import streamlit as st
from PIL import Image
from predict import predict
import os

st.set_page_config(page_title="Oral Disease Classifier", layout="centered")
st.title("🦷 Oral Disease Image Classifier")
st.markdown("Upload a mouth image and the ViT + ANN model will classify the oral disease.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            result = predict(image)
            st.success(f"🩺 Predicted Disease: **{result}**")
