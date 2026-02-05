import cv2
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

# ===== FEATURE: PDF GENERATION LIBRARIES =====
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io


# ===== FEATURE: USER-FRIENDLY LABEL MAPPING =====
# Converts technical ML labels into simple human readable names
FRIENDLY_LABELS = {
    "loupe": "Magnifying Glass",
    "lens_cap": "Camera Lens Cap",
    
}


# ===== CORE MODEL LOADING =====
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


# ===== IMAGE PREPROCESSING =====
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# ===== IMAGE CLASSIFICATION =====
def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


# ===== FEATURE: FRIENDLY NAME CONVERTER FUNCTION =====
def get_friendly_name(label):
    return FRIENDLY_LABELS.get(label, label.replace("_", " ").title())


# ===== FEATURE: TXT REPORT GENERATOR =====
def generate_txt(labels, scores):
    content = "AI Image Classification Results\n\n"
    for l, s in zip(labels, scores):
        content += f"{l}: {s:.2%}\n"
    return content


# ===== FEATURE: PDF REPORT GENERATOR =====
def generate_pdf(labels, scores):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "AI Image Classification Report")

    c.setFont("Helvetica", 12)
    y = 700
    for l, s in zip(labels, scores):
        c.drawString(100, y, f"{l}: {s:.2%}")
        y -= 25

    c.save()
    buffer.seek(0)
    return buffer


# ===== PREDICTION DISPLAY + EXTRA FEATURES =====
def show_predictions(predictions):
    st.subheader("Predictions")

    labels = []
    scores = []

    for _, label, score in predictions:
        friendly_label = get_friendly_name(label)
        st.write(f"**{friendly_label}**: {score:.2%}")
        labels.append(friendly_label)
        scores.append(score)

    # ===== FEATURE: BAR CHART VISUALIZATION =====
    st.subheader("Confidence Visualization")
    df = pd.DataFrame({
        "Label": labels,
        "Confidence": scores
    })
    df = df.set_index("Label")
    st.bar_chart(df)

    # ===== FEATURE: RESULT DOWNLOAD (TXT + PDF) =====
    st.subheader("Download Results")

    txt_data = generate_txt(labels, scores)
    st.download_button(
        label="Download TXT",
        data=txt_data,
        file_name="results.txt",
        mime="text/plain"
    )

    pdf_data = generate_pdf(labels, scores)
    st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name="results.pdf",
        mime="application/pdf"
    )


def main():
    st.set_page_config(page_title="SmartVision AI", page_icon="📸", layout="centered")

    st.title("Image Recognition & Classification App")
    st.write("Upload an image or capture from webcam and let AI tell you what it is!")

    # ===== FEATURE: MODEL CACHING (Performance Optimization) =====
    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    # ===== FEATURE: MULTIPLE IMAGE INPUT SOURCES =====
    option = st.radio(
        "Choose Image Source:",
        ("Upload Image", "Use Webcam")
    )

    image = None

    # ===== FEATURE: IMAGE UPLOAD =====
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===== FEATURE: WEBCAM CAPTURE =====
    if option == "Use Webcam":
        camera_photo = st.camera_input("Take a picture")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.image(image, caption="Captured Image", use_container_width=True)

    # ===== CLASSIFICATION TRIGGER =====
    if image is not None:
        if st.button("Classify Image"):
            with st.spinner("Analyzing Image..."):
                predictions = classify_image(model, image)
                if predictions:
                    show_predictions(predictions)


if __name__ == "__main__":
    main()
