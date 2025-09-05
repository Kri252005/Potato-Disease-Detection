import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------------------
# Load trained model
# ---------------------------
def load_model():
    model = tf.keras.models.load_model("model1.h5")  # Ensure model1.h5 is in the same folder
    return model

model = load_model()

# ---------------------------
# App title
# ---------------------------
st.write("""
# Potato Leaf Disease Classification
Upload a potato leaf image and the model will predict:
- Early Blight
- Healthy
- Other
""")

# ---------------------------
# File uploader
# ---------------------------
file = st.file_uploader("Upload the leaf image", type=["jpg", "png", "jpeg"])

# ---------------------------
# Image preprocessing
# ---------------------------
def prepare(image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Class labels
labels = ["Early_Blight", "Healthy", "Other"]

# ---------------------------
# Prediction
# ---------------------------
if file is None:
    st.text("Please upload an image.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    processed_image = prepare(image)
    prediction = model.predict(processed_image)[0]  # first row of prediction
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Logic: only accept Early Blight or Healthy if confidence > 60%
    if predicted_class == 0 and confidence > 0.6:
        result = "Early_Blight"
    elif predicted_class == 1 and confidence > 0.6:
        result = "Healthy"
    else:
        result = "Other"

    st.write(f"### Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}")
