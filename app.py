import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
MODEL_PATH = "waste_classification_model.h5"  # Ensure the correct path
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size (same as used in ImageDataGenerator)
IMG_SIZE = (224, 224)  # Change according to your model's input size

# Define class names (modify based on your dataset)
CLASS_NAMES = ["This is an Organic Waste", "This is a Recyclable Waste"]  # Example categories

# Function to preprocess image (similar to ImageDataGenerator)
def preprocess_image(image):
    image = image.resize(IMG_SIZE)  # Resize like ImageDataGenerator
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Waste Classification Model")
st.write("Upload an image to classify waste categories.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    # Show prediction result
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {100 * np.max(prediction):.2f}%")
