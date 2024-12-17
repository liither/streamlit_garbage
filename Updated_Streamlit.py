#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Class names for prediction
CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image for prediction.
    Resize the image to (224, 224) and normalize pixel values to [-1, 1].
    """
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize to match the model input
    image_array = np.array(image) / 127.5 - 1.0  # Normalize to [-1, 1]
    return image_array

# Function to load the trained model
@st.cache_resource
def load_trained_model():
    """
    Load the fine-tuned Keras model for garbage classification.
    """
    model_path = "mobilenetv2_finetuned.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict the class of an image
def predict_class(model, image_array):
    """
    Predict the class of the image using the trained model.
    """
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)  # Find the class with the highest probability
    confidence = np.max(predictions)  # Get the confidence score
    return CLASS_NAMES[predicted_index], confidence

# Streamlit App Logic
st.title("Garbage Classification with Fine-Tuned MobileNetV2")
st.write("Upload an image to classify garbage.")

# Upload Image Section
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

    # Load the trained model
    model = load_trained_model()

    # Predict the class and confidence
    prediction, confidence = predict_class(model, image)

    # Display the prediction result
    st.write(f"### Predicted Class: **{prediction}**")
    st.write(f"### Confidence: **{confidence:.2f}**")
