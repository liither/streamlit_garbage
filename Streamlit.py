#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    Resize the image to (150, 150) and normalize pixel values.
    """
    image = Image.open(uploaded_file)
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array

# Function to load the trained model
def load_trained_model():
    """
    Load the fine-tuned Keras model for garbage classification.
    """
    model_path = "garbage_classification_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict the class of an image
def predict_class(model, image_array):
    """
    Predict the class of the image using the trained model.
    """
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions)
    return CLASS_NAMES[predicted_index]

# Streamlit App Logic
st.title("Garbage Classification with Fine-Tuned Model")
st.write("Upload an image to classify garbage.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the trained model
    model = load_trained_model()

    # Predict the class
    prediction = predict_class(model, image)

    # Display the prediction result
    st.write(f"Predicted Class: **{prediction}**")

