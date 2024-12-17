#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title of the application
st.title("Image Classification using TFLite Model (MobileNetV2)")

# Load the TFLite model
@st.cache(allow_output_mutation=True)
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")  # Path to your TFLite model
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions using TFLite interpreter
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Upload image from user
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction using TFLite model
    prediction = predict_tflite(interpreter, processed_image)
    
    # Display prediction results
    st.write("Prediction Probabilities:")
    st.write(prediction)

