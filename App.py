#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model(r"C:\Users\lenovo\Documents\Kuliah Semester5\Deep Learning\Lec\UAS\garbage_classification_model.h5")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("garbage_classification_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite.")


# In[6]:


import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="garbage_classification_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Streamlit app
st.title("Garbage Classification")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]

    st.write(f"Predicted Class: *{predicted_class}*")


# In[ ]:




