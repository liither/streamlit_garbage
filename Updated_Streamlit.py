#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[6]:


import tensorflow as tf

# Build MobileNetV2 model with fine-tuning
def build_mobilenetv2_finetune_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True  # Enable fine-tuning
    
    # Add classification layers
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    
    # Assemble the model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and save the model (replace `train_data` and `val_data` with your datasets)
model = build_mobilenetv2_finetune_model()
# model.fit(train_data, epochs=5, validation_data=val_data)
model.save('mobilenetv2_finetuned.h5')

