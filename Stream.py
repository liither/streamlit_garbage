import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title of the application
st.title("Image Classification for Waste Categories")

# Load the TFLite model using st.cache_resource
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")  # Path to your TFLite model
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Define class labels
class_labels = [
    "Battery", "Biological", "Brown Glass", "Cardboard",
    "Clothes", "Green Glass", "Metal", "Paper",
    "Plastic", "Shoes", "Trash", "White Glass"
]

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize to 150x150
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions using TFLite interpreter
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Upload image from user
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the TFLite model
    prediction = predict_tflite(interpreter, processed_image)
    
    # Determine the predicted class
    predicted_index = np.argmax(prediction[0])  # Get the index of max probability
    predicted_class = class_labels[predicted_index]  # Get the class name
    
    # Display the predicted class
    st.write(f"### Predicted Class: {predicted_class}")
