import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title of the application
st.title("Image Classification for Waste Categories")

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    model_path = "model_quantized.tflite"  # Pastikan path model benar
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

interpreter = load_tflite_model()

# Define class labels
class_labels = [
    "Battery", "Biological", "Brown Glass", "Cardboard",
    "Clothes", "Green Glass", "Metal", "Paper",
    "Plastic", "Shoes", "Trash", "White Glass"
]

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Konsisten dengan resolusi pelatihan
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalisasi ke 0-1
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
    
    # Determine the predicted class with confidence threshold
    confidence_threshold = 0.5
    predicted_index = np.argmax(prediction[0])
    predicted_confidence = prediction[0][predicted_index]
    predicted_class = class_labels[predicted_index]

    if predicted_confidence >= confidence_threshold:
        st.write(f"### Predicted Class: {predicted_class}")
    else:
        st.write("### Unable to classify: Confidence too low.")
    
    # Debugging: Show prediction probabilities
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob:.4f}")
