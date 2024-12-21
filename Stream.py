import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Image Classification for Waste Categories")

MODEL_PATH = "model_quantized.tflite"

@st.cache_resource
def load_tflite_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please check the path.")
        st.stop()
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        st.success("Model loaded successfully!")
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        st.stop()

interpreter = load_tflite_model()

class_labels = [
    "Battery", "Biological", "Brown Glass", "Cardboard",
    "Clothes", "Green Glass", "Metal", "Paper",
    "Plastic", "Shoes", "Trash", "White Glass"
]

def preprocess_image(image):
    img = image.resize((150, 150))  # Resize ke 150x150 (sesuai dengan input model)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalisasi 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    return img_array

def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()  # Jalankan inferensi
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = predict_tflite(interpreter, processed_image)

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_labels[predicted_index]

    st.write(f"### Predicted Class: {predicted_class}")
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob:.4f}")
