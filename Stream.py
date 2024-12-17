import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Title of the application
st.title("Image Classification for Waste Categories")

# Path ke model TFLite
MODEL_PATH = "model_quantized.tflite"

# Fungsi untuk memuat model TFLite
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

# Label kelas
class_labels = [
    "Battery", "Biological", "Brown Glass", "Cardboard",
    "Clothes", "Green Glass", "Metal", "Paper",
    "Plastic", "Shoes", "Trash", "White Glass"
]

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    img = image.resize((224, 224))  # Konsisten dengan resolusi pelatihan
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalisasi 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    return img_array

# Fungsi untuk prediksi menggunakan TFLite
def predict_tflite(interpreter
