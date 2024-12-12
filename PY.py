#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "garbage_classification_model.h5"

try:
    model = load_model(MODEL_PATH)
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error("Gagal memuat model. Pastikan file garbage_classification_model.h5 ada di direktori.")
    st.stop()

class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']


st.title("Garbage Classification")
st.write("Upload gambar untuk klasifikasi sampah.")

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_idx]


    st.write(f"Predicted Class: *{predicted_class}*")


# In[2]:


get_ipython().system('pip install streamlit tensorflow')


# In[4]:


streamlit run app.py


# In[ ]:




