#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
import pandas as pd
import shutil
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# In[2]:


import os
import random
import shutil

base_dir = r"C:\Users\MSI BRAVO 15\Downloads\garbage_classification"
output_dir = r"C:\Users\MSI BRAVO 15\Downloads\garbage_classification_500"

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if os.path.isdir(class_path):

        all_images = os.listdir(class_path)
        
        sampled_images = random.sample(all_images, min(500, len(all_images)))
   
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        for image_name in sampled_images:
            src = os.path.join(class_path, image_name)
            dest = os.path.join(output_class_path, image_name)
            shutil.copy(src, dest)

        print(f"Class {class_name}: {len(sampled_images)} images sampled.")


# In[3]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    output_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
from sklearn.model_selection import train_test_split


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# In[5]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',        
    patience=3,                
    restore_best_weights=True  
)


# ### Melatih Model dengan EarlyStopping

# In[6]:


history = model.fit(
    train_generator,  
    validation_data=validation_generator,  
    epochs=50,  
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]  
)


# ### Evaluasi Model

# In[7]:


loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")


# ### Visualisasi Hasil

# In[8]:


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Fine-Tuning

# ### Base Model (MobileNetV2)

# In[9]:


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

base_model.trainable = False


# ### Layer Tambahan untuk Fine-Tuning

# In[11]:


from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([
    base_model,  
    GlobalAveragePooling2D(),  
    Dense(128, activation='relu'),  
    Dropout(0.5),  
    Dense(train_generator.num_classes, activation='softmax')  
])


# ### Melakukan proses Fine-Tuning

# In[12]:


base_model.trainable = True
fine_tune_at = 100 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# ### Compile Model

# In[13]:


model.compile(
    optimizer=Adam(learning_rate=1e-5),  
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)


# ### EarlyStopping

# In[14]:


early_stopping = EarlyStopping(
    monitor='val_loss',        
    patience=3,                
    restore_best_weights=True  
)


# ### Melatih Model

# In[15]:


history = model.fit(
    train_generator, 
    validation_data=validation_generator,  
    epochs=50,  
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)


# In[16]:


model.summary()


# ### Evaluasi Model

# In[17]:


validation_loss, validation_accuracy = model.evaluate(
    validation_generator,  
    steps=validation_generator.samples // validation_generator.batch_size  

print(f"Validation Loss: {validation_loss}")
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")


# ### Visualisasi 

# In[18]:


import matplotlib.pyplot as plt

# Grafik Akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Grafik Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[19]:


model.save("garbage_classification_model.h5")


# In[20]:


get_ipython().system('pip install streamlit tensorflow')


# In[24]:


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Memuat model yang sudah dilatih
model = load_model(r"C:\Users\MSI BRAVO 15\Downloads\garbage_classification_model.h5")

st.title("Garbage Classification")
st.write("Upload gambar untuk klasifikasi sampah.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Mengubah gambar menjadi array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # Normalisasi gambar
    img_array = img_array / 255.0

    # Prediksi kelas
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)
    class_names = list(train_generator.class_indices.keys())  # Menyusun nama kelas dari train_generator

    st.write(f"Predicted Class: {class_names[class_idx[0]]}")


# In[25]:


from tensorflow.keras.models import load_model
model = load_model('garbage_classification_model.h5')
print(model.summary())


# In[27]:


import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model(r"C:\Users\MSI BRAVO 15\Downloads\garbage_classification_model.h5")

# Convert to TensorFlow Lite format with float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the quantized model
with open("garbage_classification_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model has been quantized and saved as a TensorFlow Lite model.")

