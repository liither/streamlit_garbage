{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c55b0fdd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-12-17 20:03:10.760 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Public\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "# Class names for prediction\n",
    "CLASS_NAMES = [\n",
    "    \"battery\", \"biological\", \"brown-glass\", \"cardboard\", \"clothes\",\n",
    "    \"green-glass\", \"metal\", \"paper\", \"plastic\", \"shoes\", \"trash\", \"white-glass\"\n",
    "]\n",
    "\n",
    "# Function to preprocess the uploaded image\n",
    "def preprocess_image(uploaded_file):\n",
    "    \"\"\"\n",
    "    Preprocess the uploaded image for prediction.\n",
    "    Resize the image to (150, 150) and normalize pixel values.\n",
    "    \"\"\"\n",
    "    image = Image.open(uploaded_file)\n",
    "    image = image.resize((150, 150))\n",
    "    image_array = np.array(image) / 255.0  # Normalize to [0, 1]\n",
    "    return image_array\n",
    "\n",
    "# Function to load the trained model\n",
    "def load_trained_model():\n",
    "    \"\"\"\n",
    "    Load the fine-tuned Keras model for garbage classification.\n",
    "    \"\"\"\n",
    "    model_path = \"garbage_classification_model.h5\"\n",
    "    model = tf.keras.models.load_model(model_path)\n",
    "    return model\n",
    "\n",
    "# Function to predict the class of an image\n",
    "def predict_class(model, image_array):\n",
    "    \"\"\"\n",
    "    Predict the class of the image using the trained model.\n",
    "    \"\"\"\n",
    "    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension\n",
    "    predictions = model.predict(image_array)\n",
    "    predicted_index = np.argmax(predictions)\n",
    "    return CLASS_NAMES[predicted_index]\n",
    "\n",
    "# Streamlit App Logic\n",
    "st.title(\"Garbage Classification with Fine-Tuned Model\")\n",
    "st.write(\"Upload an image to classify garbage.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Choose an image file\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file:\n",
    "    # Preprocess the uploaded image\n",
    "    image = preprocess_image(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    # Load the trained model\n",
    "    model = load_trained_model()\n",
    "\n",
    "    # Predict the class\n",
    "    prediction = predict_class(model, image)\n",
    "\n",
    "    # Display the prediction result\n",
    "    st.write(f\"Predicted Class: **{prediction}**\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b5d107d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "# Build MobileNetV2 model with fine-tuning\n",
    "def build_mobilenetv2_finetune_model():\n",
    "    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')\n",
    "    base_model.trainable = True  # Enable fine-tuning\n",
    "    \n",
    "    # Add classification layers\n",
    "    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()\n",
    "    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification\n",
    "    \n",
    "    # Assemble the model\n",
    "    inputs = tf.keras.Input(shape=(224, 224, 3))\n",
    "    x = base_model(inputs, training=True)\n",
    "    x = global_average_layer(x)\n",
    "    outputs = prediction_layer(x)\n",
    "    model = tf.keras.Model(inputs, outputs)\n",
    "    \n",
    "    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),\n",
    "                  loss='binary_crossentropy',\n",
    "                  metrics=['accuracy'])\n",
    "    return model\n",
    "\n",
    "# Train and save the model (replace `train_data` and `val_data` with your datasets)\n",
    "model = build_mobilenetv2_finetune_model()\n",
    "# model.fit(train_data, epochs=5, validation_data=val_data)\n",
    "model.save('mobilenetv2_finetuned.h5')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
