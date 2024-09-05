import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load your trained model (ensure you have uploaded your model file)
model = tf.keras.models.load_model('dinosaur_classification_model_latestupdated.h5')

# Function to predict image
def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
st.title('Dinosaur Species Classifying App')

# Allow all image types by not restricting the file type
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    label = predict_image(uploaded_file)
    st.write(f'Prediction: {label}')
