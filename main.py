import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load your trained model (ensure you have uploaded your model file)
model = tf.keras.models.load_model('dinovision/dinosaur_classification_model_latestupdated.h5')

# List of actual class names (dinosaur species)
class_names = [
    'Allosaurus', 'Dilophosaurus', 'Mamenchisaurus', 'Pachyrhinosaurus', 'Stygimoloch',
    'Ankylosaurus', 'Dimetrodon', 'Microceratus', 'Parasaurolophus', 'Suchomimus',
    'Apatosaurus', 'Dimorphodon', 'Monolophosaurus', 'Pteranodon', 'Tarbosaurus',
    'Baryonyx', 'Dreadnoughtus', 'Mosasaurus', 'Pyroraptor', 'Therizinosaurus',
    'Brachiosaurus', 'Gallimimus', 'Nasutoceratops', 'Quetzalcoatlus', 'Triceratops',
    'Carnotaurus', 'Giganotosaurus', 'Nothosaurus', 'Sinoceratops', 'Tyrannosaurus rex',
    'Ceratosaurus', 'Iguanodon', 'Ouranosaurus', 'Smilodon', 'Velociraptor',
    'Compsognathus', 'Kentrosaurus', 'Oviraptor', 'Spinosaurus',
    'Corythosaurus', 'Lystrosaurus', 'Pachycephalosaurus', 'Stegosaurus'
]

# Function to predict image
def predict_image(image):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize the uploaded image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image to [0, 1]
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get the index of the class with the highest prediction
    predicted_class = class_names[predicted_class_index]  # Get the class name
    
    return predicted_class

# Streamlit app
st.title('Dinosaur Species Classifying App')

# Allow all image types by not restricting the file type
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Use PIL to open the image from the BytesIO object
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Call the predict function with the image object
    predicted_class = predict_image(image)

    # Display the predicted class (species name)
    st.write(f'Prediction: {predicted_class}')

