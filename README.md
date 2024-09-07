## Dino Vision Project

### Contributors:
- **Zin Lynn Thant**
- **Hein Htet Aung (david chang)**

### Supervisor:
- **Tr. Cynthia**

### Institution:
- **Simbolo**

---

### Project Overview

Dino Vision is a computer vision-based classifier designed to identify 43 different species of dinosaurs. This project uses **TensorFlow** and **Keras** to build a Convolutional Neural Network (CNN) that processes augmented image data for each dinosaur species, ensuring that all classes are balanced with 600 images each.

---

### Dataset and Augmentation

The dataset consists of various dinosaur images categorized by species. We applied **image augmentation** techniques to ensure each class contains exactly 600 images, using **rotation**, **brightness adjustment**, **zoom**, and **horizontal flipping** to prevent overfitting and improve model performance.

---

### Image Augmentation Code

```python
# Augment data to achieve 400 images per species
for species in os.listdir(dataset_dir):
    species_dir = os.path.join(dataset_dir, species)
    augmented_species_dir = os.path.join(augmented_dir, species)
    os.makedirs(augmented_species_dir, exist_ok=True)

    image_count = len(os.listdir(species_dir))
    target_image_count = 400
    augmentations_needed = (target_image_count - image_count) // image_count + 1

    for image_name in os.listdir(species_dir):
        image_path = os.path.join(species_dir, image_name)
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_species_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= augmentations_needed:
                break
```

---

### Dataset Splitting

The dataset is split into three parts:
- **Training Set**: 70% of the total dataset
- **Validation Set**: 20% of the total dataset
- **Testing Set**: 10% of the total dataset

This splitting helps ensure the model generalizes well to unseen data.

```python
train_split = 0.7
valid_split = 0.2
test_split = 0.1

# Further code for splitting
```

---

### Model Architecture

The model is a **Convolutional Neural Network (CNN)** built using **Keras Sequential API**. The architecture includes **4 convolutional layers**, **max-pooling**, and a **dense layer** with **Dropout** to reduce overfitting. The final layer uses a **softmax** activation function, allowing classification across 43 dinosaur species.

#### Model Layers:
1. **Conv2D** layer with 32 filters
2. **MaxPooling2D** layer
3. **Conv2D** layer with 64 filters
4. **MaxPooling2D** layer
5. **Conv2D** layer with 128 filters
6. **MaxPooling2D** layer
7. **Flatten** layer
8. **Dropout (0.5)** for regularization
9. **Dense (512)** fully connected layer
10. **Dense (43)** output layer

```python
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(43, activation='softmax') # Output layer for 43 species
])
```

---

### Model Compilation

We used **categorical crossentropy** as the loss function since this is a multi-class classification problem. **Adam optimizer** was chosen for its efficiency, and **accuracy** was used as the performance metric.

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

### Training the Model

We trained the model using the **ImageDataGenerator** to rescale image data, with 15 epochs for convergence. The training generator processes images from the training set, while the validation generator evaluates the model’s performance during training.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)
```

---

### Testing the Model

After training, the model's performance is evaluated on the **testing set**. This ensures that the model generalizes well to completely unseen data.

```python
# Load test images using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')
```

---

### Accuracy and Performance Metrics

Once the model is trained and tested, we can plot the training and validation accuracy/loss to assess performance:

```python
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Final Test Accuracy:
The final test accuracy can be seen from the `test_accuracy` variable, which evaluates the model on the unseen test dataset.

---

### Conclusion

The **Dino Vision** project successfully classifies 43 species of dinosaurs using a CNN model. With data augmentation and Dropout regularization, the model achieves good accuracy while avoiding overfitting. Testing the model on an unseen dataset validates its generalization performance.
