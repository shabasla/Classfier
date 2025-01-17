# Image Classification Using Convolutional Neural Networks (CNNs)

This repository contains an image classification project built using Keras and TensorFlow. The model is trained on a set of images, and the trained model predicts labels for new test images. The project utilizes convolutional neural networks (CNNs) with multiple convolutional layers, dropout, and pooling layers for image classification.

## Project Overview

1. **Dataset**: 
   - The dataset consists of labeled images that are located in the `Train` folder.
   - Each class of images is contained in its own subfolder inside the `Train` directory.
   - The model is trained to predict two classes (binary classification).

2. **Model**:
   - The model is a custom CNN architecture consisting of several Conv2D layers with different kernel sizes, followed by MaxPooling2D and AveragePooling2D layers. 
   - After the convolutional layers, the model uses Flatten and Dense layers to make the final predictions.
   - The model is compiled with the Adam optimizer and Huber loss.

3. **Training**:
   - The model is trained for 20 epochs with a batch size of 20.
   - Training images are loaded, resized to a fixed size of 289x289 pixels, and normalized.

4. **Prediction**:
   - After training, the model predicts labels for images in the test set, located in the `Test_Images` folder.
   - The predictions are then written to a CSV file for submission.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- scikit-learn
- tqdm

To install these dependencies, you can use:

```bash
pip install tensorflow keras pandas numpy scikit-learn tqdm
