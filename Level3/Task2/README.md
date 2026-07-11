# Handwritten Digit Classification with a Deep Neural Network (MNIST)

## Overview

This project implements a **Deep Neural Network (DNN)** for handwritten digit recognition using the **MNIST dataset**. The model learns to classify grayscale images of handwritten digits (0–9) by training a fully connected neural network built with TensorFlow/Keras.

The project demonstrates the complete deep learning workflow, including data preprocessing, model design, training, evaluation, and prediction.

---

## Features

* Loads the MNIST handwritten digit dataset
* Normalizes image pixel values
* Builds a multi-layer fully connected neural network
* Trains the model using backpropagation and the Adam optimizer
* Evaluates classification performance on unseen test data
* Predicts handwritten digits from test images
* Visualizes sample predictions

---

## Dataset

The project uses the **MNIST** dataset, which contains:

* **60,000** training images
* **10,000** testing images
* Image size: **28 × 28 pixels**
* 10 output classes (digits **0–9**)

Each image is a grayscale handwritten digit represented as pixel intensity values.

---

## Project Structure

```
deep-learning-mnist/
│
├── main.py              # Main script
├── model.py             # Neural network architecture
├── data.py              # Data loading and preprocessing
├── visualize.py         # Visualization utilities
├── requirements.txt
└── README.md
```

---

## Model Architecture

The neural network consists of:

* Flatten input layer
* Dense hidden layer with ReLU activation
* Dense hidden layer with ReLU activation
* Output layer with Softmax activation

Typical architecture:

```
Input (28×28)
      │
   Flatten
      │
 Dense (128, ReLU)
      │
 Dense (64, ReLU)
      │
Dense (10, Softmax)
```

The output layer produces probability scores for each digit class.

---

## Training

The model is trained using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Evaluation Metric:** Accuracy

During training, the network updates its weights through backpropagation to minimize classification error.

---

## Results

After training, the model achieves high classification accuracy on the MNIST test dataset (typically above **97%**, depending on training configuration and random initialization).

The trained model can correctly recognize most handwritten digits from unseen images.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/deep-learning-mnist.git
cd deep-learning-mnist
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python main.py
```

The program will:

1. Load the dataset
2. Preprocess the images
3. Build the neural network
4. Train the model
5. Evaluate test accuracy
6. Display sample predictions

---

## Skills Demonstrated

* Deep Learning
* Neural Networks
* TensorFlow
* Keras
* Image Classification
* Data Preprocessing
* Model Evaluation
* Computer Vision Fundamentals

---

## Future Improvements

Possible enhancements include:

* Convolutional Neural Networks (CNNs)
* Dropout regularization
* Batch normalization
* Learning rate scheduling
* Hyperparameter tuning
* Model checkpointing
* Data augmentation
* Saving and loading trained models

---

## License

This project is intended for educational purposes and portfolio demonstration.
