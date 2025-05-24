MNIST Digit Classifier from Scratch
This repository contains a simple implementation of a two-layer neural network for classifying handwritten digits from the MNIST dataset, built entirely from scratch using NumPy.

Features
Two-Layer Neural Network: Implements a basic feedforward neural network with one hidden layer.

ReLU Activation: Uses the Rectified Linear Unit (ReLU) activation function for the hidden layer.

Softmax Output: Employs the Softmax function for the output layer to provide probability distributions over the 10 digit classes.

Backpropagation: Implements the backpropagation algorithm for efficient gradient calculation.

Gradient Descent: Uses vanilla gradient descent for updating weights and biases.

Numerical Stability: Includes a numerically stable implementation of the Softmax function to prevent overflow issues.

Accuracy Monitoring: Tracks and prints accuracy during training.

Jupyter Notebook: The code is provided in a Jupyter Notebook (mnist-from-scratch.ipynb) for easy understanding and execution.

Getting Started
Prerequisites
You'll need the following Python libraries installed:

numpy

pandas

matplotlib

You can install them using pip:

pip install numpy pandas matplotlib

Dataset
This project uses the train.csv file from the MNIST "Digit Recognizer" Kaggle competition.
Download train.csv and place it in the same directory as the notebook.

Running the Notebook
Clone this repository or download the mnist-from-scratch.ipynb file.

Ensure you have the train.csv dataset in the same directory.

Open the Jupyter Notebook:

jupyter notebook mnist-from-scratch.ipynb

Run all cells in the notebook.

Code Structure
The core logic is contained within the mnist-from-scratch.ipynb file, with functions defined for:

init_params(): Initializes weights and biases for the neural network.

ReLU(Z): Implements the ReLU activation function.

softMax(Z): Implements the numerically stable Softmax activation function.

forward_prop(W1, B1, W2, B2, X): Performs the forward pass through the network.

one_hot(Y): Converts digit labels into one-hot encoded vectors.

deriv_ReLU(Z): Calculates the derivative of the ReLU function.

back_prop(Z1, A1, Z2, A2, W2, X, Y): Implements the backpropagation algorithm to compute gradients.

updatee_params(W1, B1, W2, B2, dW1, db1, dW2, db2, alpha): Updates the model's weights and biases using gradient descent.

get_predictions(A2): Converts Softmax probabilities into predicted digit labels.

get_accuracy(predictions, Y): Calculates the classification accuracy.

gradient(X, Y, iterations, alpha): The main training loop that orchestrates the entire learning process.

How it Works
The neural network processes the 784-pixel input images (28x28) through a hidden layer of 10 neurons with ReLU activation, followed by an output layer of 10 neurons (one for each digit 0-9) with Softmax activation. The gradient function iteratively performs forward propagation to get predictions, then uses backpropagation to calculate gradients, and finally updates the network's parameters (weights and biases) using gradient descent to minimize the classification error.

Optimization Notes
During development, several optimizations were implemented:

Numerically Stable Softmax: Modified softMax to subtract the maximum value before exponentiation, preventing overflow and NaN issues.

Correct Bias Updates: Ensured db1 and db2 sums were correctly applied with axis=1 and keepdims=True in back_prop.

Refined Weight Initialization: Initialized weights with a smaller scale (* 0.01) and biases to zeros to promote more stable training.

Data Transposition: Correctly shaped input data (X_train, X_dev) to have features as rows and samples as columns, aligning with common neural network conventions.

One-Hot Encoding Fixes: Corrected the one_hot function's dimension creation to properly encode labels.

Hyperparameter Tuning: It was observed that a smaller learning rate (alpha, e.g., 0.001 or 0.0001) was crucial for the model to learn effectively, overcoming initial random-guessing performance.

License
This project is open-source and available under the MIT License.
