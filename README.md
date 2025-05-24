# MNIST Digit Classifier from Scratch

A simple implementation of a two-layer neural network for classifying handwritten digits from the MNIST dataset, built entirely from scratch(why?...coz I hate my sanity) using NumPy.

## Features

- **Two-Layer Neural Network**: Basic feedforward neural network with one hidden layer
- **ReLU Activation**: Rectified Linear Unit activation for the hidden layer
- **Softmax Output**: Probability distributions over 10 digit classes
- **Backpropagation**: Efficient gradient calculation algorithm
- **Gradient Descent**: Vanilla gradient descent for weight updates
- **Numerical Stability**: Stable Softmax implementation to prevent overflow
- **Accuracy Monitoring**: Tracks and prints accuracy during training
- **Jupyter Notebook**: Easy-to-follow implementation in notebook format

## Getting Started

### Prerequisites

You'll need the following Python libraries:
- numpy
- pandas
- matplotlib

Install them using pip:
```bash
pip install numpy pandas matplotlib
```
## Dataset
This project uses the train.csv file from the MNIST "Digit Recognizer" Kaggle competition.

## Running the Notebook
- Clone this repository or download the folder
- Ensure you have train.csv in the same directory
- Open the Jupyter Notebook:
```bash
jupyter notebook mnist-from-scratch.ipynb
```
Run all cells in the notebook

## Code Structure

The core logic includes functions for:

| Function | Description |
|----------|-------------|
| [`init_params()`](#init_params) | Initializes weights and biases |
| [`ReLU(Z)`](#relu) | ReLU activation function |
| [`softMax(Z)`](#softmax) | Numerically stable Softmax |
| [`forward_prop(W1, B1, W2, B2, X)`](#forward_prop) | Forward pass through network |
| [`one_hot(Y)`](#one_hot) | One-hot encodes labels |
| [`deriv_ReLU(Z)`](#deriv_relu) | Derivative of ReLU |
| [`back_prop(Z1, A1, Z2, A2, W2, X, Y)`](#back_prop) | Backpropagation algorithm |
| [`update_params(...)`](#update_params) | Updates weights and biases |
| [`get_predictions(A2)`](#get_predictions) | Converts to predicted labels |
| [`get_accuracy(predictions, Y)`](#get_accuracy) | Calculates accuracy |
| [`gradient(X, Y, iterations, alpha)`](#gradient) | Main training loop |
