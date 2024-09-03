# MLP from Scratch

This project implements a simple Multi-Layer Perceptron (MLP) neural network from scratch in Python. The implementation includes basic components such as neurons, layers, loss calculation, and optimization, with a focus on understanding the core principles behind neural networks without relying on external libraries like TensorFlow or PyTorch.

## Table of Contents
- [Introduction](#introduction)
- [Classes and Methods](#classes-and-methods)
  - [Tensor](#tensor-class)
  - [Neuron](#neuron-class)
  - [Layer](#layer-class)
  - [MLP](#mlp-class)
  - [Optimizer](#optimizer-class)
- [Functions](#functions)
  - [compute_loss](#compute_loss-function)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Introduction

This project demonstrates the construction of a Multi-Layer Perceptron (MLP) using Python. It includes the creation of basic neural network components, gradient calculation through backpropagation, and a simple optimizer for training the network. The project is designed to give a clear understanding of how neural networks work under the hood.

## Classes and Methods

### Tensor Class

The `Tensor` class is used to represent and manage the weights and biases of the network. It also supports gradient calculation and interaction within the computational graph.

- **`__init__`**: Initializes the tensor with an initial value, typically a scalar, along with its children, the last operation, and the node's gradient.
- **`__repr__`**: Controls how the tensor instance is displayed when printed.
- **Operator Methods**: Methods like `__mul__`, `__add__`, `__pow__`, and `__sub__` are used for operations such as multiplication, addition, exponentiation, and subtraction, and they also compute the gradients.
- **`backward`**: Computes and updates gradients in reverse order through the computational graph.

### Neuron Class

The `Neuron` class represents a single neuron in the network.

- **`__init__`**: Creates a neuron with random weights and bias.
- **`forward`**: Calculates the neuron's output using the hyperbolic tangent activation function.
- **`__call__`**: Allows the neuron to be invoked as a function.
- **`parameters`**: Returns a list of the neuron's weights and bias.

### Layer Class

The `Layer` class represents a layer of neurons.

- **`__init__`**: Creates a layer with the specified number of neurons.
- **`forward`**: Computes the output of the layer using its neurons.
- **`__call__`**: Allows the layer to be invoked as a function.
- **`parameters`**: Returns the weights and biases of all neurons in the layer.

### MLP Class

The `MLP` class represents the multi-layer neural network.

- **`__init__`**: Creates a multi-layer network with a given number of inputs and specified layer sizes.
- **`forward`**: Computes the network's output using all its layers.
- **`__call__`**: Allows the network to be invoked as a function.
- **`parameters`**: Returns the weights and biases of all neurons across all layers.

### Optimizer Class

The `Optimizer` class is used to update the network's weights and biases based on the calculated gradients.

- **`__init__`**: Initializes the optimizer with the network's parameters and a learning rate.
- **`zero_grad`**: Resets all gradients to zero, preparing them for the next update.
- **`step`**: Updates the network's weights and biases using the calculated gradients and the learning rate.

## Functions

### `compute_loss` Function

The `compute_loss` function calculates the loss of the network, which is used to measure the difference between the predicted and actual values.

- **Parameters**: 
  - `y_hats`: Network's predictions.
  - `real_y`: Actual values.
- **Functionality**: Calculates the squared error for each prediction and actual value pair, sums these errors, and returns the overall network loss.
- **Gradient Calculation**: Initializes the gradient for the overall network error, which is used in backpropagation.

## Usage

To use this MLP implementation:

1. Create an instance of the `MLP` class with the desired input size and layer configuration.
2. Define an `Optimizer` with the network's parameters and a learning rate.
3. In each training cycle:
   - Pass the inputs through the network to get predictions.
   - Calculate the loss using `compute_loss`.
   - Perform backpropagation to compute gradients.
   - Use the optimizer's `step` function to update the weights and biases.
4. Print or log the training information such as the loss value and cycle number.

## Example

Here’s a brief example of how to set up and train an MLP using this implementation:

```python
# Create the MLP with specified layers
mlp = MLP(input_size=3, layer_sizes=[4, 4, 1])

# Define the optimizer
optim = Optimizer(mlp.parameters(), learning_rate=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    y_hats = mlp(x)

    # Compute loss
    loss = compute_loss(y_hats, real_y)

    # Backpropagation
    loss.backward()

    # Update weights and biases
    optim.step()

    # Zero gradients
    optim.zero_grad()

    # Print training progress
    print(f'Epoch {epoch}: Loss = {loss}')
