"""Layers of the neural network."""
import numpy as np


def sigmoid(array: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: Sigmoid of the input array
    """
    return 1 / (1 + np.exp(-array))


def sigmoid_derivative(array: np.ndarray) -> np.ndarray:
    """Sigmoid derivative.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: Sigmoid derivative of the input array
    """
    sig = sigmoid(array)
    return sig * (1 - sig)


def relu(array: np.ndarray) -> np.ndarray:
    """Relu activation function.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: ReLU of the input array
    """
    return np.maximum(0, array)


def relu_derivative(array: np.ndarray) -> np.ndarray:
    """Relu derivative.

    Args:
        array (np.ndarray): Input array

    Returns:
        np.ndarray: ReLU derivative of the input array
    """
    return np.where(array > 0, 1, 0)


ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "sigmoid_derivative": sigmoid_derivative,
    "relu_derivative": relu_derivative,
}


class Layer:
    """A generic Layer.

    Takes the number of input nodes and output nodes Creates biases
    and weights for outgoing connections.

    Attributes:
        input_nodes (int): Number of input nodes
        output_nodes (int): Number of output nodes
        weights (np.ndarray): Weights of the layer
        biases (np.ndarray): Biases of the layer
        inputs (np.ndarray): Inputs of the layer
    """

    input_nodes: int
    output_nodes: int
    weights: np.ndarray
    biases: np.ndarray
    inputs: np.ndarray

    def __init__(self, input_nodes: int, output_nodes: int) -> None:
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def generate_weights_and_biases(self) -> None:
        """Generates weights and biases for the layer."""
        self.weights = np.random.randn(self.output_nodes, self.input_nodes)
        self.biases = np.random.randn(self.output_nodes)

    def adjust_weights_and_biases(self) -> None:
        """Adjusts weights and biases, if the network is stuck."""
        self.weights += np.random.randn(self.output_nodes, self.input_nodes) * 0.1
        self.biases += np.random.randn(self.output_nodes) * 0.1

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation Adds biases and applies activation funciton.

        Args:
            inputs (np.ndarray): already weighted array of inputs

        Returns:
            np.ndarray: Forward propagated array
        """
        self.inputs = inputs
        return np.dot(self.weights, inputs) + self.biases

    def backward(self, error_derivative: np.ndarray, learn_rate: float) -> np.ndarray:
        """Backward propagation Calculates the error_derivative for the previous layer.

        Args:
            error_derivative (np.ndarray): Error derivative of the current layer
            learn_rate (float): learn rate of the network

        Returns:
            np.ndarray: error_derivative of the previous layer
        """
        weights_derivative = np.outer(error_derivative, self.inputs)
        self.weights -= learn_rate * weights_derivative
        self.biases -= learn_rate * error_derivative
        return np.dot(self.weights.T, error_derivative)


class ActivationLayer:
    """A generic activation layer.

    Args:
        activation_function (str): Activation function of the layer
    """

    activation_function: str

    def __init__(self, nodes: int, activation_function: str) -> None:
        self.input_nodes = nodes
        self.output_nodes = nodes
        self.activation_function = activation_function

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Use activation function on the inputs.

        Args:
            inputs (np.ndarray): The values of the previous layer,
                which should undergo the activation

        Returns:
            np.ndarray: Activated values
        """
        self.inputs = inputs
        if self.activation_function in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[self.activation_function](inputs)
        return inputs

    def backward(self, error_derivative: np.ndarray) -> np.ndarray:
        """Backward propagation Calculates the error_derivative for the previous layer.

        Args:
            error_derivative (np.ndarray): Error derivative of the current layer

        Returns:
            np.ndarray: error_derivative of the previous layer
        """
        return np.multiply(
            error_derivative,
            ACTIVATION_FUNCTIONS[self.activation_function + "_derivative"](self.inputs),
        )
