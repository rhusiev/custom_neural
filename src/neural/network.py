"""Neural networks."""
from __future__ import annotations

import os

import numpy as np

from .layers import ActivationLayer, Layer


class Network:
    """A generic neural network.

    Attributes:
        layers (list[Layer]): List of layers in the network
        activation_function (str): Activation function of the network
        learn_rate (float): Learn rate of the network
        end_activation_function (str): Activation function of the last layer
        outputs (np.ndarray): Outputs of the network
    """

    layers: list[Layer | ActivationLayer]
    activation_function: str
    learn_rate: float
    end_activation_function: str
    outputs: np.ndarray

    def __init__(
        self,
        layers: list[int],
        activation_function: str,
        learn_rate: float = 0.1,
        end_activation_function: str = "sigmoid",
    ) -> None:
        self.layers = []
        self.activation_function = activation_function
        self.end_activation_function = end_activation_function
        self.learn_rate = learn_rate
        for i in range(len(layers) - 2):
            self.layers.append(Layer(layers[i], layers[i + 1]))
            self.layers.append(ActivationLayer(layers[i + 1], activation_function))
        self.layers.append(Layer(layers[-2], layers[-1]))
        self.layers.append(ActivationLayer(layers[-1], self.end_activation_function))

    def generate_weights_and_biases(self) -> None:
        """Generate random weights and biases for each layer."""
        for layer in self.layers:
            if isinstance(layer, ActivationLayer):
                continue
            layer.generate_weights_and_biases()

    def adjust_weights_and_biases(self) -> None:
        """Adjusts weights and biases, if the network is stuck."""
        for layer in self.layers:
            if isinstance(layer, ActivationLayer):
                continue
            layer.adjust_weights_and_biases()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation.

        Args:
            inputs (np.ndarray): Inputs to the network

        Returns:
            np.ndarray: Outputs of the network
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        self.outputs = inputs
        return inputs

    def get_cost(self, desired: np.ndarray) -> float:
        """Get the cost of the network.

        Args:
            desired (np.ndarray): Desired outputs

        Returns:
            float: Cost of the network
        """
        return np.sum(np.power(self.outputs - desired, 2)) / len(desired)

    def backward(self, desired: np.ndarray) -> None:
        """Backward propagation.

        Args:
            desired (np.ndarray): The desired outputs
        """
        cost_derivative = 2 * (self.outputs - desired) / np.size(desired)
        for layer in reversed(self.layers):
            if isinstance(layer, ActivationLayer):
                cost_derivative = layer.backward(cost_derivative)
            if isinstance(layer, Layer):
                cost_derivative = layer.backward(cost_derivative, self.learn_rate)

    def save(self, name: str = "network") -> None:
        """Save the network to a file.

        Save weights and biases of each layer to a file.
        Save activation functions list as well

        Args:
            name (str, optional): Name of the files. Defaults to "network".
        """
        if not os.path.exists(name):
            os.mkdir(name)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, ActivationLayer):
                continue
            np.save(os.path.join(name, f"layer_{i}_weights.npy"), layer.weights)
            np.save(os.path.join(name, f"layer_{i}_biases.npy"), layer.biases)

        with open(os.path.join(name, "metadata.txt"), "w") as f:
            f.write(self.activation_function)
            f.write("\n")
            f.write(self.end_activation_function)
            f.write("\n")
            for layer in self.layers:
                if isinstance(layer, ActivationLayer):
                    continue
                f.write(f"{layer.input_nodes} ")
            f.write(str(self.layers[-1].output_nodes))
            f.write("\n")
            f.write(str(self.learn_rate))

    @staticmethod
    def load(name: str = "network") -> Network:
        """Load the network from a file.

        Load weights and biases of each layer from a file.
        The files must be in a directory with the name `name`.
        Each file must be named `layer_{i}_weights.npy` or `layer_{i}_biases.npy`.

        Args:
            name (str, optional): Name of the files. Defaults to "network".

        Returns:
            Network: The loaded network
        """
        with open(os.path.join(name, "metadata.txt"), "r") as f:
            activation_function = f.readline().strip()
            end_activation_function = f.readline().strip()
            layers = [int(i) for i in f.readline().strip().split(" ")]
            learn_rate = float(f.readline().strip())

        network = Network(
            layers, activation_function, learn_rate, end_activation_function
        )

        for i, layer in enumerate(network.layers):
            if isinstance(layer, ActivationLayer):
                continue
            layer.weights = np.load(os.path.join(name, f"layer_{i}_weights.npy"))
            layer.biases = np.load(os.path.join(name, f"layer_{i}_biases.npy"))

        return network
