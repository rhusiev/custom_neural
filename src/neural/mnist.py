"""Create a neural network that can XOR two numbers."""
import csv
from random import SystemRandom

import numpy as np

from neural.network import Network


def read_mnist_csv(path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read the mnist csv file and return the inputs and desired outputs.

    Args:
        path (str): The path to the mnist csv file.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The inputs and desired outputs.
    """
    inputs = []
    desired = []
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # Omit the first line
        next(reader)
        for row in reader:
            inputs.append(np.array(row[1:], dtype=float))
            desired.append(np.array([float(row[0])]))
    return inputs, desired


shuffle = SystemRandom().shuffle

print("Reading mnist dataset...")
inputs, desired = read_mnist_csv("mnist_dataset/mnist_train.csv")
print("Done reading mnist dataset.")

# Make desired outputs into vectors. For example, 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
desired = [np.array([1 if i == j else 0 for i in range(10)]) for j in desired]
DESIRED_LENGTH = len(desired)

indexes = list(range(DESIRED_LENGTH))
shuffle(indexes)

LENGTH = 20

# network = Network([784, 80, 24, 10], "sigmoid", 0.1)
# network.generate_weights_and_biases()
network = Network.load("mnist")
print("Loaded network. Starting training...")

try:
    for i in range(LENGTH):
        print(f"Epoch {i+1}/{LENGTH}")
        shuffle(indexes)
        correct = 0
        for j in indexes:
            forward = network.forward(inputs[j])
            network.backward(desired[j])
            if np.argmax(forward) == np.argmax(desired[j]):
                correct += 1
        print(
            f"Accuracy: {correct / DESIRED_LENGTH * 100}% ({correct}/{DESIRED_LENGTH})"
        )
except KeyboardInterrupt:
    print("Stopping training...")

print("Saving network...")
network.save("mnist")
print("Saved network.")
