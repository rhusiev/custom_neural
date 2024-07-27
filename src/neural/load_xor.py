"""Test loading XOR neural network and its accuracy."""
from random import SystemRandom

import numpy as np

from neural.dynamic_plot import DynamicPlot
from neural.network import Network

shuffle = SystemRandom().shuffle

inputs = [
    np.array(i)
    for i in (
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    )
]
desired = np.array([[0], [0], [0], [0], [1], [1], [1], [0]])

indexes = list(range(len(inputs)))

network = Network.load("xor")

correct = 0.0
checked = 0
for _ in range(100):
    shuffle(indexes)
    for i in indexes:
        inp = inputs[i]
        print(network.forward(inp), desired[i])
        if abs(network.outputs - desired[i]) < 0.5:
            correct += 1
        checked += 1

print(f"Accuracy: {correct / checked * 100}%")
