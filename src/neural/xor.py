"""Create a neural network that can XOR two numbers."""
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

LENGTH = 100
plot = DynamicPlot((0, LENGTH), (0, 110), "XOR, [3, 4, 2, 1], relu, 0.1")

network = Network([3, 4, 2, 1], "relu", 0.1)
network.generate_weights_and_biases()
previous_correct = -1.0
for j in range(LENGTH):
    correct = 0.0
    checked = 0
    for _ in range(LENGTH):
        shuffle(indexes)
        for i in indexes:
            inp = inputs[i]
            print(network.forward(inp), desired[i])
            network.backward(desired[i])
            if abs(network.outputs - desired[i]) < 0.1:
                correct += 1
            elif abs(network.outputs - desired[i]) < 0.5:
                correct += 1 - abs(network.outputs[0] - desired[i])
            checked += 1
    plot.update(j, int(correct / checked * 100))
    if correct / checked == 1:
        break
    if (correct / checked - previous_correct) * (correct / checked + 1) < 0.0005:
        network.adjust_weights_and_biases()
    previous_correct = correct / checked

plot.show()

# network.save("xor")
