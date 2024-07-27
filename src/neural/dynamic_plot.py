"""A module to plot data in real time."""
import matplotlib.pyplot as plt
import numpy as np


class DynamicPlot:
    """A class to plot data in real time.

    Attributes:
        x_data (np.array): The x data to plot.
        y_data (np.array): The y data to plot.
        axes (plt.gca): The axes to plot on.
    """

    def __init__(
        self,
        x_boundaries: tuple[int, int],
        y_boundaries: tuple[int, int],
        title: str = "",
    ) -> None:
        self.x_data = np.array([])
        self.y_data = np.array([])
        plt.show()
        plt.title(title)
        plt.grid()
        self.axes = plt.gca()
        self.axes.set_xlim(x_boundaries)
        self.axes.set_ylim(y_boundaries)
        (self.line,) = self.axes.plot(self.x_data, self.y_data, "r-")

    def update(self, x_val: int, y_val: int) -> None:
        """Update the plot.

        Args:
            x_val  (int): The new x value.
            y_val  (int): The new y value.
        """
        self.x_data = np.append(self.x_data, x_val)
        self.y_data = np.append(self.y_data, y_val)
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        plt.draw()
        plt.pause(0.0001)

    def show(self) -> None:
        """Show the plot."""
        plt.show()
