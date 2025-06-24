from .StateSpaceModelLinear import StateSpaceModelLinear
import numpy as np


class CVM(StateSpaceModelLinear):
    """
    2D Constant Velocity Model.
    """

    def __init__(self, q: float = 0.04, r: float = 0.4, dt: float = 1):
        """
        Initialize the Constant Velocity Model.

        :param q: The process noise intensity.
        :param r: The measurement noise intensity.
        :param dt: The time step.
        """
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.Q = q * np.array(
            [
                [dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**3 / 3, 0, dt**2 / 2],
                [dt**2 / 2, 0, dt, 0],
                [0, dt**2 / 2, 0, dt],
            ]
        )

        self.R = r * np.eye(2)
