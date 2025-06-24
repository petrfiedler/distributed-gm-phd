import numpy as np


class StateSpaceModel:
    """
    Abstract class for a state space model.
    """
    def __repr__(self):
        """
        Return a string representation of the StateSpaceModelLinear object.
        """
        return "StateSpaceModel()"

    def evolve(self, x_old: np.ndarray) -> np.ndarray:
        """
        Evolve the state using the state equation.

        :param x_old: The old state.
        :return: The new state.
        """
        pass

    def measure(self, x: np.ndarray) -> np.ndarray:
        """
        Measure the state using the observation equation.

        :param x: The state.
        :return: The measurement.
        """
        pass
