from .StateSpaceModel import StateSpaceModel
import numpy as np


class StateSpaceModelLinear(StateSpaceModel):
    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        """
        Initialize the state space model.

        :param F: The state transition matrix.
        :param H: The observation matrix.
        :param Q: The process noise covariance matrix.
        :param R: The measurement noise covariance matrix.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        assert F.shape[0] == F.shape[1], "F must be square"
        assert Q.shape[0] == Q.shape[1], "Q must be square"
        assert R.shape[0] == R.shape[1], "R must be square"
        assert F.shape[0] == Q.shape[0], "F must be the same size as Q"
        assert (
            H.shape[0] == R.shape[0]
        ), "H must have the same number of rows \
            as R"

    def __repr__(self):
        """
        Return a string representation of the StateSpaceModelLinear object.
        """
        return f"StateSpaceModelLinear(F={self.F}, H={self.H}, \
            Q={self.Q}, R={self.R})"

    def evolve(self, x_old: np.ndarray) -> np.ndarray:
        """
        Evolve the state using the state equation
        $x_{k+1} = Fx_k + w_k, w_k ~ N(0, Q)$.

        :param x_old: The old state.
        :return: The new state.
        """
        assert (
            x_old.shape[0] == self.F.shape[0]
        ), "x_old must have the same \
            dimension as F"

        return self.F @ x_old + np.random.multivariate_normal(
            np.zeros(self.F.shape[0]), self.Q
        )

    def measure(self, x: np.ndarray) -> np.ndarray:
        """
        Measure the state using the observation equation
        $z_k = Hx_k + v_k, v_k ~ N(0, R)$.

        :param x: The state.
        :return: The measurement.
        """
        assert (
            x.shape[0] == self.H.shape[1]
        ), "x must have the same number of \
            compounds as H has columns"

        return self.H @ x + np.random.multivariate_normal(
            np.zeros(self.H.shape[0]), self.R
        )
