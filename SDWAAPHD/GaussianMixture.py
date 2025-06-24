from __future__ import annotations
import numpy as np
from .helpers import bhattacharyya_distance


class GaussianMixture:
    def __init__(
        self,
        mean: np.ndarray | list[np.ndarray] = None,
        covariance: np.ndarray | list[np.ndarray] = None,
        weight: float | list[float] = 1,
    ):
        """
        Initialize a GaussianMixture object that can handle either:
        - A single Gaussian component (mean, covariance, weight), or
        - A list of Gaussian components (means, covariances, weights).
        """

        # If no mean and covariance are provided, create empty mixture
        if mean is None and covariance is None:
            self.means = []
            self.covariances = []
            self.weights = []

        # If a single mean, covariance, and weight are provided,
        # convert them to lists
        elif (
            isinstance(mean, np.ndarray)
            and isinstance(covariance, np.ndarray)
            and isinstance(weight, (float, int))
        ):
            self.means = [mean]
            self.covariances = [covariance]
            self.weights = [weight]

        # Multiple components case
        elif isinstance(mean, list) and isinstance(covariance, list):
            assert len(mean) == len(
                covariance
            ), "Means and covariances must have the same length"

            if isinstance(weight, (float, int)):
                weight = [weight] * len(mean)

            assert len(weight) == len(
                mean
            ), "The length of weights must match the number of components"

            self.means = mean
            self.covariances = covariance
            self.weights = weight

        # incompatible types
        else:
            raise ValueError(
                "Inputed values should be all lists or all " "single values"
            )

        # Check dimensionality consistency for all components
        dimension = None
        for m, c in zip(self.means, self.covariances):
            dimension = m.shape[0]
            assert (
                c.shape[0] == dimension and c.shape[1] == dimension
            ), "Covariance matrix must match the dimensionality \
                of the mean vector"
        self.dimension = dimension

    def copy(self) -> GaussianMixture:
        """
        Return a copy of the GaussianMixture object.
        """
        return GaussianMixture(
            [mean.copy() for mean in self.means],
            [covariance.copy() for covariance in self.covariances],
            self.weights.copy(),
        )

    def __repr__(self):
        """
        Return a string representation of the GaussianMixture object.
        """
        return (
            f"GaussianMixture(means={self.means}, "
            + f"covariances={self.covariances}, weights={self.weights})"
        )

    def __len__(self):
        """
        Get the number of Gaussian components.
        """
        return len(self.means)

    def __getitem__(self, index: int):
        """
        Get a Gaussian component by index.
        """
        return GaussianMixture(
            self.means[index], self.covariances[index], self.weights[index]
        )

    def __iter__(self):
        """
        Iterate over the Gaussian components.
        """
        return iter(
            GaussianMixture(mean, covariance, weight)
            for mean, covariance, weight in zip(
                self.means, self.covariances, self.weights
            )
        )

    def __add__(self, other: GaussianMixture):
        """
        Add two GaussianMixture objects together.
        """
        assert (
            self.dimension == other.dimension
            or self.dimension is None
            or other.dimension is None
        )

        means = self.means + other.means
        covariances = self.covariances + other.covariances
        weights = self.weights + other.weights

        return GaussianMixture(means, covariances, weights)

    def __mul__(self, scalar: float | int):
        """
        Multiply the GaussianMixture by a scalar.
        """
        weights = [scalar * weight for weight in self.weights]

        return GaussianMixture(self.means, self.covariances, weights)

    def __rmul__(self, scalar: float | int):
        """
        Multiply the GaussianMixture by a scalar.
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float | int):
        """
        Divide the GaussianMixture by a scalar.
        """
        weights = [weight / scalar for weight in self.weights]

        return GaussianMixture(self.means, self.covariances, weights)

    def __rtruediv__(self, scalar: float | int):
        """
        Divide the GaussianMixture by a scalar.
        """
        return self.__truediv__(scalar)

    def __eq__(self, other: GaussianMixture):
        """
        Check if two GaussianMixture objects are equal.
        """
        return (
            self.means == other.means
            and self.covariances == other.covariances
            and self.weights == other.weights
        )

    def __ne__(self, other: GaussianMixture):
        """
        Check if two GaussianMixture objects are not equal.
        """
        return not self == other

    def sum_of_weights(self) -> float:
        """
        Return the sum of the weights of all components.
        """
        return sum(self.weights)

    def sample(self) -> list[np.ndarray]:
        """
        Sample from the GaussianMixture.
        """
        mean_number_of_outcomes = self.sum_of_weights()
        # sample a number of outcomes from poisson distribution
        number_of_outcomes = np.random.poisson(mean_number_of_outcomes)
        # sample the outcomes
        outcomes = []
        for _ in range(number_of_outcomes):
            # select a gaussian with probability proportional to its weight
            normalized_weights = [
                weight / mean_number_of_outcomes for weight in self.weights
            ]
            gaussian = np.random.choice(range(len(self.weights)), p=normalized_weights)
            # sample from the selected gaussian
            outcomes.append(
                np.random.multivariate_normal(
                    self.means[gaussian], self.covariances[gaussian]
                )
            )
        return outcomes

    def min_weight(self, threshold: float) -> GaussianMixture:
        """
        Remove components with weights below a threshold.
        """
        means = []
        covariances = []
        weights = []

        for mean, covariance, weight in zip(self.means, self.covariances, self.weights):
            if weight >= threshold:
                means.append(mean)
                covariances.append(covariance)
                weights.append(weight)

        return GaussianMixture(means, covariances, weights)

    def max_weight(self, threshold: float) -> GaussianMixture:
        """
        Remove components with weights above a threshold.
        """
        means = []
        covariances = []
        weights = []

        for mean, covariance, weight in zip(self.means, self.covariances, self.weights):
            if weight <= threshold:
                means.append(mean)
                covariances.append(covariance)
                weights.append(weight)

        return GaussianMixture(means, covariances, weights)

    def top_k(self, k: int) -> GaussianMixture:
        """
        Keep only the k components with the highest weights.
        """
        if k >= len(self):
            return self.copy()

        # sort the components by weight
        sorted_components = sorted(
            zip(self.means, self.covariances, self.weights),
            key=lambda x: x[2],
            reverse=True,
        )

        # keep the k components with the highest weights
        means = [component[0] for component in sorted_components[:k]]
        covariances = [component[1] for component in sorted_components[:k]]
        weights = [component[2] for component in sorted_components[:k]]

        return GaussianMixture(means, covariances, weights)

    def argmax_weight(self) -> int:
        """
        Return the index of the component with the highest weight.
        """
        return np.argmax(self.weights)

    def remove_component(self, index: int) -> GaussianMixture:
        """
        Remove a component by index.
        """
        means = self.means.copy()
        covariances = self.covariances.copy()
        weights = self.weights.copy()

        means.pop(index)
        covariances.pop(index)
        weights.pop(index)

        return GaussianMixture(means, covariances, weights)

    def remove_components(self, indices: list[int]) -> GaussianMixture:
        """
        Remove components by indices.
        """
        means = self.means.copy()
        covariances = self.covariances.copy()
        weights = self.weights.copy()

        for index in sorted(indices, reverse=True):
            means.pop(index)
            covariances.pop(index)
            weights.pop(index)

        return GaussianMixture(means, covariances, weights)

    def similar_component_indices(
        self, component: GaussianMixture, threshold: float
    ) -> list[int]:
        """
        Find the indices of components similar to a given component.
        """
        similar_indices = []
        for i, (mean, covariance) in enumerate(zip(self.means, self.covariances)):
            if (
                bhattacharyya_distance(
                    mean, covariance, component.means[0], component.covariances[0]
                )
                < threshold
            ):
                similar_indices.append(i)
        return similar_indices

    def closest_component(self, component: GaussianMixture) -> tuple[int, float]:
        """
        Get the index and distance of the closest component to a given component.
        """
        closest_index = None
        closest_distance = float("inf")
        for i, (mean, covariance) in enumerate(zip(self.means, self.covariances)):
            distance = bhattacharyya_distance(
                mean, covariance, component.means[0], component.covariances[0]
            )
            if distance < closest_distance:
                closest_index = i
                closest_distance = distance
        return closest_index, closest_distance

    def total_weight(self) -> float:
        """
        Return the sum of the weights of all components.
        """
        return sum(self.weights)
