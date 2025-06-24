from .StateSpaceModelLinear import StateSpaceModelLinear
from .GaussianMixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal


class SDWAAPHDUnit:
    """
    A single collaborative SD-WAA GM-PHD filter unit for one radar.
    """

    def __init__(
        self,
        fov: tuple[np.ndarray, float],
        model: StateSpaceModelLinear,
        birth_intensity: GaussianMixture,
        survival_probability: float,
        detection_probability: float,
        clutter_intensity_per_unit: float,
        estimate_threshold: float = 0.5,
        pruning_threshold: float = 0.001,
        merging_threshold: float = 3,
        max_gaussians: int = 100,
        gate_size: float = 3,
    ):
        """
        Initialize the Probability Hypothesis Density filter.

        :param fov: The field of view of the radar (center and radius).
        :param model: The state space model.
        :param birth_intensity: The birth intensity.
        :param survival_probability: The probability of a target's survival.
        :param detection_probability: The probability of a target's detection.
        :param clutter_intensity_total: The total clutter intensity.
        :param estimate_threshold: The threshold for selecting estimates.
        :param pruning_threshold: The threshold for pruning the intensity.
        :param merging_threshold: The threshold for merging components.
        :param max_gaussians: The maximum number of Gaussians to keep in the
            posterior intensity.
        :param gate_size: The size of the gate.
        """
        self.fov = fov
        self.model = model
        self.birth_intensity = birth_intensity
        self.survival_probability = survival_probability
        self.detection_probability = detection_probability
        self.clutter_intensity_per_unit = clutter_intensity_per_unit
        self.estimate_threshold = estimate_threshold
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.max_gaussians = max_gaussians
        self.gate_size = gate_size
        self.posterior_intensity = GaussianMixture()
        self.uncombined_intensity = GaussianMixture()
        self.estimates = GaussianMixture()

    def copy(self):
        """
        Create a deep copy of the SDWAAPHD unit.
        """
        new_unit = SDWAAPHDUnit(
            fov=self.fov,
            model=self.model,
            birth_intensity=self.birth_intensity,
            survival_probability=self.survival_probability,
            detection_probability=self.detection_probability,
            clutter_intensity_per_unit=self.clutter_intensity_per_unit,
            estimate_threshold=self.estimate_threshold,
            pruning_threshold=self.pruning_threshold,
            merging_threshold=self.merging_threshold,
            max_gaussians=self.max_gaussians,
            gate_size=self.gate_size,
        )

        new_unit.posterior_intensity = self.posterior_intensity.copy()
        new_unit.uncombined_intensity = self.uncombined_intensity.copy()
        new_unit.estimates = self.estimates.copy()

        return new_unit

    def predict(self) -> GaussianMixture:
        """
        Do a PHDF prediction based on the posterior intensity.

        :returns GaussianMixture: predicted intensity
        """
        new_weights = [
            self.survival_probability * weight
            for weight in self.posterior_intensity.weights
        ]

        new_means = [self.model.F @ mean for mean in self.posterior_intensity.means]

        new_covariances = [
            self.model.F @ covariance @ self.model.F.T + self.model.Q
            for covariance in self.posterior_intensity.covariances
        ]

        existing_prediction = GaussianMixture(new_means, new_covariances, new_weights)

        predicted_intensity = existing_prediction + self.birth_intensity

        return predicted_intensity

    def update(
        self, measurements: list[np.ndarray], prediction: GaussianMixture = None
    ):
        """
        Update the posterior intensity based on the predicted intensity and
        measurements.

        :param measurements: Collection of cluttered measurements.
        :param prediction: The predicted intensity (defaults to the posterior).
        """

        clutter_intensity = self.clutter_intensity_per_unit

        # construct the PHD update components
        predicted_measurements = [self.model.H @ mean for mean in prediction.means]

        innovation_covariances = [
            self.model.H @ covariance @ self.model.H.T + self.model.R
            for covariance in prediction.covariances
        ]

        kalman_gains = [
            prediction.covariances[i]
            @ self.model.H.T
            @ np.linalg.inv(innovation_covariances[i])
            for i in range(len(prediction))
        ]

        posterior_covariances = [
            (np.eye(prediction.dimension) - kalman_gains[i] @ self.model.H)
            @ prediction.covariances[i]
            for i in range(len(prediction))
        ]

        # calculate the detection probability for each predicted component
        detection_probability = [0] * len(prediction)
        gate_size_multiplier = self.gate_size

        for j in range(len(prediction)):
            mean = predicted_measurements[j]
            cov = innovation_covariances[j]

            cov_eigenvalues = np.linalg.eigvals(cov)
            max_gate_size = np.sqrt(max(cov_eigenvalues)) * gate_size_multiplier

            if not self.is_in_fov(mean, (self.fov[0], self.fov[1] - max_gate_size)):
                predictive_mass = self.predictive_mass_in_intersection(
                    mean, cov, self.fov
                )
                detection_probability[j] = predictive_mass * self.detection_probability
            else:
                detection_probability[j] = self.detection_probability

        not_detected_intensity = GaussianMixture(
            prediction.means,
            prediction.covariances,
            [
                (1 - detection_probability[j]) * prediction.weights[j]
                for j in range(len(prediction))
            ],
        )

        detected_intensity = GaussianMixture()

        # contribute to the detected intensity from each measurement
        for measurement in measurements:
            if not self.is_in_fov(measurement):
                continue
            weights = []
            means = []
            covariances = []

            # add a component for each predicted measurement
            for j in range(len(prediction)):
                # optimization
                if detection_probability[j] == 0:
                    continue

                mean = predicted_measurements[j]
                cov = innovation_covariances[j]

                mvn = multivariate_normal(mean=mean, cov=cov)
                likelihood = mvn.pdf(measurement)

                weight = detection_probability[j] * prediction.weights[j] * likelihood
                weights.append(weight)

                mean = prediction.means[j] + kalman_gains[j] @ (
                    measurement - predicted_measurements[j]
                )
                means.append(mean)
                covariances.append(posterior_covariances[j])

            # normalize the weights
            sum_of_weights = sum(weights)
            weights = [
                weight / (clutter_intensity + sum_of_weights) for weight in weights
            ]

            # add the mixture for this measurement to the detected intensity
            detected_intensity += GaussianMixture(means, covariances, weights)

        # update the posterior intensity
        self.posterior_intensity = detected_intensity + not_detected_intensity
        self.prune()
        self.uncombined_intensity = self.posterior_intensity

    def is_in_fov(
        self, point: np.ndarray, fov: tuple[np.ndarray, float] = None
    ) -> bool:
        """
        Check if a point is in the field of view of the radar.

        :param point: The point to check.
        :returns bool: True if the point is in the field of view.
        """
        if fov is None:
            fov = self.fov
        return np.linalg.norm(point - fov[0]) <= fov[1]

    def predictive_mass_in_intersection(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        neighbor_fov: tuple[np.ndarray, float],
        n_samples: int = 1000,
    ) -> float:
        """
        Calculate the predictive mass of a gaussian in the intersection of the FoVs.

        :param mean: The mean of the gaussian.
        :param cov: The covariance of the gaussian.
        :param neighbor_fov: The field of view of the neighboring radar.
        :param n_samples: The number of samples to use in the Monte Carlo estimation.
        """
        # Monte Carlo: sample n points from the gaussian and check if they are in the
        # intersection
        n_inside = 0
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        for sample in samples:
            if self.is_in_fov(sample) and self.is_in_fov(sample, neighbor_fov):
                n_inside += 1

        return n_inside / n_samples

    def prune(self):
        """
        Prune the posterior intensity.
        """
        # remove components with mean outside the FoV
        new_posterior = GaussianMixture()
        for i in range(len(self.posterior_intensity)):
            if self.is_in_fov(self.model.H @ self.posterior_intensity.means[i]):
                new_posterior += self.posterior_intensity[i]
        self.posterior_intensity = new_posterior

        # remove components with weights below the threshold
        self.posterior_intensity = self.posterior_intensity.min_weight(
            self.pruning_threshold
        )

        # merge similar components
        self.merge_similar_components()

        # keep only the max_gaussians most weighted components
        self.posterior_intensity = self.posterior_intensity.top_k(self.max_gaussians)

    def merge_similar_components(self):
        """
        Merge similar components in the posterior intensity.
        """
        merged = GaussianMixture()
        not_merged = self.posterior_intensity

        while len(not_merged) > 0:
            # find the most weighted component
            most_weighted_index = not_merged.argmax_weight()
            most_weighted = not_merged[most_weighted_index]

            # remove the most weighted component from the not_merged list
            not_merged = not_merged.remove_component(most_weighted_index)

            # find the components that are similar to the most weighted
            similar_component_indices = not_merged.similar_component_indices(
                most_weighted, self.merging_threshold
            )

            # merge the most weighted component with the similar components
            components_to_merge = [most_weighted] + [
                not_merged[i] for i in similar_component_indices
            ]

            merged_weight = sum(
                component.weights[0] for component in components_to_merge
            )

            merged_mean = (
                sum(
                    component.weights[0] * component.means[0]
                    for component in components_to_merge
                )
                / merged_weight
            )

            merged_covariance = (
                sum(
                    component.weights[0]
                    * (
                        component.covariances[0]
                        + np.outer(
                            component.means[0] - merged_mean,
                            component.means[0] - merged_mean,
                        )
                    )
                    for component in components_to_merge
                )
                / merged_weight
            )

            merged += GaussianMixture(merged_mean, merged_covariance, merged_weight)

            # remove the similar components from the not_merged list
            not_merged = not_merged.remove_components(similar_component_indices)

        self.posterior_intensity = merged

    def get_estimates(self):
        """
        Get the current estimates.
        """
        self.estimates = self.posterior_intensity.min_weight(self.estimate_threshold)
