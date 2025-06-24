from .StateSpaceModelLinear import StateSpaceModelLinear
from .GaussianMixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal
from typing import Literal
from scipy.spatial.distance import mahalanobis


class DiffPHDUnit:
    """
    A single collaborative diffusion GM-PHD filter unit for one radar.
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
        self_referencing_adapt: bool = True,
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
        :param self_referencing_adapt: Enable self-referencing adaptation.
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
        self.self_referencing_adapt = self_referencing_adapt
        self.posterior_intensity = GaussianMixture()
        self.uncombined_intensity = GaussianMixture()
        self.estimates = GaussianMixture()
        self.peripheral_intensity = GaussianMixture()

    def copy(self):
        """
        Create a deep copy of the DiffPHD unit.
        """
        new_unit = DiffPHDUnit(
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
            self_referencing_adapt=self.self_referencing_adapt,
        )

        new_unit.posterior_intensity = self.posterior_intensity.copy()
        new_unit.uncombined_intensity = self.uncombined_intensity.copy()
        new_unit.estimates = self.estimates.copy()

        return new_unit

    def predict(self, intensity=None) -> GaussianMixture:
        """
        Do a PHDF prediction based on the posterior intensity.

        :param intensity: The intensity to predict (defaults to the posterior).

        :returns GaussianMixture: predicted intensity
        """
        if intensity is None:
            intensity = self.posterior_intensity

        new_weights = [
            self.survival_probability * weight for weight in intensity.weights
        ]

        new_means = [self.model.F @ mean for mean in intensity.means]

        new_covariances = [
            self.model.F @ covariance @ self.model.F.T + self.model.Q
            for covariance in intensity.covariances
        ]

        existing_prediction = GaussianMixture(new_means, new_covariances, new_weights)

        predicted_intensity = existing_prediction + self.birth_intensity

        return predicted_intensity

    def update(
        self,
        measurements: list[np.ndarray],
        prediction: GaussianMixture = None,
        neighbor_fov: list[tuple[np.ndarray, float]] = None,
        neighbor_clutter_intensity_per_unit: float = None,
        neighbor_detection_probability: float = None,
        neighbor_R: np.ndarray = None,
    ):
        """
        Update the posterior intensity based on the predicted intensity and
        measurements.

        :param measurements: Collection of cluttered measurements.
        :param prediction: The predicted intensity (defaults to the posterior).
        """

        # determine if the update is an adaptation and edit the behavior accordingly
        is_adapt = False
        if prediction is None:
            prediction = self.posterior_intensity
            is_adapt = True

        clutter_intensity = self.clutter_intensity_per_unit
        if is_adapt:
            clutter_intensity = neighbor_clutter_intensity_per_unit

        # predict and select peripheral components
        if not is_adapt:
            # predict the peripherals
            predicted_peripheral = self.predict(self.peripheral_intensity)
            # intensity used, empty it
            self.peripheral_intensity = GaussianMixture()
            kept_peripheral = GaussianMixture()

            # filter the predicted peripherals
            for i in range(len(predicted_peripheral)):
                mean = self.model.H @ predicted_peripheral.means[i]
                cov = (
                    self.model.H @ predicted_peripheral.covariances[i] @ self.model.H.T
                )
                # find if there is any measurement in gate
                for measurement in measurements:
                    distance = mahalanobis(measurement, mean, cov)
                    if distance < self.gate_size:
                        kept_peripheral += predicted_peripheral[i]
                        break

            # add the kept peripheral components to the prediction intensity
            prediction += kept_peripheral

        if neighbor_R is None:
            R = self.model.R
        else:
            R = neighbor_R

        # construct the PHD update components
        predicted_measurements = [self.model.H @ mean for mean in prediction.means]

        innovation_covariances = [
            self.model.H @ covariance @ self.model.H.T + R
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

        # if adapting, consider the intersection of the FoVs
        if is_adapt:
            fov_mean = neighbor_fov[0]
            fov_radius = neighbor_fov[1]

            for j in range(len(prediction)):
                mean = predicted_measurements[j]
                cov = innovation_covariances[j]

                cov_eigenvalues = np.linalg.eigvals(cov)
                max_gate_size = np.sqrt(max(cov_eigenvalues)) * gate_size_multiplier

                # gate outside the intersection
                if not self.is_in_fov(mean, (fov_mean, fov_radius + max_gate_size)):
                    detection_probability[j] = 0

                # gate inside the intersection
                elif self.is_in_fov(
                    mean, (fov_mean, fov_radius - max_gate_size)
                ) and self.is_in_fov(mean, (self.fov[0], self.fov[1] - max_gate_size)):
                    detection_probability[j] = neighbor_detection_probability

                # gate on the edge of the intersection
                else:
                    predictive_mass = self.predictive_mass_in_intersection(
                        mean, cov, (fov_mean, fov_radius)
                    )
                    detection_probability[j] = (
                        predictive_mass * neighbor_detection_probability
                    )
        # if just updating, consider the FoV of the radar
        else:
            for j in range(len(prediction)):
                mean = predicted_measurements[j]
                cov = innovation_covariances[j]

                cov_eigenvalues = np.linalg.eigvals(cov)
                max_gate_size = np.sqrt(max(cov_eigenvalues)) * gate_size_multiplier

                if not self.is_in_fov(mean, (self.fov[0], self.fov[1] - max_gate_size)):
                    predictive_mass = self.predictive_mass_in_intersection(
                        mean, cov, self.fov
                    )
                    detection_probability[j] = (
                        predictive_mass * self.detection_probability
                    )
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

        had_measurement_in_gate = [False] * len(prediction)

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

                # check if in gate
                distance = mahalanobis(measurement, mean, cov)
                if distance <= self.gate_size:
                    had_measurement_in_gate[j] = True
                else:
                    if is_adapt and self.self_referencing_adapt:
                        continue

                mvn = multivariate_normal(mean=mean, cov=cov)
                likelihood = mvn.pdf(measurement)

                weight = detection_probability[j] * prediction.weights[j] * likelihood
                weights.append(weight)

                mean = prediction.means[j] + kalman_gains[j] @ (
                    measurement - predicted_measurements[j]
                )
                means.append(mean)
                covariances.append(posterior_covariances[j])

            if len(weights) == 0:
                continue

            # normalize the weights
            sum_of_weights = sum(weights)
            weights = [
                weight / (clutter_intensity + sum_of_weights) for weight in weights
            ]

            # add the mixture for this measurement to the detected intensity
            detected_intensity += GaussianMixture(means, covariances, weights)

        # keep the components that did not have a measurement in gate (self-referencing)
        if is_adapt and self.self_referencing_adapt:
            for j in range(len(prediction)):
                if not had_measurement_in_gate[j]:
                    detected_intensity += detection_probability[j] * prediction[j]

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

    def combine(
        self,
        neighbor_intensity: GaussianMixture,
        combine_strategy: Literal["gci", "mm", "uwaa", "cwaa"] = "gci",
        share_components: bool = True,
    ):
        """
        Combine the posterior intensity with the intensity of a neighboring radar.

        :param neighbor_intensity: The intensity of the neighboring radar.
        :param combine_strategy: Strategy for merging gaussian components.
        :param share_components: Share close nonmerged components between radars.
        """
        # find closest neighbor component for each component in the posterior intensity
        combined_intensity = GaussianMixture()
        merged_neighbor_indices = set()
        for posterior_component in self.posterior_intensity:
            closest_neighbor_index, closest_neighbor_distance = (
                neighbor_intensity.closest_component(posterior_component)
            )

            # if there is no closest component in the distance threshold, do not merge
            # using the same distance thershold as for PHD merging
            if (
                closest_neighbor_index is None
                or closest_neighbor_distance > self.merging_threshold
            ):
                combined_intensity += posterior_component
                continue

            # merge
            closest_neighbor = neighbor_intensity[closest_neighbor_index]
            merged_neighbor_indices.add(closest_neighbor_index)

            posterior_mean = posterior_component.means[0]
            posterior_covariance = posterior_component.covariances[0]
            posterior_weight = posterior_component.weights[0]

            neighbor_mean = closest_neighbor.means[0]
            neighbor_covariance = closest_neighbor.covariances[0]
            neighbor_weight = closest_neighbor.weights[0]

            # covariance intersection
            # Battistelli et al. 10.1109/JSTSP.2013.2250911
            if combine_strategy == "gci":
                # uniform combination weights (1/2 and 1/2)
                combination_weight = 0.5

                posterior_covariance_inv = np.linalg.inv(posterior_covariance)
                neighbor_covariance_inv = np.linalg.inv(neighbor_covariance)

                combined_covariance = np.linalg.inv(
                    (posterior_covariance_inv + neighbor_covariance_inv)
                    * combination_weight
                )

                combined_mean = (
                    combination_weight
                    * combined_covariance
                    @ (
                        posterior_covariance_inv @ posterior_mean
                        + neighbor_covariance_inv @ neighbor_mean
                    )
                )

                def eps(covariance, combination_weight):
                    return np.sqrt(
                        np.linalg.det(2 * np.pi * covariance / combination_weight)
                        / (np.linalg.det(2 * np.pi * covariance)) ** combination_weight
                    )

                posterior_eps = eps(posterior_covariance, combination_weight)
                neighbor_eps = eps(neighbor_covariance, combination_weight)

                mvn = multivariate_normal(
                    np.zeros_like(posterior_mean),
                    (posterior_covariance + neighbor_covariance) / combination_weight,
                )
                likelihood = mvn.pdf(posterior_mean - neighbor_mean)

                combined_weight = (
                    posterior_weight**combination_weight
                    * neighbor_weight**combination_weight
                    * posterior_eps
                    * neighbor_eps
                    * likelihood
                )

            # moment matching
            elif combine_strategy == "mm":
                weight_sum = posterior_weight + neighbor_weight

                combined_mean = (
                    posterior_weight * posterior_mean + neighbor_weight * neighbor_mean
                ) / weight_sum

                posterior_difference = posterior_mean - combined_mean
                posterior_dispersion = np.outer(
                    posterior_difference, posterior_difference
                )
                neighbor_difference = neighbor_mean - combined_mean
                neighbor_dispersion = np.outer(neighbor_difference, neighbor_difference)

                combined_covariance = (
                    (posterior_weight * (posterior_covariance + posterior_dispersion))
                    + (neighbor_weight * (neighbor_covariance + neighbor_dispersion))
                ) / weight_sum

                combined_weight = weight_sum / 2

            # uniformly weighted arithmetic average
            elif combine_strategy == "uwaa":
                combined_mean = (posterior_mean + neighbor_mean) / 2

                posterior_difference = posterior_mean - combined_mean
                posterior_dispersion = np.outer(
                    posterior_difference, posterior_difference
                )
                neighbor_difference = neighbor_mean - combined_mean
                neighbor_dispersion = np.outer(neighbor_difference, neighbor_difference)

                combined_covariance = (
                    (posterior_covariance + posterior_dispersion)
                    + (neighbor_covariance + neighbor_dispersion)
                ) / 2

                combined_weight = (posterior_weight + neighbor_weight) / 2

            # confidence weighted arithmetic average
            elif combine_strategy == "cwaa":
                posterior_confidence = 1 / np.trace(posterior_covariance)
                neighbor_confidence = 1 / np.trace(neighbor_covariance)
                confidence_sum = posterior_confidence + neighbor_confidence

                combined_mean = (
                    posterior_confidence * posterior_mean
                    + neighbor_confidence * neighbor_mean
                ) / confidence_sum

                posterior_difference = posterior_mean - combined_mean
                posterior_dispersion = np.outer(
                    posterior_difference, posterior_difference
                )
                neighbor_difference = neighbor_mean - combined_mean
                neighbor_dispersion = np.outer(neighbor_difference, neighbor_difference)

                combined_covariance = (
                    posterior_confidence * (posterior_covariance + posterior_dispersion)
                    + neighbor_confidence * (neighbor_covariance + neighbor_dispersion)
                ) / confidence_sum

                combined_weight = (
                    posterior_confidence * posterior_weight
                    + neighbor_confidence * neighbor_weight
                ) / confidence_sum

            # invalid combine strategy
            else:
                raise ValueError(f"Unknown combine strategy: {combine_strategy}")

            combined_intensity += GaussianMixture(
                combined_mean, combined_covariance, combined_weight
            )

        # rescale weights to match the original posterior
        if combine_strategy == "gci":
            posterior_total_weight = self.posterior_intensity.total_weight()
            combined_total_weight = combined_intensity.total_weight()
            combined_intensity /= combined_total_weight
            combined_intensity *= posterior_total_weight

        self.posterior_intensity = combined_intensity

        # add peripheral components (non-merged and at least partly in the fov)
        if share_components:
            for i in range(len(neighbor_intensity)):
                if i in merged_neighbor_indices:
                    continue

                mean = self.model.H @ neighbor_intensity.means[i]
                cov = self.model.H @ neighbor_intensity.covariances[i] @ self.model.H.T

                cov_eigenvalues = np.linalg.eigvals(cov)
                max_gate_size = np.sqrt(max(cov_eigenvalues)) * self.gate_size

                if self.is_in_fov(
                    mean,
                    (self.fov[0], self.fov[1] + max_gate_size),
                ):
                    self.peripheral_intensity += neighbor_intensity

    def get_estimates(self):
        """
        Get the current estimates.
        """
        self.estimates = self.posterior_intensity.min_weight(self.estimate_threshold)
