from .SDWAAPHDUnit import SDWAAPHDUnit
from .StateSpaceModel import StateSpaceModel
from .GaussianMixture import GaussianMixture
import numpy as np


class SDWAAPHD:
    """
    SD-WAA GM-PHD filter network.
    """

    def __init__(
        self,
        graph: dict[str, list[str]],
        fovs: dict[str, tuple[np.ndarray, float]],
        models: dict[str, StateSpaceModel] = None,
        birth_intensity: GaussianMixture = None,
        survival_probability: float = 1.0,
        detection_probabilities: dict[str, float] | float = 1.0,
        global_clutter_intensity_per_unit: float = 0,
        local_clutter_intensities_per_unit: dict[str, float] | float = 0,
        estimate_threshold: float = 0.5,
        pruning_threshold: float = 0.001,
        merging_threshold: float = 3,
        max_gaussians: int = 100,
    ):
        """
        Initialize the SD-WAA PHD filter network.

        :param graph: The graph of the radar network given by a list of neighbors
            (their ids).
        :param fovs: The field of views of each radar (center and radius).
        :param models: The state space models for each radar.
        :param birth_intensity: The birth intensity.
        :param survival_probability: The probability of a target's survival.
        :param detection_probabilities: The probabilities of a target's detection for
            each radar.
        :param global_clutter_intensity_per_unit: The global clutter intensity per unit.
        :param local_clutter_intensities_per_unit: The local clutter intensities per
            unit for each radar.
        :param estimate_threshold: The threshold for selecting estimates.
        :param pruning_threshold: The threshold for pruning the intensity.
        :param merging_threshold: The threshold for merging components.
        :param max_gaussians: The maximum number of Gaussians to keep in the posterior
            intensity of each SDWAAPHD unit.
        """
        self.graph = graph
        self.fovs = fovs
        self.models = models
        self.radar_ids = list(graph.keys())
        if not isinstance(detection_probabilities, dict):
            detection_probabilities = {
                radar_id: detection_probabilities for radar_id in self.radar_ids
            }
        if not isinstance(local_clutter_intensities_per_unit, dict):
            local_clutter_intensities_per_unit = {
                radar_id: local_clutter_intensities_per_unit
                for radar_id in self.radar_ids
            }

        self.phds = {
            id: SDWAAPHDUnit(
                fov=fovs[id],
                model=models[id],
                birth_intensity=birth_intensity,
                survival_probability=survival_probability,
                detection_probability=detection_probabilities[id],
                clutter_intensity_per_unit=global_clutter_intensity_per_unit
                + local_clutter_intensities_per_unit[id],
                estimate_threshold=estimate_threshold,
                pruning_threshold=pruning_threshold,
                merging_threshold=merging_threshold,
                max_gaussians=max_gaussians,
            )
            for id in self.radar_ids
        }

    def copy(self):
        """
        Create a deep copy of the SD-WAA PHD filter network.
        """
        new_sdwaaphd = SDWAAPHD(
            graph=self.graph,
            fovs=self.fovs,
            models=self.models,
            birth_intensity=self.phds[self.radar_ids[0]].birth_intensity,
            survival_probability=self.phds[self.radar_ids[0]].survival_probability,
            detection_probabilities={
                id: phd.detection_probability for id, phd in self.phds.items()
            },
            global_clutter_intensity_per_unit=self.phds[
                self.radar_ids[0]
            ].clutter_intensity_per_unit,
            local_clutter_intensities_per_unit={
                id: phd.clutter_intensity_per_unit for id, phd in self.phds.items()
            },
            estimate_threshold=self.phds[self.radar_ids[0]].estimate_threshold,
            pruning_threshold=self.phds[self.radar_ids[0]].pruning_threshold,
            merging_threshold=self.phds[self.radar_ids[0]].merging_threshold,
            max_gaussians=self.phds[self.radar_ids[0]].max_gaussians,
        )

        new_sdwaaphd.phds = {id: phd.copy() for id, phd in self.phds.items()}

        return new_sdwaaphd

    def predict(self) -> dict[str, GaussianMixture]:
        """
        Do a SDWAAPHD prediction based on the posterior intensities.

        :returns dict[str, GaussianMixture]: predicted intensities of each radar
        """
        return {id: phd.predict() for id, phd in self.phds.items()}

    def update(
        self,
        predictions: dict[str, GaussianMixture],
        measurements: dict[str, list[np.ndarray]],
    ):
        """
        Update the posterior intensities based on the predicted intensities and
        measurements taking advantage of the collaborative network.

        :param predictions: The predicted intensities of each radar.
        :param measurements: The measurements of each radar.
        """
        for id in self.radar_ids:
            self.phds[id].update(measurements[id], predictions[id])

        self.mix_neighbors_components()

        for id in self.radar_ids:
            self.phds[id].get_estimates()

    def mix_neighbors_components(self):
        """
        Mix the components of the posterior intensities of the SDWAAPHD units
        based on the collaborative network.
        """
        for id in self.radar_ids:
            phd = self.phds[id]
            neighbors = self.graph[id]

            # get neighbors components that are in the FoV
            neighbors_components = GaussianMixture()
            for neighbor in neighbors:
                neighbor_intensity = self.phds[neighbor].posterior_intensity
                for component in neighbor_intensity:
                    component_center = component.means[0][:2]
                    if phd.is_in_fov(component_center):
                        neighbors_components += component

            # mix the components
            components_to_mix = phd.posterior_intensity + neighbors_components
            reweighted_components = GaussianMixture()

            for component in components_to_mix:
                # find the number of sensors which have the component's position within their FOV
                in_fov = 1
                for neighbor in neighbors:
                    component_center = component.means[0][:2]
                    if self.phds[neighbor].is_in_fov(component_center):
                        in_fov += 1

                component /= in_fov
                reweighted_components += component

            phd.posterior_intensity = reweighted_components
            phd.prune()

    def step(self, measurements: dict[str, list[np.ndarray]]):
        """
        Perform one SDWAAPHD step based on the measurements.

        :param measurements: The measurements of each radar.
        """
        predictions = self.predict()
        self.update(predictions, measurements)

    def set_estimate_threshold(self, threshold: float):
        """
        Set the estimate threshold for selecting estimates.

        :param threshold: The threshold for selecting estimates.
        """
        for id in self.radar_ids:
            self.phds[id].estimate_threshold = threshold
