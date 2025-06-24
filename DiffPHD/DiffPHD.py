from .DiffPHDUnit import DiffPHDUnit
from .StateSpaceModel import StateSpaceModel
from .CVM import CVM
from .GaussianMixture import GaussianMixture
import numpy as np
from typing import Literal


class DiffPHD:
    """
    Diffusion GM-PHD filter network.
    """

    def __init__(
        self,
        graph: dict[str, list[str]],
        fovs: dict[str, tuple[np.ndarray, float]],
        adapt: bool = True,
        self_referencing_adapt: bool = True,
        combine: bool = True,
        combine_strategy: Literal["gci", "mm", "uwaa", "cwaa"] = "gci",
        share_components: bool = True,
        model: StateSpaceModel = CVM(),
        birth_intensity: GaussianMixture = None,
        sensor_birth_intensities: dict[str, GaussianMixture] = None,
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
        Initialize the Diffusion PHD filter network.

        :param graph: The graph of the radar network given by a list of neighbors
            (their ids).
        :param fovs: The field of views of each radar (center and radius).
        :param adapt: Enable the adaptation step.
        :param self_referencing_adapt: Enable self-referencing adaptation.
        :param combine: Enable the combination step.
        :param combine_strategy: Strategy of merging components in the combine step.
        :param share_components: Share nearby components between radars in the combine step.
        :param model: The state space model.
        :param birth_intensity: The birth intensity.
        :param sensor_birth_intensities: The birth intensities for each radar.
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
            intensity of each DiffPHD unit.
        """
        self.adapt = adapt
        self.combine = combine
        self.combine_strategy = combine_strategy
        self.share_components = share_components
        self.graph = graph
        self.fovs = fovs
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

        # set default birth intensity for sensor with no explicit birth intensity
        for id in self.radar_ids:
            if sensor_birth_intensities is not None:
                if id not in sensor_birth_intensities:
                    sensor_birth_intensities[id] = birth_intensity
            else:
                sensor_birth_intensities = {
                    id: birth_intensity for id in self.radar_ids
                }

        self.phds = {
            id: DiffPHDUnit(
                fov=fovs[id],
                model=model,
                birth_intensity=sensor_birth_intensities[id],
                survival_probability=survival_probability,
                detection_probability=detection_probabilities[id],
                clutter_intensity_per_unit=global_clutter_intensity_per_unit
                + local_clutter_intensities_per_unit[id],
                estimate_threshold=estimate_threshold,
                pruning_threshold=pruning_threshold,
                merging_threshold=merging_threshold,
                max_gaussians=max_gaussians,
                self_referencing_adapt=self_referencing_adapt,
            )
            for id in self.radar_ids
        }

    def copy(self):
        """
        Create a deep copy of the Diffusion PHD filter network.
        """
        new_diff_phd = DiffPHD(
            graph=self.graph,
            fovs=self.fovs,
            adapt=self.adapt,
            combine=self.combine,
            combine_strategy=self.combine_strategy,
            share_components=self.share_components,
            model=self.phds[self.radar_ids[0]].model,
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
            self_referencing_adapt=self.phds[self.radar_ids[0]].self_referencing_adapt,
        )

        new_diff_phd.phds = {id: phd.copy() for id, phd in self.phds.items()}

        return new_diff_phd

    def predict(self) -> dict[str, GaussianMixture]:
        """
        Do a DiffPHD prediction based on the posterior intensities.

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

        if self.adapt:
            for id in self.radar_ids:
                neighbors = self.graph[id]
                for neighbor_id in neighbors:
                    self.phds[id].update(
                        measurements[neighbor_id],
                        neighbor_fov=self.fovs[neighbor_id],
                        neighbor_clutter_intensity_per_unit=self.phds[
                            neighbor_id
                        ].clutter_intensity_per_unit,
                        neighbor_detection_probability=self.phds[
                            neighbor_id
                        ].detection_probability,
                        neighbor_R=self.phds[neighbor_id].model.R,
                    )

        if self.combine:
            for id in self.radar_ids:
                combination_intensity = self.phds[id].uncombined_intensity
                neighbors = self.graph[id]
                for neighbor_id in neighbors:
                    self.phds[neighbor_id].combine(
                        combination_intensity,
                        self.combine_strategy,
                        self.share_components,
                    )

        for id in self.radar_ids:
            self.phds[id].get_estimates()

    def step(self, measurements: dict[str, list[np.ndarray]]):
        """
        Perform one DiffPHD step based on the measurements.

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
