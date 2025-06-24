from .DiffPHD import DiffPHD
from .Radar import Radar
from .RadarNetwork import RadarNetwork
from .StateSpaceModel import StateSpaceModel
from .CVM import CVM
from .GaussianMixture import GaussianMixture
from .Animator import Animator
import numpy as np
from .gospapy import calculate_gospa as gospa
from typing import Literal


class Simulator:
    """
    A simulator for the collaborative diffusion GM-PHD filter network.
    """

    def __init__(
        self,
        radars: list[Radar],
        graph: dict[str, list[str]] = None,
        adapt: bool = True,
        self_referencing_adapt: bool = True,
        combine: bool = True,
        combine_strategy: Literal["gci", "mm", "uwaa", "cwaa"] = "gci",
        share_components: bool = True,
        model: StateSpaceModel = CVM(),
        birth_intensity: GaussianMixture = None,
        sensor_birth_intensities: dict[str, GaussianMixture] = None,
        survival_probability: float = 1.0,
        global_clutter_intensity_per_unit: float = 0,
        estimate_threshold: float = 0.5,
        pruning_threshold: float = 0.001,
        merging_threshold: float = 3,
        max_gaussians: int = 100,
        animation_save: bool = False,
        animation_fps: int = 5,
        animation_loop: bool = True,
        animation_figsize: tuple[int, int] = (15, 15),
        seed: int = None,
    ):
        """
        Initialize the simulator.

        :param radars: The radars in the network.
        :param graph: The graph of the radar network given by a list of neighbors
            (their ids).
        :param adapt: Enable the adaptation step.
        :param self_referencing_adapt: Enable self-referencing adaptation.
        :param combine: Enable the combination step.
        :param combine_strategy: Strategy of merging components in the combine step.
        :param share_components: Share nearby components between radars in the combine step.
        :param model: The state space model.
        :param birth_intensity: The birth intensity.
        :param sensor_birth_intensities: The birth intensities for each radar.
        :param survival_probability: The probability of a target's survival.
        :param global_clutter_intensity_per_unit: The global clutter intensity per unit.
        :param estimate_threshold: The threshold for selecting estimates.
        :param pruning_threshold: The threshold for pruning the intensity.
        :param merging_threshold: The threshold for merging components.
        :param max_gaussians: The maximum number of Gaussians to keep in the posterior
            intensity of each DiffPHD unit.
        :param animation_save: Save the animation.
        :param animation_fps: The frames per second of the animation.
        :param animation_loop: Loop the animation.
        :param animation_figsize: The size of the animation figure.
        :param seed: The random seed.
        """
        self.seed = seed

        # initialize the radar network
        self.radar_network = RadarNetwork(
            radars=radars,
            model=model,
            birth_intensity=birth_intensity,
            survival_probability=survival_probability,
            global_clutter_intensity_per_unit=global_clutter_intensity_per_unit,
        )

        # if graph is not given, create a fully connected graph
        if graph is None:
            graph = {
                radar.id: [r.id for r in radars if r.id != radar.id] for radar in radars
            }

        # initialize the diffusion PHD filter network
        detection_probabilities = {
            radar.id: radar.detection_probability for radar in radars
        }
        local_clutter_intensities_per_unit = {
            radar.id: radar.clutter_intensity_per_unit for radar in radars
        }
        fovs = {radar.id: (radar.center, radar.radius) for radar in radars}

        self.diff_phd: DiffPHD = DiffPHD(
            graph=graph,
            fovs=fovs,
            adapt=adapt,
            self_referencing_adapt=self_referencing_adapt,
            combine=combine,
            combine_strategy=combine_strategy,
            share_components=share_components,
            model=model,
            birth_intensity=birth_intensity,
            sensor_birth_intensities=sensor_birth_intensities,
            survival_probability=survival_probability,
            detection_probabilities=detection_probabilities,
            global_clutter_intensity_per_unit=global_clutter_intensity_per_unit,
            local_clutter_intensities_per_unit=local_clutter_intensities_per_unit,
            estimate_threshold=estimate_threshold,
            pruning_threshold=pruning_threshold,
            merging_threshold=merging_threshold,
            max_gaussians=max_gaussians,
        )

        # initialize the animator
        self.animator = Animator(
            self.radar_network,
            self.diff_phd,
            animation_save,
            animation_fps,
            animation_loop,
            animation_figsize,
        )

        self.animation = None

    def copy(self):
        """
        Create a deep copy of the simulator.

        :return: The deep copy of the simulator.
        """
        one_phd = list(self.diff_phd.phds.values())[0]
        return Simulator(
            radars=self.radar_network.radars,
            graph=self.diff_phd.graph,
            adapt=self.diff_phd.adapt,
            combine=self.diff_phd.combine,
            combine_strategy=self.diff_phd.combine_strategy,
            share_components=self.diff_phd.share_components,
            model=self.radar_network.model,
            birth_intensity=self.radar_network.birth_intensity,
            survival_probability=self.radar_network.survival_probability,
            global_clutter_intensity_per_unit=self.radar_network.global_clutter_intensity_per_unit,
            estimate_threshold=one_phd.estimate_threshold,
            pruning_threshold=one_phd.pruning_threshold,
            merging_threshold=one_phd.merging_threshold,
            max_gaussians=one_phd.max_gaussians,
            animation_save=self.animator.save,
            animation_fps=self.animator.fps,
            animation_loop=self.animator.loop,
            animation_figsize=self.animator.figsize,
            seed=self.seed,
            self_referencing_adapt=self.diff_phd.phds[
                self.diff_phd.radar_ids[0]
            ].self_referencing_adapt,
        )

    def step(self):
        """
        Perform a single step of the simulation.
        """
        self.seed += 1
        np.random.seed(self.seed)

        measurements = self.radar_network.scan()
        self.diff_phd.step(measurements)

    def simulate(
        self,
        steps: int,
        wait_for_birth: bool = False,
        animate: bool = True,
        show_animation: bool = True,
        calculate_gospa: bool = False,
        gospa_c: float = 3,
        gospa_alpha: float = 2,
        gospa_p: float = 2,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Simulate the filtering process for a number of steps.

        :param steps: The number of steps to simulate.
        :param wait_for_birth: Guarantee that the first target is born in the first
            step.
        :param animate: Animate the simulation.
        :param show_animation: Show the animation.
        :param calculate_gospa: Calculate the GOSPA metric.
        :param gospa_c: The GOSPA c parameter.
        :param gospa_alpha: The GOSPA alpha parameter.
        :param gospa_p: The GOSPA p parameter.
        :return gospa_total: The total GOSPA metric.
        :return gospa_localization: The localization GOSPA metric.
        :return gospa_missed: The missed target GOSPA metric.
        :return gospa_false: The false target GOSPA metric.
        """
        # wait for the first target to be born
        if wait_for_birth:
            while not self.radar_network.targets:
                self.step()

        gospa_total: list[float] = []
        gospa_localization: list[float] = []
        gospa_missed: list[float] = []
        gospa_false: list[float] = []

        for _ in range(steps):
            self.step()
            if animate:
                self.animator.snap()
            if calculate_gospa:
                # calculate the GOSPA metric for each radar
                gospa_total_radars = []
                gospa_localization_radars = []
                gospa_missed_radars = []
                gospa_false_radars = []

                actual_targets = self.radar_network.targets

                for phd in self.diff_phd.phds.values():
                    actual_targets_in_fov = [
                        target for target in actual_targets if phd.is_in_fov(target[:2])
                    ]

                    estimated_targets = phd.estimates.means

                    (
                        gospa_total_radar,
                        _,  # target to track assignments
                        gospa_localization_radar,
                        gospa_missed_radar,
                        gospa_false_radar,
                    ) = gospa(
                        actual_targets_in_fov,
                        estimated_targets,
                        c=gospa_c,
                        alpha=gospa_alpha,
                        p=gospa_p,
                    )

                    gospa_total_radars.append(gospa_total_radar)
                    gospa_localization_radars.append(gospa_localization_radar)
                    gospa_missed_radars.append(gospa_missed_radar)
                    gospa_false_radars.append(gospa_false_radar)

                gospa_total.append(np.mean(gospa_total_radars))
                gospa_localization.append(np.mean(gospa_localization_radars))
                gospa_missed.append(np.mean(gospa_missed_radars))
                gospa_false.append(np.mean(gospa_false_radars))

        if animate:
            self.animation = self.animator.animate(show=show_animation)

        return gospa_total, gospa_localization, gospa_missed, gospa_false
